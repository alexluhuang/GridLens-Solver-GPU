/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   session.cpp
 * @date   2026-10-05
 *
 * @brief Accelerator session: device setup, backend and batch-size choice,
 * worker thread. See session.hpp.
 */

#include "session.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>

#include "platform.hpp"

namespace gridpack {
namespace batchpf {

namespace {

const char *backendName(int b)
{
  switch (b) {
    case BATCHPF_BACKEND_CUDSS: return "cudss";
    case BATCHPF_BACKEND_ALG2: return "alg2";
    case BATCHPF_BACKEND_CPU_REFERENCE: return "cpu_reference";
    default: return "auto";
  }
}

/// Validated batch cap of a backend: the setting wins, then the backend's
/// own record (0 = none)
int validatedCap(const batchpf_settings &s, const BackendCaps &caps)
{
  if (s.max_validated_batch > 0) return s.max_validated_batch;
  return caps.validated_batch;
}

}  // namespace

Session::Session(const batchpf_settings &settings)
    : p_settings(settings),
      p_plugin_dir(settings.plugin_dir ? settings.plugin_dir : ""),
      p_log(settings.log_fn, settings.log_user, settings.log_level),
      p_on_device(settings.backend != BATCHPF_BACKEND_CPU_REFERENCE)
{
  p_settings.plugin_dir = nullptr;   // use the owned copy only
  p_info.struct_size = sizeof(p_info);
  std::string why;
  const batchpf_status st = probeDevice(settings.device, &p_info, &why);
  if (p_on_device) {
    if (st != BATCHPF_OK) throw Error(st, why);
    verifyKernelsLoad();
    p_profile = selectMemoryProfile(p_info, settings.memory_profile);
    p_info.memory_profile = p_profile;
    cudaCheck(cudaSetDevice(settings.device), "cudaSetDevice");
    p_stream = Stream(true);
    p_log.info(memoryPoolsSupported() ? "allocation: stream-ordered device pools"
                                     : "allocation: legacy device buffers (pools unsupported)");
  } else {
    // The CPU reference backend needs no GPU; probe only for the log
    p_profile = BATCHPF_MEMORY_AUTO;
    p_info.memory_profile = p_profile;
    if (st != BATCHPF_OK) cudaGetLastError();
  }
  p_worker = std::thread(&Session::workerLoop, this);
}

Session::~Session()
{
  {
    std::lock_guard<std::mutex> lock(p_mutex);
    p_stop = true;
  }
  p_cv.notify_all();
  if (p_worker.joinable()) p_worker.join();
}

batchpf_device_info Session::deviceInfo() const
{
  std::lock_guard<std::mutex> lock(p_mutex);
  return p_info;
}

void Session::setModel(const batchpf_model &model)
{
  p_model = std::make_unique<ModelHost>(copyModel(model));
  std::ostringstream os;
  os << "model: " << p_model->n_bus << " buses, " << p_model->n_edge
     << " directed branch entries, base " << p_model->sbase << " MVA";
  p_log.info(os.str());
}

int Session::chooseBackend(int requested)
{
  if (requested == BATCHPF_BACKEND_CPU_REFERENCE) return requested;
  if (requested == BATCHPF_BACKEND_ALG2) return requested;
  BackendCaps caps;
  std::string why;
  const bool have_cudss = probePluginBackend(p_plugin_dir, "cudss", &caps, &why);
  if (requested == BATCHPF_BACKEND_CUDSS) {
    if (!have_cudss) {
      throw Error(BATCHPF_ERR_UNAVAILABLE, "GPUBatch/backend=cudss but " + why);
    }
    return requested;
  }
  if (!have_cudss) {
    p_log.info("cuDSS backend not available (" + why + "); using alg2");
    return BATCHPF_BACKEND_ALG2;
  }
  return BATCHPF_BACKEND_CUDSS;
}

/**
 * Batch size (guide 8.7): within the memory budget, within the validated
 * cap of the backend, no more than the cases there are, and, when automatic,
 * the smallest size within 5% of the best per-case time found by timing
 * the reference factorization at increasing sizes (PERF-3).
 */
int Session::chooseCapacity(int64_t expected_cases)
{
  const int64_t expected = std::max<int64_t>(expected_cases, 1);
  if (!p_on_device) {
    int b = p_settings.batch_size > 0 ? p_settings.batch_size
                                            : static_cast<int>(std::min<int64_t>(expected, 64));
    if (p_settings.max_validated_batch > 0) b = std::min(b, p_settings.max_validated_batch);
    p_engine->allocate(b);
    return b;
  }
  const double per_slot = p_engine->bytesPerSlot();
  const double shared = p_engine->sharedBytes();
  const double budget = memoryBudget(p_info, p_profile, p_settings);
  p_info.budget_bytes = budget;
  const double fit = std::floor((budget - shared) / per_slot);
  if (fit < 1.0) {
    std::ostringstream os;
    os << "not enough memory for one batch member: budget " << budget / 1e9
       << " GB, shared " << shared / 1e9 << " GB, per member " << per_slot / 1e9
       << " GB (GPUBatch/memoryHeadroomGB, GPUBatch/maxMemoryGB)";
    throw Error(BATCHPF_ERR_OUT_OF_MEMORY, os.str());
  }
  const int mem_cap = static_cast<int>(std::min(fit, 65536.0));
  int cudss_cap = 0;
  if (p_backend == BATCHPF_BACKEND_CUDSS) {
    BackendCaps caps;
    std::string why;
    if (probePluginBackend(p_plugin_dir, "cudss", &caps, &why)) {
      cudss_cap = validatedCap(p_settings, caps);
    }
  }
  const int alg2_cap = p_settings.max_validated_batch > 0
                           ? p_settings.max_validated_batch : kAlg2ValidatedBatch;
  auto capOf = [&](int backend) {
    int c = mem_cap;
    if (backend == BATCHPF_BACKEND_CUDSS && cudss_cap > 0) c = std::min(c, cudss_cap);
    if (backend == BATCHPF_BACKEND_ALG2 && alg2_cap > 0) c = std::min(c, alg2_cap);
    return c;
  };

  if (p_settings.batch_size > 0) {
    int b = p_settings.batch_size;
    if (p_backend == BATCHPF_BACKEND_CUDSS && cudss_cap > 0 && b > cudss_cap) {
      if (p_settings.backend == BATCHPF_BACKEND_AUTO) {
        p_log.info("batch size " + std::to_string(b) + " is above the validated "
                   "cuDSS cap " + std::to_string(cudss_cap) + "; using alg2");
        p_backend = BATCHPF_BACKEND_ALG2;
      } else {
        p_log.warn("batch size " + std::to_string(b) + " lowered to the validated "
                   "cuDSS cap " + std::to_string(cudss_cap));
        b = cudss_cap;
      }
    }
    if (p_backend == BATCHPF_BACKEND_ALG2 && b > alg2_cap) {
      p_log.warn("batch size " + std::to_string(b) + " lowered to the validated "
                 "alg2 cap " + std::to_string(alg2_cap));
      b = alg2_cap;
    }
    if (b > mem_cap) {
      p_log.warn("batch size " + std::to_string(b) + " lowered to " +
                 std::to_string(mem_cap) + " to stay within the memory budget");
      b = mem_cap;
    }
    p_engine->setBackend(p_backend);
    p_engine->allocate(b);
    return b;
  }

  // Automatic: sweep sizes, and backends when cuDSS is a candidate
  std::vector<int> backends{p_backend};
  if (p_backend == BATCHPF_BACKEND_CUDSS && p_settings.backend == BATCHPF_BACKEND_AUTO) {
    backends.push_back(BATCHPF_BACKEND_ALG2);
  }
  // With few cases, one wave (B = cases) is best. With many, keep B at no
  // more than a quarter of them so refilled slots keep the batch busy and
  // the slowest cases of the last wave are a small part of the run.
  const int64_t quarter = p_settings.backfill ? expected / 4 : expected;
  const int64_t want = std::max<int64_t>(
      32, ((std::max<int64_t>(quarter, std::min<int64_t>(expected, 128)) + 31) / 32) * 32);
  struct Trial { int backend; int B; double t; };
  std::vector<Trial> trials;
  for (int backend : backends) {
    const int cap = static_cast<int>(std::min<int64_t>(capOf(backend), want));
    std::vector<int> sizes;
    for (int b = 32; b <= 4096; b *= 2) {
      if (b <= cap) sizes.push_back(b);
    }
    if (sizes.empty()) sizes.push_back(std::max(1, cap));
    double prev = 0.0;
    for (int b : sizes) {
      try {
        p_engine->setBackend(backend);
        p_engine->allocate(b);
        const double t = p_engine->timeReferenceSolve(2);
        trials.push_back({backend, b, t});
        std::ostringstream os;
        os << "batch-size sweep: " << backendName(backend) << " B=" << b
           << ": " << t * 1e6 << " us per member (factor + solve)";
        p_log.info(os.str());
        // Stop once doubling the batch gains less than 5% per member
        if (prev > 0.0 && t > 0.95 * prev) break;
        prev = t;
      } catch (const Error &e) {
        p_log.warn(std::string("batch-size sweep: ") + backendName(backend) +
                   " B=" + std::to_string(b) + " failed: " + e.what());
        if (e.code() == BATCHPF_ERR_OUT_OF_MEMORY) break;
      }
    }
  }
  if (trials.empty()) {
    throw Error(BATCHPF_ERR_BACKEND, "no batch size could be set up");
  }
  double best = trials.front().t;
  int best_backend = trials.front().backend;
  for (const Trial &t : trials) {
    if (t.t < best) {
      best = t.t;
      best_backend = t.backend;
    }
  }
  int chosen = 0;
  for (const Trial &t : trials) {
    if (t.backend == best_backend && t.t <= 1.05 * best) {
      chosen = (chosen == 0) ? t.B : std::min(chosen, t.B);
    }
  }
  p_backend = best_backend;
  p_engine->setBackend(p_backend);
  p_engine->allocate(chosen);
  return chosen;
}

int Session::plan(const batchpf_solver_params &params, int64_t expected_cases)
{
  if (!p_model) throw Error(BATCHPF_ERR_STATE, "plan() called before set_model()");
  if (params.struct_size < sizeof(batchpf_solver_params)) {
    throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "invalid solver parameter record");
  }
  if (p_on_device) cudaCheck(cudaSetDevice(p_settings.device), "cudaSetDevice");
  const auto t0 = std::chrono::steady_clock::now();
  p_backend = chooseBackend(p_settings.backend);
  EngineConfig cfg;
  cfg.on_device = p_on_device;
  cfg.stream = p_stream.get();
  cfg.device = p_settings.device;
  cfg.threads_per_block = p_settings.threads_per_block;
  cfg.backend = p_backend;
  cfg.plugin_dir = p_plugin_dir;
  cfg.ordering = (p_settings.planner_ordering == BATCHPF_ORDERING_COLAMD) ? 1 : 0;
  cfg.pivot_tolerance = p_settings.pivot_tolerance;
  cfg.refinement_steps = p_settings.refinement_steps;
  cfg.residual_limit = p_settings.health_residual_limit;
  cfg.pivot_limit = p_settings.health_pivot_limit;
  cfg.check_nonfinite = p_settings.health_check_nonfinite != 0;
  cfg.backfill = p_settings.backfill != 0;
  cfg.host_solve = p_settings.solve_placement == BATCHPF_SOLVE_HOST;
  cfg.exchange_pinned = p_profile != BATCHPF_MEMORY_DISCRETE;
  cfg.telemetry = p_settings.telemetry;
  cfg.profiler_ranges = p_settings.profiler_ranges != 0;
  p_engine = std::make_unique<Engine>(*p_model, cfg, p_log);
  p_engine->plan(params);
  p_capacity = chooseCapacity(expected_cases);
  const BackendCaps caps = p_engine->backendCaps();
  {
    std::lock_guard<std::mutex> lock(p_mutex);
    p_info.backend = p_backend;
    p_info.batch_size = p_capacity;
    p_info.threads_per_block = p_settings.threads_per_block;
    copyMessage(caps.version, std::begin(p_info.backend_version),
                sizeof(p_info.backend_version));
  }
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  std::ostringstream os;
  os << "backend " << backendName(p_backend) << " (" << caps.version
     << "), batch size " << p_capacity << ", memory profile "
     << memoryProfileName(p_profile) << ", setup " << secs << " s";
  p_log.info(os.str());
  return p_capacity;
}

int64_t Session::submit(const batchpf_batch &batch, batchpf_results &results)
{
  if (!p_engine) throw Error(BATCHPF_ERR_STATE, "submit() called before plan()");
  std::lock_guard<std::mutex> lock(p_mutex);
  Job job;
  job.ticket = p_next_ticket++;
  job.batch = &batch;
  job.results = &results;
  p_queue.push_back(job);
  p_cv.notify_all();
  return job.ticket;
}

batchpf_status Session::wait(int64_t ticket, int timeout_ms)
{
  std::unique_lock<std::mutex> lock(p_mutex);
  auto ready = [&] { return p_done.count(ticket) > 0; };
  if (timeout_ms < 0) {
    p_cv.wait(lock, ready);
  } else if (!p_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready)) {
    return BATCHPF_ERR_TIMEOUT;
  }
  const Done d = p_done[ticket];
  p_done.erase(ticket);
  if (d.status != BATCHPF_OK) p_last_error = d.message;
  return d.status;
}

batchpf_diagnostics Session::diagnostics() const
{
  std::lock_guard<std::mutex> lock(p_mutex);
  batchpf_diagnostics d{};
  if (p_engine && p_queue.empty()) d = p_engine->diagnostics();
  d.struct_size = sizeof(d);
  d.struct_version = 1;
  d.backend = p_backend;
  d.batch_size = p_capacity;
  return d;
}

std::string Session::lastError() const
{
  std::lock_guard<std::mutex> lock(p_mutex);
  return p_last_error;
}

void Session::setLastError(const std::string &msg)
{
  std::lock_guard<std::mutex> lock(p_mutex);
  p_last_error = msg;
}

void Session::workerLoop()
{
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(p_mutex);
      p_cv.wait(lock, [&] { return p_stop || !p_queue.empty(); });
      if (p_stop && p_queue.empty()) return;
      job = p_queue.front();
    }
    Done done;
    try {
      if (p_on_device) cudaCheck(cudaSetDevice(p_settings.device), "worker cudaSetDevice");
      p_engine->run(*job.batch, *job.results);
    } catch (const Error &e) {
      done.status = e.code();
      done.message = e.what();
    } catch (const std::exception &e) {
      done.status = BATCHPF_ERR_INTERNAL;
      done.message = e.what();
    }
    {
      std::lock_guard<std::mutex> lock(p_mutex);
      p_queue.pop_front();
      p_done[job.ticket] = done;
    }
    p_cv.notify_all();
  }
}

}  // namespace batchpf
}  // namespace gridpack
