/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   engine.cu
 * @date   2026-10-05
 *
 * @brief Batched Newton-Raphson engine (block B8). See engine.hpp.
 */

#include "engine.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <deque>
#include <sstream>
#include <utility>

#include "engine_internal.cuh"
#include "profiler.hpp"

namespace gridpack {
namespace batchpf {

Engine::Engine(ModelHost model, EngineConfig config, Logger logger)
    : p_model(std::move(model)), p_config(std::move(config)),
      p_log(logger), p_buf(new EngineBuffers)
{
  p_diag.struct_size = sizeof(p_diag);
  p_diag.struct_version = 1;
}

Engine::~Engine() = default;

BackendCaps Engine::backendCaps() const
{
  return p_backend ? p_backend->caps() : BackendCaps();
}

// -------------------------------------------------------------------------
// Batch execution
// -------------------------------------------------------------------------

void Engine::run(const batchpf_batch &batch, batchpf_results &results)
{
  if (!p_backend) throw Error(BATCHPF_ERR_STATE, "engine used before plan/allocate");
  if (batch.struct_size < sizeof(batchpf_batch) || batch.n_cases < 0 ||
      (batch.n_cases > 0 && batch.cases == nullptr)) {
    throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "invalid batch record");
  }
  if (results.struct_size < sizeof(batchpf_results) ||
      results.n_cases < batch.n_cases || results.n_bus != p_model.n_bus ||
      results.outcomes == nullptr || results.v == nullptr ||
      results.theta == nullptr) {
    throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "invalid results record");
  }
  const ProfilerRange range(p_config.profiler_ranges, "batchpf batch");
  const auto t0 = std::chrono::steady_clock::now();
  runImpl(batch, results);
  const auto t1 = std::chrono::steady_clock::now();
  p_diag.seconds_total += std::chrono::duration<double>(t1 - t0).count();
  p_diag.batches++;
}

void Engine::runImpl(const batchpf_batch &batch, batchpf_results &results)
{
  std::deque<int> queue;
  for (int c = 0; c < batch.n_cases; c++) queue.push_back(c);
  for (SlotState &s : p_slots) s = SlotState();
  while (true) {
    // Fill free slots: always with backfill, else only between waves
    bool any_busy = false;
    for (const SlotState &s : p_slots) any_busy = any_busy || s.stage != SlotState::Stage::Free;
    if (!queue.empty() && (p_config.backfill || !any_busy)) {
      std::vector<int> slots, cases;
      for (int b = 0; b < p_B && !queue.empty(); b++) {
        if (p_slots[b].stage == SlotState::Stage::Free) {
          slots.push_back(b);
          cases.push_back(queue.front());
          queue.pop_front();
        }
      }
      if (!slots.empty()) fillSlots(batch, slots, cases);
      any_busy = any_busy || !slots.empty();
    }
    if (!any_busy) break;
    step();
    std::vector<int> done;
    for (int b = 0; b < p_B; b++) {
      SlotState &s = p_slots[b];
      if (s.stage == SlotState::Stage::Free) continue;
      advanceSlot(s, p_results[b], p_rules);
      if (s.stage == SlotState::Stage::Done) done.push_back(b);
    }
    if (!done.empty()) finishSlots(batch, results, done);
  }
}

void Engine::fillSlots(const batchpf_batch &batch, const std::vector<int> &slots,
                       const std::vector<int> &cases)
{
  EngineBuffers &d = *p_buf;
  const bool dev = p_config.on_device;
  const cudaStream_t st = p_config.stream;
  std::fill(d.h_fill.begin(), d.h_fill.end(), 0);
  std::size_t nb = 0, ne = 0;
  for (std::size_t i = 0; i < slots.size(); i++) {
    const batchpf_case &c = batch.cases[cases[i]];
    if (c.n_bus_updates < 0 || c.n_edge_updates < 0 ||
        (c.n_bus_updates > 0 && c.bus_updates == nullptr) ||
        (c.n_edge_updates > 0 && c.edge_updates == nullptr) ||
        c.slack_bus < 0 || c.slack_bus >= p_model.n_bus) {
      throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "invalid case record");
    }
    nb += static_cast<std::size_t>(c.n_bus_updates);
    ne += static_cast<std::size_t>(c.n_edge_updates);
  }
  d.u_bus_member.reserve(nb, dev, p_config.exchange_pinned);
  d.u_bus.reserve(nb, dev, p_config.exchange_pinned);
  d.u_edge_member.reserve(ne, dev, p_config.exchange_pinned);
  d.u_edge.reserve(ne, dev, p_config.exchange_pinned);
  std::size_t ib = 0, ie = 0;
  for (std::size_t i = 0; i < slots.size(); i++) {
    const int b = slots[i];
    const batchpf_case &c = batch.cases[cases[i]];
    d.h_fill[b] = 1;
    d.h_slack[b] = c.slack_bus;
    for (int k = 0; k < c.n_bus_updates; k++) {
      const batchpf_bus_update &r = c.bus_updates[k];
      if (r.bus < 0 || r.bus >= p_model.n_bus) {
        throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "bus update out of range");
      }
      d.u_bus_member.host()[ib] = b;
      d.u_bus.host()[ib++] = r;
    }
    for (int k = 0; k < c.n_edge_updates; k++) {
      const batchpf_edge_update &r = c.edge_updates[k];
      if (r.edge < 0 || r.edge >= p_model.n_edge) {
        throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "edge update out of range");
      }
      d.u_edge_member.host()[ie] = b;
      d.u_edge.host()[ie++] = r;
    }
    startSlot(p_slots[b], cases[i]);
  }
  const Executor ex(dev, st, p_config.threads_per_block);
  if (dev) cudaCheck(cudaEventRecord(d.ev_start[PH_MATERIALIZE].get(), st), "event");
  const auto t0 = std::chrono::steady_clock::now();
  d.u_bus_member.toDevice(nb, st);
  d.u_bus.toDevice(nb, st);
  d.u_edge_member.toDevice(ne, st);
  d.u_edge.toDevice(ne, st);
  d.m_fill.upload(d.h_fill.data(), p_B, st);
  d.m_slack.upload(d.h_slack.data(), p_B, st);

  const ModelView m = modelView(p_model, d);
  const BatchView w = batchView(d, p_B, p_params);
  UpdateView u;
  u.n_bus_updates = static_cast<int>(nb);
  u.n_edge_updates = static_cast<int>(ne);
  u.bus_member = d.u_bus_member.device();
  u.bus = d.u_bus.device();
  u.edge_member = d.u_edge_member.device();
  u.edge = d.u_edge.device();
  const int nfill = static_cast<int>(slots.size());
  std::copy(slots.begin(), slots.end(), d.fill_slots.host());
  d.fill_slots.toDevice(slots.size(), st);
  const int *fs = d.fill_slots.device();
  ex.run(static_cast<int64_t>(p_model.n_bus) * nfill, FillBusBase{m, w, fs, nfill},
         "FillBusBase");
  ex.run(static_cast<int64_t>(p_model.n_edge) * nfill, FillEdgeBase{m, w, fs, nfill},
         "FillEdgeBase");
  ex.run(static_cast<int64_t>(nb), ApplyBusUpdates{w, u}, "ApplyBusUpdates");
  ex.run(static_cast<int64_t>(ne), ApplyEdgeUpdates{w, u}, "ApplyEdgeUpdates");
  ex.run(static_cast<int64_t>(p_model.n_bus) * nfill, InitState{m, w, fs, nfill},
         "InitState");
  if (dev) {
    cudaCheck(cudaEventRecord(d.ev_stop[PH_MATERIALIZE].get(), st), "event");
  } else {
    d.host_seconds[PH_MATERIALIZE] += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
  }
  d.used[PH_MATERIALIZE] = true;
}

double Engine::elapsed(int phase) const
{
  const EngineBuffers &d = *p_buf;
  if (!d.used[phase]) return 0.0;
  if (!p_config.on_device) return 0.0;   // host phases are added directly
  float ms = 0.0f;
  cudaCheck(cudaEventElapsedTime(&ms, d.ev_start[phase].get(),
                                 d.ev_stop[phase].get()),
            "cudaEventElapsedTime");
  return ms * 1.0e-3;
}

void Engine::step()
{
  EngineBuffers &d = *p_buf;
  const bool dev = p_config.on_device;
  const cudaStream_t st = p_config.stream;
  const Executor ex(dev, st, p_config.threads_per_block);
  const int B = p_B;
  bool any_apply = false, any_q = false, any_eval = false;
  for (int b = 0; b < B; b++) {
    const SlotState &s = p_slots[b];
    d.h_apply[b] = s.act_apply ? 1 : 0;
    d.h_qcheck[b] = s.act_qcheck ? 1 : 0;
    d.h_eval[b] = s.act_eval ? 1 : 0;
    any_apply = any_apply || s.act_apply;
    any_q = any_q || s.act_qcheck;
    any_eval = any_eval || s.act_eval;
  }
  std::fill(d.h_maxp.begin(), d.h_maxp.end(), 0ULL);
  std::fill(d.h_maxq.begin(), d.h_maxq.end(), 0ULL);
  std::fill(d.h_res.begin(), d.h_res.end(), 0ULL);
  std::fill(d.h_rhs.begin(), d.h_rhs.end(), 0ULL);
  std::fill(d.h_argp.begin(), d.h_argp.end(), INT_MAX);
  std::fill(d.h_argq.begin(), d.h_argq.end(), INT_MAX);
  std::fill(d.h_qviol.begin(), d.h_qviol.end(), 0);
  std::fill(d.h_status.begin(), d.h_status.end(), 0);
  d.m_apply.upload(d.h_apply.data(), B, st);
  d.m_qcheck.upload(d.h_qcheck.data(), B, st);
  d.m_eval.upload(d.h_eval.data(), B, st);
  d.m_maxp.upload(d.h_maxp.data(), B, st);
  d.m_maxq.upload(d.h_maxq.data(), B, st);
  d.m_res.upload(d.h_res.data(), B, st);
  d.m_rhs.upload(d.h_rhs.data(), B, st);
  d.m_argp.upload(d.h_argp.data(), B, st);
  d.m_argq.upload(d.h_argq.data(), B, st);
  d.m_qviol.upload(d.h_qviol.data(), B, st);
  d.m_status.upload(d.h_status.data(), B, st);

  const ModelView m = modelView(p_model, d);
  const BatchView w = batchView(d, B, p_params);
  const int64_t nB = static_cast<int64_t>(p_model.n_bus) * B;
  const int64_t eB = static_cast<int64_t>(p_model.n_edge) * B;

  // Each phase is bracketed by events (GPU) or a clock (CPU)
  auto phase = [&](int ph, auto &&body) {
    const auto t0 = std::chrono::steady_clock::now();
    if (dev) cudaCheck(cudaEventRecord(d.ev_start[ph].get(), st), "event");
    body();
    if (dev) {
      cudaCheck(cudaEventRecord(d.ev_stop[ph].get(), st), "event");
    } else {
      d.host_seconds[ph] += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - t0).count();
    }
    d.used[ph] = true;
  };
  for (int ph = 0; ph < PH_COUNT; ph++) {
    if (ph != PH_MATERIALIZE) d.used[ph] = false;
  }

  if (any_apply) phase(PH_UPDATE, [&] { ex.run(nB, ApplyStep{w}, "ApplyStep"); });
  if (any_q) phase(PH_QLIM, [&] { ex.run(nB, QlimCheck{m, w}, "QlimCheck"); });
  if (any_eval) {
    phase(PH_MISMATCH, [&] {
      ex.run(nB, Mismatch{m, w}, "Mismatch");
      ex.run(nB, MismatchArgmax{w}, "MismatchArgmax");
    });
    phase(PH_JACOBIAN, [&] {
      ex.run(nB, JacobianDiag{m, w}, "JacobianDiag");
      ex.run(eB, JacobianEdge{m, w, d.edge_row.data()}, "JacobianEdge");
    });
    phase(PH_FACTOR, [&] {
      p_backend->refactorize(d.J.data(), d.m_eval.data(), d.m_status.data());
    });
    phase(PH_SOLVE, [&] {
      p_backend->solve(d.F.data(), d.X.data(), d.m_eval.data(), d.m_status.data());
      if (p_config.residual_limit > 0.0) {
        ex.run(static_cast<int64_t>(p_pattern.n_rows) * B,
               Residual{w, d.jrow_ptr.data(), d.jcol_idx.data(),
                        d.m_res.data(), d.m_rhs.data()},
               "Residual");
      }
      if (p_params.damping_factor < 1.0) {
        ex.run(static_cast<int64_t>(p_pattern.n_rows) * B, ScaleStep{w}, "ScaleStep");
      }
    });
  }
  d.m_maxp.download(d.h_maxp.data(), B, st);
  d.m_maxq.download(d.h_maxq.data(), B, st);
  d.m_argp.download(d.h_argp.data(), B, st);
  d.m_argq.download(d.h_argq.data(), B, st);
  d.m_qviol.download(d.h_qviol.data(), B, st);
  d.m_status.download(d.h_status.data(), B, st);
  d.m_res.download(d.h_res.data(), B, st);
  d.m_rhs.download(d.h_rhs.data(), B, st);
  if (dev) cudaCheck(cudaStreamSynchronize(st), "step");

  // telemetry totals
  if (dev && p_config.telemetry != BATCHPF_TELEMETRY_OFF) {
    p_diag.seconds_materialize += elapsed(PH_MATERIALIZE);
    p_diag.seconds_update += elapsed(PH_UPDATE);
    p_diag.seconds_qlim += elapsed(PH_QLIM);
    p_diag.seconds_mismatch += elapsed(PH_MISMATCH);
    p_diag.seconds_jacobian += elapsed(PH_JACOBIAN);
    p_diag.seconds_factor += elapsed(PH_FACTOR);
    p_diag.seconds_solve += elapsed(PH_SOLVE);
  } else if (!dev) {
    p_diag.seconds_materialize += d.host_seconds[PH_MATERIALIZE];
    p_diag.seconds_update += d.host_seconds[PH_UPDATE];
    p_diag.seconds_qlim += d.host_seconds[PH_QLIM];
    p_diag.seconds_mismatch += d.host_seconds[PH_MISMATCH];
    p_diag.seconds_jacobian += d.host_seconds[PH_JACOBIAN];
    p_diag.seconds_factor += d.host_seconds[PH_FACTOR];
    p_diag.seconds_solve += d.host_seconds[PH_SOLVE];
    std::fill(d.host_seconds.begin(), d.host_seconds.end(), 0.0);
  }
  d.used[PH_MATERIALIZE] = false;
  if (any_eval) {
    int active = 0;
    for (int b = 0; b < B; b++) active += d.h_eval[b];
    p_diag.newton_steps += active;
    p_diag.slot_steps += B;
    const double nnz = static_cast<double>(p_pattern.nnz);
    const double lu = static_cast<double>(p_lu.nnz);
    const double fl = static_cast<double>(p_lu.flops);
    const double nb = p_model.n_bus, ne = p_model.n_edge;
    p_diag.bytes_factor += active * 8.0 * (nnz + lu + 3.0 * fl);
    p_diag.bytes_solve += active * 8.0 * (lu + 4.0 * p_pattern.n_rows);
    p_diag.bytes_jacobian += active * 8.0 * (nnz + 6.0 * nb + 6.0 * ne);
    p_diag.bytes_mismatch += active * 8.0 * (12.0 * nb + 4.0 * ne);
  }
  for (int b = 0; b < B; b++) {
    StepResult &r = p_results[b];
    r.maxp = bitsToDouble(d.h_maxp[b]);
    r.maxq = bitsToDouble(d.h_maxq[b]);
    r.argp = (d.h_argp[b] == INT_MAX) ? -1 : d.h_argp[b];
    r.argq = (d.h_argq[b] == INT_MAX) ? -1 : d.h_argq[b];
    r.qviol = d.h_qviol[b];
    r.member_status = d.h_status[b];
    r.residual = bitsToDouble(d.h_res[b]);
    r.rhs_norm = bitsToDouble(d.h_rhs[b]);
  }
}

void Engine::finishSlots(const batchpf_batch &batch, batchpf_results &results,
                         const std::vector<int> &slots)
{
  EngineBuffers &d = *p_buf;
  const bool dev = p_config.on_device;
  const cudaStream_t st = p_config.stream;
  const int n = p_model.n_bus;
  const int cnt = static_cast<int>(slots.size());
  const auto t0 = std::chrono::steady_clock::now();
  std::copy(slots.begin(), slots.end(), d.gather_slots.host());
  d.gather_slots.toDevice(cnt, st);
  const BatchView w = batchView(d, p_B, p_params);
  const Executor ex(dev, st, p_config.threads_per_block);
  ex.run(static_cast<int64_t>(n) * cnt,
         GatherState{w, n, d.gather_slots.device(), d.out_v.deviceMutable(),
                     d.out_theta.deviceMutable(), d.out_conv.deviceMutable(),
                     d.out_q.deviceMutable()},
         "GatherState");
  const std::size_t len = static_cast<std::size_t>(n) * cnt;
  d.out_v.toHost(len, st);
  d.out_theta.toHost(len, st);
  d.out_conv.toHost(len, st);
  d.out_q.toHost(len, st);
  if (dev) cudaCheck(cudaStreamSynchronize(st), "gather");
  for (int li = 0; li < cnt; li++) {
    SlotState &s = p_slots[slots[li]];
    const int c = s.case_idx;
    const std::size_t off = static_cast<std::size_t>(c) * n;
    const std::size_t src = static_cast<std::size_t>(li) * n;
    std::copy(d.out_v.host() + src, d.out_v.host() + src + n, results.v + off);
    std::copy(d.out_theta.host() + src, d.out_theta.host() + src + n, results.theta + off);
    if (results.qlim_conversion) {
      std::copy(d.out_conv.host() + src, d.out_conv.host() + src + n,
                results.qlim_conversion + off);
    }
    if (results.q_required) {
      std::copy(d.out_q.host() + src, d.out_q.host() + src + n, results.q_required + off);
    }
    if (results.history && results.history_count && results.history_capacity > 0) {
      const int cap = results.history_capacity;
      const int cnt = std::min<int>(cap, static_cast<int>(s.history.size()));
      batchpf_mismatch_record *h = results.history + static_cast<std::size_t>(c) * cap;
      for (int k = 0; k < cnt; k++) {
        const IterationRecord &rec = s.history[s.history.size() - cnt + k];
        h[k].max_p_bus = (rec.maxp > 0.0) ? rec.argp : -1;
        h[k].max_q_bus = (rec.maxq > 0.0) ? rec.argq : -1;
        h[k].max_p_mismatch = rec.maxp * p_model.sbase;
        h[k].max_q_mismatch = rec.maxq * p_model.sbase;
      }
      results.history_count[c] = cnt;
    }
    batchpf_outcome &o = results.outcomes[c];
    o.case_id = batch.cases[c].case_id;
    o.status = s.status;
    o.health_events = s.health;
    o.iterations = s.rec_iter;
    o.total_iterations = s.total_iters;
    o.controller_iterations = s.ctrl_total;
    o.solves = s.solve_no;
    o.pv_to_pq = s.pv_to_pq;
    o.final_tolerance = s.rec_tol;
    o.max_p_bus = (s.rec_maxp > 0.0) ? s.rec_argp : -1;
    o.max_q_bus = (s.rec_maxq > 0.0) ? s.rec_argq : -1;
    o.max_p_mismatch = s.rec_maxp * p_model.sbase;
    o.max_q_mismatch = s.rec_maxq * p_model.sbase;
    o.reserved = 0;
    p_diag.cases++;
    if (s.status == BATCHPF_CASE_CONVERGED) p_diag.converged++;
    if (s.status == BATCHPF_CASE_DIVERGED) p_diag.diverged++;
    if (s.status == BATCHPF_CASE_FLAGGED) p_diag.flagged++;
    s = SlotState();
  }
  p_diag.seconds_exchange += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
}

}  // namespace batchpf
}  // namespace gridpack
