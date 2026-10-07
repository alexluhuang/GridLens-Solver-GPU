/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   batch_path.cpp
 * @date   2026-10-05
 *
 * @brief Start-up (scenario R0) and preparation (scenario R1) of the batch
 * path. See batch_path.hpp.
 */

#include "batch_path_impl.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <iterator>
#include <memory>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <string>
#include <vector>
#include <mpi.h>

#include "dc_records.hpp"
#include "gridpack/configuration/configuration.hpp"
#include "gridpack/timer/coarse_timer.hpp"

namespace gridpack {
namespace batchpf {

namespace {

double now()
{
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// Stop the whole run with a message from rank 0 (RT-5)
[[noreturn]] void stopRun(gridpack::parallel::Communicator &world,
                          const std::string &msg)
{
  if (world.rank() == 0) {
    std::cout << "ERROR: " << msg << std::endl;
  }
  world.barrier();
  MPI_Abort(static_cast<MPI_Comm>(world), 1);
  std::abort();
}

/// Append the bytes of a trivially copyable array
template <class T>
void appendBytes(std::vector<char> *out, const T *src, std::size_t count)
{
  static_assert(std::is_trivially_copyable<T>::value, "copied with memcpy");
  const std::size_t at = out->size();
  out->resize(at + count * sizeof(T));
  if (count > 0) std::memcpy(out->data() + at, src, count * sizeof(T));
}

/// Fields of a packed classification record's head, then the record counts
enum HeadField : std::size_t {
  kEvent, kPath, kReason, kFast, kIslands, kLoneBus, kSlackMoved, kSlackBus,
  kBusUpdates, kEdgeUpdates, kDcOff, kHeadSize
};

/// Serialize classification records for the all-gather
void pack(const CaseClass &c, std::vector<char> *out)
{
  const std::array<int32_t, kHeadSize> head = {c.event.value, static_cast<int32_t>(c.path),
                            static_cast<int32_t>(c.reason), c.fast ? 1 : 0,
                            c.island_count, c.lone_bus ? 1 : 0,
                            c.slack_transferred ? 1 : 0, c.slack_bus.value,
                            static_cast<int32_t>(c.bus_updates.size()),
                            static_cast<int32_t>(c.edge_updates.size()),
                            static_cast<int32_t>(c.dc_off.size())};
  appendBytes(out, head.data(), head.size());
  appendBytes(out, c.bus_updates.data(), c.bus_updates.size());
  appendBytes(out, c.edge_updates.data(), c.edge_updates.size());
  appendBytes(out, c.dc_off.data(), c.dc_off.size());
}

std::size_t unpack(const char *p, CaseClass *c)
{
  std::array<int32_t, kHeadSize> head{};
  std::memcpy(head.data(), p, sizeof(head));
  c->event = CaseIndex{head[kEvent]};
  c->path = static_cast<CasePath>(head[kPath]);
  c->reason = static_cast<CpuReason>(head[kReason]);
  c->fast = head[kFast] != 0;
  c->island_count = head[kIslands];
  c->lone_bus = head[kLoneBus] != 0;
  c->slack_transferred = head[kSlackMoved] != 0;
  c->slack_bus = BusIndex{head[kSlackBus]};
  std::size_t off = sizeof(head);
  c->bus_updates.resize(head[kBusUpdates]);
  std::memcpy(c->bus_updates.data(), p + off,
              head[kBusUpdates] * sizeof(batchpf_bus_update));
  off += head[kBusUpdates] * sizeof(batchpf_bus_update);
  c->edge_updates.resize(head[kEdgeUpdates]);
  std::memcpy(c->edge_updates.data(), p + off,
              head[kEdgeUpdates] * sizeof(batchpf_edge_update));
  off += head[kEdgeUpdates] * sizeof(batchpf_edge_update);
  c->dc_off.resize(head[kDcOff]);
  std::memcpy(c->dc_off.data(), p + off, head[kDcOff] * sizeof(int32_t));
  off += head[kDcOff] * sizeof(int32_t);
  return off;
}

}  // namespace

batchpf_model ModelArrays::record() const
{
  batchpf_model m{};
  m.struct_size = sizeof(m);
  m.struct_version = 2;
  m.n_bus = static_cast<int32_t>(bus_type.size());
  m.n_edge = static_cast<int32_t>(edge_col.size());
  m.bus_type = bus_type.data();
  m.g_diag = g.data();
  m.b_diag = b.data();
  m.p0 = p0.data();
  m.q0 = q0.data();
  m.v_init = v_init.data();
  m.theta_init = theta_init.data();
  m.v_base = v_base.data();
  m.theta_base = theta_base.data();
  m.ql = ql.data();
  m.ip = ip.data();
  m.iq = iq.data();
  m.yp = yp.data();
  m.yq = yq.data();
  m.qmax = qmax.data();
  m.qmin = qmin.data();
  m.row_start = row_start.data();
  m.edge_col = edge_col.data();
  m.edge_mate = edge_mate.data();
  m.edge_g = eg.data();
  m.edge_b = eb.data();
  m.dg_q = dg_q.data();
  m.dc_p = dc_p.data();
  m.dc_q = dc_q.data();
  m.n_dc_line = static_cast<int32_t>(dc_lines.size());
  m.dc_lines = dc_lines.empty() ? nullptr : dc_lines.data();
  return m;
}

bool ModelArrays::needsInterface11() const
{
  auto nonzero = [](const std::vector<double> &x) {
    return std::any_of(x.begin(), x.end(), [](double y) { return y != 0.0; });
  };
  return nonzero(dg_q) || nonzero(dc_p) || nonzero(dc_q) || !dc_lines.empty();
}

std::vector<int32_t> BatchPath::Impl::dcStatus(int event) const
{
  std::vector<int32_t> status(model.dc_status.begin(), model.dc_status.end());
  for (const int32_t l : classes[event].dc_off) {
    if (l >= 0 && static_cast<std::size_t>(l) < status.size()) {
      status[static_cast<std::size_t>(l)] = 0;
    }
  }
  return status;
}

batchpf_settings BatchPath::Impl::pluginSettings() const
{
  const GpuBatchSettings &g = settings.gpu;
  batchpf_settings s{};
  s.struct_size = sizeof(s);
  s.struct_version = 1;
  s.backend = g.backend.value;
  s.device = device;
  s.batch_size = g.batch_size.value;
  s.max_validated_batch = g.max_validated_batch.value;
  s.backfill = g.backfill.value ? 1 : 0;
  s.threads_per_block = g.threads_per_block.value;
  s.memory_profile = g.memory_profile.value;
  s.planner_ordering = g.planner_ordering.value;
  s.solve_placement = g.solve_placement.value;
  s.refinement_steps = g.refinement_steps.value;
  s.health_check_nonfinite = g.check_nonfinite.value ? 1 : 0;
  s.telemetry = g.telemetry.value;
  s.profiler_ranges = g.profiler_ranges.value ? 1 : 0;
  s.log_level = settings.exec.log_level.value;
  s.memory_headroom_bytes = g.memory_headroom_gb.value < 0.0
                                ? -1.0 : g.memory_headroom_gb.value * 1.0e9;
  s.max_memory_bytes = g.max_memory_gb.value * 1.0e9;
  s.host_available_bytes = hostAvailableBytes();
  // GridPACK's network copies already count against MemAvailable; keep a
  // margin for their growth while results are collected (guide 7.1.3)
  s.host_reserved_bytes = 0.5 * replica_bytes * local_size;
  s.pivot_tolerance = g.pivot_tolerance.value;
  s.health_residual_limit = g.residual_limit.value;
  s.health_pivot_limit = g.pivot_limit.value;
  s.log_fn = &HostLogger::callback;
  s.log_user = log.get();
  return s;
}

bool BatchPath::Impl::shadowSelected(int event) const
{
  const double f = settings.gpu.shadow_fraction.value;
  if (f <= 0.0) return false;
  if (f >= 1.0) return true;
  // deterministic spread over the case list
  const unsigned h = static_cast<unsigned>(event) * 2654435761u;
  return (h % 1000000u) < static_cast<unsigned>(f * 1000000.0);
}

BatchPath::BatchPath(gridpack::utility::Configuration *config,
                     gridpack::parallel::Communicator &world)
    : p_impl(std::make_unique<Impl>())
{
  Impl &d = *p_impl;
  d.world = world;
  d.rank = world.rank();
  d.size = world.size();
  d.t_start = now();
  try {
    d.settings = resolveSettings(config);
  } catch (const SettingsError &e) {
    stopRun(world, e.what());
  }
  const GpuBatchSettings &g = d.settings.gpu;
  d.requested = g.block_present && g.enabled.value != Enabled::Off;
  if (!d.requested) return;
  d.log = std::make_unique<HostLogger>(d.rank, d.settings.exec.log_level.value);
  if (d.rank == 0) {
    d.info("GPU batch contingency path requested; effective settings:");
    for (const std::string &line : describeSettings(d.settings)) d.info("  " + line);
  }
  MPI_Comm_split_type(static_cast<MPI_Comm>(world), MPI_COMM_TYPE_SHARED, d.rank,
                      MPI_INFO_NULL, &d.node_comm);
  MPI_Comm_rank(d.node_comm, &d.local_rank);
  MPI_Comm_size(d.node_comm, &d.local_size);
  d.plugin_dirs = pluginSearchPath(g.plugin_path.value, executableDirectory());

  // Which ranks drive GPUs (Execution/acceleratorRanks; guide 8.16 #6)
  const bool cpu_ref = g.backend.value == BATCHPF_BACKEND_CPU_REFERENCE;
  int count = 0;
  std::string why;
  if (d.local_rank == 0) count = Accelerator::countDevices(d.plugin_dirs, &why);
  MPI_Bcast(&count, 1, MPI_INT, 0, d.node_comm);
  const std::vector<int> &listed = d.settings.exec.accelerator_ranks.value;
  if (!listed.empty()) {
    d.is_accelerator = std::find(listed.begin(), listed.end(), d.rank) != listed.end();
    int pos = 0;   // position among this node's listed ranks
    std::vector<int> flags(d.local_size, 0);
    int mine = d.is_accelerator ? 1 : 0;
    MPI_Allgather(&mine, 1, MPI_INT, flags.data(), 1, MPI_INT, d.node_comm);
    for (int r = 0; r < d.local_rank; r++) pos += flags[r];
    d.device = (g.device.source == Source::Xml || count <= 0)
                   ? g.device.value : pos % count;
  } else if (g.device.source == Source::Xml || cpu_ref) {
    d.is_accelerator = d.local_rank == 0;   // one rank per node, chosen device
    d.device = g.device.value;
  } else {
    d.is_accelerator = d.local_rank < count;   // one rank per visible GPU
    d.device = d.local_rank;
  }

  // CPU topology and placement (B9-H)
  d.topology = discoverTopology();
  std::vector<int> acc_flags(d.local_size, 0);
  int mine = d.is_accelerator ? 1 : 0;
  MPI_Allgather(&mine, 1, MPI_INT, acc_flags.data(), 1, MPI_INT, d.node_comm);
  std::vector<int> acc_local;
  for (int r = 0; r < d.local_size; r++) {
    if (acc_flags[r]) acc_local.push_back(r);
  }
  const std::string bind = applyBinding(d.topology, d.settings.exec.cpu_binding.value,
                                        d.local_rank, d.local_size, acc_local);
  if (d.local_rank == 0) d.info(describeTopology(d.topology));
  if (d.log->enabled(BATCHPF_LOG_DEBUG) || d.local_rank == 0) d.info(bind);

  // Load the plugin and probe the GPU on accelerator ranks (B13, R0)
  int ok = 0;
  std::string reason = why;
  if (d.is_accelerator) {
    d.acc = Accelerator::load(d.plugin_dirs, d.device, &reason);
    if (d.acc && !d.acc->probeOk() && !cpu_ref) {
      reason = d.acc->probeMessage();
      d.acc.reset();
    }
    if (d.acc) {
      ok = 1;
      const batchpf_device_info &p = d.acc->probeInfo();
      std::ostringstream os;
      os << "plugin " << d.acc->pluginFile() << " (" << d.acc->pluginVersion() << ", "
         << d.acc->buildInfo() << ")";
      d.info(os.str());
      if (d.acc->probeOk()) {
        std::ostringstream ds;
        ds << "GPU " << p.device << " of " << p.device_count << ": " << std::begin(p.name)
           << ", compute capability " << p.cc_major << "." << p.cc_minor
           << ", " << p.multiprocessors << " multiprocessors, integrated=" << p.integrated
           << ", concurrent managed access=" << p.concurrent_managed_access
           << ", pageable memory access=" << p.pageable_memory_access
           << " (host page tables " << p.pageable_uses_host_page_tables << ")"
           << ", host native atomics=" << p.host_native_atomics
           << ", GPUDirect RDMA=" << p.gpudirect_rdma
           << ", DMA-BUF=" << p.dmabuf
           << ", memory " << p.total_memory_bytes / 1e9 << " GB total, "
           << p.free_memory_bytes / 1e9 << " GB free (CUDA runtime), driver "
           << p.driver_version << ", runtime " << p.runtime_version;
        d.info(ds.str());
      }
    } else {
      d.warn("accelerator not available on this rank: " + reason);
    }
  }
  std::vector<int> oks(d.size, 0);
  MPI_Allgather(&ok, 1, MPI_INT, oks.data(), 1, MPI_INT, static_cast<MPI_Comm>(world));
  for (int r = 0; r < d.size; r++) {
    if (oks[r]) d.accelerator_ranks.push_back(r);
  }
  d.active = !d.accelerator_ranks.empty();
  if (!d.active) {
    if (g.enabled.value == Enabled::On &&
        g.on_unavailable.value == OnUnavailable::Error) {
      stopRun(world, "GPUBatch/enabled=on and onUnavailable=error, but no rank "
                     "could use the accelerator" +
                     (reason.empty() ? std::string() : " (" + reason + ")"));
    }
    if (d.rank == 0) {
      d.info("no usable accelerator; running GridPACK's CPU contingency loop" +
             (reason.empty() ? std::string() : " (" + reason + ")"));
    }
  }
}

BatchPath::~BatchPath()
{
  if (p_impl && p_impl->node_comm != MPI_COMM_NULL) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Comm_free(&p_impl->node_comm);
  }
}

bool BatchPath::active() const { return p_impl->active; }

void BatchPath::prepare(gridpack::powerflow::PFAppModule &pf_app,
                        boost::shared_ptr<gridpack::powerflow::PFNetwork> network,
                        std::vector<gridpack::powerflow::Contingency> &events,
                        bool ca_qlim, const std::string &output_file)
{
  Impl &d = *p_impl;
  if (!d.active) return;
  gridpack::utility::CoarseTimer *timer = gridpack::utility::CoarseTimer::instance();
  const int t_prep = timer->createCategory("GPU batch: prepare (export, classify, plan)");
  timer->start(t_prep);
  d.app = &pf_app;
  d.network = network;
  d.events = &events;
  d.ca_qlim = ca_qlim;
  d.output_file = output_file;
  d.replica_bytes = processResidentBytes();
  d.swap_start = hostSwapCounters();

  // B5: export the network as every contingency solve starts from it
  pf_app.exportSupersetModel(&d.model);
  const int nbus = static_cast<int>(d.model.buses.size());
  // Every rank holds the whole network; check the copies agree, since
  // results computed on one rank are reported on another
  std::array<long, 2> sig = {nbus, 0};
  for (int k = 0; k < nbus; k++) {
    sig[1] = (sig[1] * 1000003L + d.model.buses[k].original_index) % 2147483647L;
  }
  std::array<long, 2> lo{}, hi{};
  MPI_Allreduce(sig.data(), lo.data(), 2, MPI_LONG, MPI_MIN, static_cast<MPI_Comm>(d.world));
  MPI_Allreduce(sig.data(), hi.data(), 2, MPI_LONG, MPI_MAX, static_cast<MPI_Comm>(d.world));
  if (lo[0] != hi[0] || lo[1] != hi[1]) {
    d.active = false;
    if (d.rank == 0) {
      d.warn("network copies differ between ranks; using GridPACK's CPU loop");
    }
    timer->stop(t_prep);
    return;
  }

  // B4: classify every case, spread over the ranks
  const double t0 = now();
  const gridpack::powerflow::PFAppModule::SolverParameters prm = pf_app.getSolverParameters();
  const bool controls = prm.switched_shunt || prm.ltc || prm.area_interchange;
  // dc lines need their converter buses on every rank's network copy
  const bool dc_missing = !d.model.dc_lines.empty() && !d.model.dc_found;
  if (dc_missing && d.rank == 0) {
    d.warn("a dc converter bus is not in the local network; every case uses GridPACK's "
           "CPU loop");
  }
  d.classifier = std::make_unique<Classifier>(pf_app, network, d.model,
                                              controls || dc_missing);
  if (d.rank == 0) d.info("contingency classifier fast path: " + d.classifier->fastPathNote());
  const int n = static_cast<int>(events.size());
  std::vector<char> local;
  for (int e = d.rank; e < n; e += d.size) {
    pack(d.classifier->classify(CaseIndex{e}, events[e]), &local);
  }
  // Q-limit warnings printed by GridPACK while applying cases are not
  // results; clear them as the stock loop does between cases
  gridpack::powerflow::PFBus::clearQlimWarnings();
  int len = static_cast<int>(local.size());
  std::vector<int> lens(d.size), offs(d.size, 0);
  MPI_Allgather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, static_cast<MPI_Comm>(d.world));
  int total = 0;
  for (int r = 0; r < d.size; r++) {
    offs[r] = total;
    total += lens[r];
  }
  std::vector<char> all(total > 0 ? total : 1);
  MPI_Allgatherv(local.data(), len, MPI_BYTE, all.data(), lens.data(), offs.data(),
                 MPI_BYTE, static_cast<MPI_Comm>(d.world));
  d.classes.assign(n, CaseClass());
  std::size_t off = 0;
  while (off < static_cast<std::size_t>(total)) {
    CaseClass c;
    off += unpack(all.data() + off, &c);
    if (c.event.value >= 0 && c.event.value < n) d.classes[c.event.value] = c;
  }
  int fast = 0;
  for (int e = 0; e < n; e++) {
    (d.classes[e].path == CasePath::Gpu ? d.gpu_events : d.cpu_events).push_back(e);
    fast += d.classes[e].fast ? 1 : 0;
  }
  d.t_classify = now() - t0;
  if (d.rank == 0) {
    std::ostringstream os;
    os << "classified " << n << " cases in " << d.t_classify << " s: "
       << d.gpu_events.size() << " GPU, " << d.cpu_events.size() << " CPU ("
       << fast << " by the fast path)";
    d.info(os.str());
  }

  // Split GPU cases into contiguous blocks, one per accelerator rank
  const int na = static_cast<int>(d.accelerator_ranks.size());
  const int ng = static_cast<int>(d.gpu_events.size());
  d.owner.assign(n, -1);
  for (int a = 0; a < na; a++) {
    const int beg = static_cast<int>((static_cast<long>(ng) * a) / na);
    const int end = static_cast<int>((static_cast<long>(ng) * (a + 1)) / na);
    for (int i = beg; i < end; i++) {
      d.owner[d.gpu_events[i]] = d.accelerator_ranks[a];
      if (d.accelerator_ranks[a] == d.rank) d.my_gpu_events.push_back(d.gpu_events[i]);
    }
  }

  // Solver rules for the GPU, from GridPACK's own settings
  std::memset(&d.params, 0, sizeof(d.params));
  d.params.struct_size = sizeof(d.params);
  d.params.struct_version = 2;
  d.params.tolerance = prm.tolerance;
  d.params.damping_factor = prm.damping_factor;
  d.params.qlim_deadband = prm.qlim_deadband;
  d.params.max_iteration = prm.max_iteration;
  d.params.pf_qlim = prm.qlim ? 1 : 0;
  d.params.max_controller_iterations = prm.max_controller_iterations;
  d.params.ca_qlim = ca_qlim ? 1 : 0;
  d.params.warm_start = d.settings.gpu.warm_start.value;
  d.params.hvdc_tolerance = prm.hvdc_tolerance;
  d.history_capacity = std::max(1, prm.max_iteration);

  // B6 (in the plugin): one-time planning on each accelerator rank
  const double t1 = now();
  int ok = 1;
  if (d.acc && !d.my_gpu_events.empty()) {
    ModelArrays a;
    for (const auto &b : d.model.buses) {
      a.bus_type.push_back(b.type);
      a.g.push_back(b.g_diag);
      a.b.push_back(b.b_diag);
      a.p0.push_back(b.p0);
      a.q0.push_back(b.q0);
      a.v_init.push_back(b.v_init);
      a.theta_init.push_back(b.theta_init);
      a.v_base.push_back(b.v_solved);
      a.theta_base.push_back(b.theta_solved);
      a.ql.push_back(b.ql);
      a.ip.push_back(b.ip);
      a.iq.push_back(b.iq);
      a.yp.push_back(b.yp);
      a.yq.push_back(b.yq);
      a.qmax.push_back(b.qmax);
      a.qmin.push_back(b.qmin);
      a.dg_q.push_back(b.dg_q);
      a.dc_p.push_back(b.dc_p);
      a.dc_q.push_back(b.dc_q);
    }
    a.row_start.assign(d.model.row_start.begin(), d.model.row_start.end());
    a.edge_col.assign(d.model.edge_col.begin(), d.model.edge_col.end());
    a.edge_mate.assign(d.model.edge_mate.begin(), d.model.edge_mate.end());
    a.eg = d.model.edge_g;
    a.eb = d.model.edge_b;
    for (std::size_t l = 0; l < d.model.dc_lines.size(); l++) {
      a.dc_lines.push_back(dcLineRecord(d.model.dc_lines[l], d.model.dc_rect_bus[l],
                                        d.model.dc_inv_bus[l], d.model.dc_reference[l]));
    }
    std::string why;
    try {
      if (a.needsInterface11() && d.acc->apiMinor() < 1) {
        throw AcceleratorError(BATCHPF_ERR_VERSION,
                               "the plugin implements interface 1." +
                                   std::to_string(d.acc->apiMinor()) +
                                   ", which does not model distributed generation or dc lines");
      }
      if (!d.acc->createSession(d.pluginSettings(), &why)) throw AcceleratorError(BATCHPF_ERR_UNAVAILABLE, why);
      batchpf_model rec = a.record();
      rec.sbase = d.model.sbase;
      d.acc->setModel(rec);
      d.capacity = d.acc->plan(d.params, static_cast<int64_t>(d.my_gpu_events.size()));
    } catch (const AcceleratorError &e) {
      d.warn(std::string("GPU planning failed; this rank's GPU cases go to the CPU "
                         "path: ") + e.what());
      ok = 0;
    }
  }
  d.t_plan = now() - t1;
  std::vector<int> oks(d.size, 1);
  MPI_Allgather(&ok, 1, MPI_INT, oks.data(), 1, MPI_INT, static_cast<MPI_Comm>(d.world));
  bool any_fail = false;
  for (int r = 0; r < d.size; r++) any_fail = any_fail || !oks[r];
  if (any_fail) {
    // Cases owned by failed ranks fall back to GridPACK's CPU loop
    std::vector<int> keep;
    for (int e : d.gpu_events) {
      if (oks[d.owner[e]]) {
        keep.push_back(e);
      } else {
        d.owner[e] = -1;
        d.classes[e].path = CasePath::Cpu;
        d.classes[e].reason = CpuReason::Other;
        d.cpu_events.push_back(e);
      }
    }
    d.gpu_events.swap(keep);
    std::sort(d.cpu_events.begin(), d.cpu_events.end());
    if (!ok) {
      d.my_gpu_events.clear();
      d.acc.reset();
    }
    std::vector<int> still;
    for (int r : d.accelerator_ranks) {
      if (oks[r]) still.push_back(r);
    }
    d.accelerator_ranks.swap(still);
    if (d.accelerator_ranks.empty() && d.settings.gpu.enabled.value == Enabled::On &&
        d.settings.gpu.on_unavailable.value == OnUnavailable::Error) {
      stopRun(d.world, "GPU planning failed and GPUBatch/onUnavailable=error");
    }
  }
  timer->stop(t_prep);
}

}  // namespace batchpf
}  // namespace gridpack
