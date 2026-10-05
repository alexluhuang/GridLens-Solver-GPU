/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   batch_run.cpp
 * @date   2026-10-05
 *
 * @brief Execution (scenarios R2, R3) and reporting (R5) of the batch path.
 *
 * Two phases, both collective over all ranks:
 *
 *  1. CPU-path cases go through GridPACK's TaskManager exactly as in the
 *     stock driver. Accelerator ranks have already queued their first GPU
 *     batches, so the GPU works meanwhile.
 *  2. GPU results are reported. Each accelerator rank ("server") splits
 *     finished batches into small packets. Any rank ("client") may ask a
 *     server for a packet, inject the states into its own network copy and
 *     let GridPACK check and write them. A server also reports its own
 *     packets when nobody is waiting. Everything is non-blocking, so a rank
 *     can serve its own results while it waits for another server's; this
 *     keeps two servers from waiting on each other.
 *
 * Cases that the GPU flagged or could not solve are solved again by
 * GridPACK on the reporting rank (B10), as the guide requires before any
 * case is reported as diverged.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <thread>
#include <type_traits>

#include "batch_path_impl.hpp"
#include "gridpack/parallel/task_manager.hpp"
#include "gridpack/timer/coarse_timer.hpp"

namespace gridpack {
namespace batchpf {

namespace {

const int kTagRequest = 7101;   // client -> server: "send me work"
const int kTagPacket = 7102;    // server -> client: packet (0 cases = done)

double now()
{
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// A submitted batch of GPU cases and its result buffers
struct Chunk {
  std::vector<int> events;
  std::vector<batchpf_case> cases;
  std::vector<batchpf_outcome> outcomes;
  std::vector<double> v, theta, qreq;
  std::vector<int32_t> conv, hist_count;
  std::vector<batchpf_mismatch_record> hist;
  batchpf_batch batch{};
  batchpf_results results{};
  int64_t ticket = 0;
};

/// A slice of a finished chunk, handed out or reported locally
struct Packet {
  std::shared_ptr<Chunk> chunk;
  int begin = 0;
  int end = 0;
};

template <class T>
void put(std::vector<char> *buf, const T *src, std::size_t n)
{
  static_assert(std::is_trivially_copyable<T>::value, "copied with memcpy");
  const std::size_t at = buf->size();
  buf->resize(at + n * sizeof(T));
  if (n) std::memcpy(buf->data() + at, src, n * sizeof(T));
}

template <class T>
const char *get(const char *p, T *dst, std::size_t n)
{
  static_assert(std::is_trivially_copyable<T>::value, "copied with memcpy");
  if (n) std::memcpy(dst, p, n * sizeof(T));
  return p + n * sizeof(T);
}

/// Wrap an angle difference to [-pi, pi)
double wrapDiff(double a)
{
  const double pi = 4.0 * std::atan(1.0);
  double x = std::fmod(a + pi, 2.0 * pi);
  if (x < 0.0) x += 2.0 * pi;
  return x - pi;
}

}  // namespace

// -------------------------------------------------------------------------
// Reporting one GPU case (B11) or sending it back to the CPU (B10)
// -------------------------------------------------------------------------

void BatchPath::Impl::processGpuCase(int event, const batchpf_outcome &o,
                                     const double *v, const double *theta,
                                     const int32_t *conv, const double *qreq,
                                     const batchpf_mismatch_record *hist,
                                     int hist_count, const ProcessCase &process)
{
  const int n = static_cast<int>(model.buses.size());
  OutcomeRow row;
  row.event = event;
  row.fast = classes[event].fast ? 1 : 0;
  row.gpu_status = o.status;
  row.health = o.health_events;
  row.iterations = o.iterations;
  row.total_iterations = o.total_iterations;
  row.controller_iterations = o.controller_iterations;
  row.solves = o.solves;
  row.pv_to_pq = o.pv_to_pq;
  row.final_tolerance = o.final_tolerance;
  auto orig = [&](int k) { return (k >= 0 && k < n) ? model.buses[k].original_index : 0; };
  if (o.status == BATCHPF_CASE_CONVERGED) {
    GpuCaseResult r;
    r.v.assign(v, v + n);
    r.theta.assign(theta, theta + n);
    r.qlim_conversion.assign(conv, conv + n);
    r.q_required.assign(qreq, qreq + n);
    gridpack::utility::ConvergenceSummary &cs = r.convergence;
    cs.converged = true;
    cs.iterations = o.iterations;
    cs.finalTolerance = o.final_tolerance;
    cs.finalMismatch.maxPBus = orig(o.max_p_bus);
    cs.finalMismatch.maxPMismatch = o.max_p_mismatch;
    cs.finalMismatch.maxQBus = orig(o.max_q_bus);
    cs.finalMismatch.maxQMismatch = o.max_q_mismatch;
    for (int k = 0; k < hist_count; k++) {
      gridpack::utility::MismatchInfo m{};
      m.maxPBus = orig(hist[k].max_p_bus);
      m.maxPMismatch = hist[k].max_p_mismatch;
      m.maxQBus = orig(hist[k].max_q_bus);
      m.maxQMismatch = hist[k].max_q_mismatch;
      cs.perIteration.push_back(m);
    }
    process(event, &r);
    reported_gpu++;
    if (shadowSelected(event)) shadowCompare(event, r);
  } else {
    // flagged, diverged or not run on the GPU: GridPACK solves it (R3)
    row.path = 2;
    process(event, nullptr);
    fallback++;
  }
  outcomes.push_back(row);
}

/**
 * Shadow validation (GPUBatch/shadowFraction): solve the case again with
 * GridPACK, without writing anything, and compare (FID-1). Fast-path
 * classifications are also checked against GridPACK's full routine.
 */
void BatchPath::Impl::shadowCompare(int event, const GpuCaseResult &res)
{
  gridpack::powerflow::Contingency &c = (*events)[event];
  ShadowRow row;
  row.event = event;
  row.gpu_ok = 1;
  app->suppressOutput(true);
  app->resetVoltages();
  network->updateBuses();
  const bool found = app->setContingency(c);
  bool ok = false;
  if (found && app->getIslandCount() <= 1) {
    try {
      ok = app->solve();
      if (ok && ca_qlim && !app->checkQlimViolations()) ok = app->solve();
    } catch (...) {
      ok = false;
    }
  }
  row.cpu_ok = ok ? 1 : 0;
  const int n = static_cast<int>(model.buses.size());
  std::vector<int> type(n);
  for (int k = 0; k < n; k++) type[k] = model.buses[k].type;
  for (const batchpf_bus_update &u : classes[event].bus_updates) type[u.bus] = u.type;
  for (int k = 0; k < n; k++) {
    if (!network->getActiveBus(k)) continue;
    auto *bus = dynamic_cast<gridpack::powerflow::PFBus *>(network->getBus(k).get());
    if (bus->isIsolated()) continue;
    row.max_dv = std::max(row.max_dv, std::fabs(bus->getVoltage() - res.v[k]));
    row.max_dtheta = std::max(row.max_dtheta,
                              std::fabs(wrapDiff(bus->getPhase() - res.theta[k])));
    const bool cpu_pv = bus->isPV() && !bus->getReferenceBus();
    const bool gpu_pv = type[k] == BATCHPF_BUS_PV && res.qlim_conversion[k] == 0;
    row.pv_cpu += cpu_pv ? 1 : 0;
    row.pv_gpu += gpu_pv ? 1 : 0;
    if (cpu_pv != gpu_pv) row.pv_set_match = 0;
  }
  app->unSetContingency(c);
  if (ca_qlim) app->clearQlimViolations();
  gridpack::powerflow::PFBus::clearQlimWarnings();
  app->suppressOutput(false);
  if (classes[event].fast) {
    const CaseClass full = classifier->classifyFull(CaseIndex{event}, c);
    auto close = [](double a, double b) {
      return a == b || std::fabs(a - b) <= 1.0e-12 * std::max(1.0, std::fabs(a));
    };
    auto sameBus = [&](const batchpf_bus_update &a, const batchpf_bus_update &b) {
      return a.type == b.type && close(a.g_diag, b.g_diag) && close(a.b_diag, b.b_diag) &&
             close(a.p0, b.p0) && close(a.q0, b.q0) &&
             close(a.qmax, b.qmax) && close(a.qmin, b.qmin);
    };
    std::map<int, batchpf_bus_update> fb, fast;
    for (const auto &u : full.bus_updates) fb[u.bus] = u;
    for (const auto &u : classes[event].bus_updates) fast[u.bus] = u;
    // Either routine may list an unchanged bus. Resolve omitted updates
    // to the base state and compare the union, including generator limits.
    const auto busValue = [&](const auto &updates, int bus) {
      const auto it = updates.find(bus);
      if (it != updates.end()) return it->second;
      const auto &b = model.buses[bus];
      return batchpf_bus_update{bus, b.type, b.g_diag, b.b_diag, b.p0, b.q0, b.qmax, b.qmin};
    };
    bool match = full.path == CasePath::Gpu &&
                 full.slack_bus.value == classes[event].slack_bus.value;
    for (const auto &u : fb) match = match && sameBus(u.second, busValue(fast, u.first));
    for (const auto &u : fast) match = match && sameBus(u.second, busValue(fb, u.first));
    std::map<int, batchpf_edge_update> fe, fast_edges;
    for (const auto &u : full.edge_updates) fe[u.edge] = u;
    for (const auto &u : classes[event].edge_updates) fast_edges[u.edge] = u;
    const auto edgeValue = [&](const auto &updates, int edge) {
      const auto it = updates.find(edge);
      if (it != updates.end()) return it->second;
      return batchpf_edge_update{edge, 0, model.edge_g[edge], model.edge_b[edge]};
    };
    for (const auto &updates : {&fe, &fast_edges}) {
      for (const auto &u : *updates) {
        const auto a = edgeValue(fe, u.first), b = edgeValue(fast_edges, u.first);
        match = match && close(a.g, b.g) && close(a.b, b.b);
      }
    }
    row.class_match = match ? 1 : 0;
  }
  shadows.push_back(row);
}

// -------------------------------------------------------------------------
// Execution
// -------------------------------------------------------------------------

void BatchPath::run(const ProcessCase &process)
{
  Impl &d = *p_impl;
  gridpack::utility::CoarseTimer *timer = gridpack::utility::CoarseTimer::instance();
  const int t_run = timer->createCategory("GPU batch: run and report");
  timer->start(t_run);
  const int n = static_cast<int>(d.model.buses.size());
  const MPI_Comm comm = static_cast<MPI_Comm>(d.world);

  // ---- GPU work of this rank: chunks of several batches ------------------
  std::deque<std::shared_ptr<Chunk>> inflight;
  std::deque<Packet> ready;
  std::size_t next = 0;
  const int chunk_cases = std::max(1, d.capacity * 4);
  const int packet_cases = std::max(1, std::min(16, d.capacity / std::max(1, d.size)));
  auto submitNext = [&]() {
    if (!d.acc || next >= d.my_gpu_events.size()) return;
    auto ch = std::make_shared<Chunk>();
    const std::size_t end = std::min(d.my_gpu_events.size(), next + chunk_cases);
    ch->events.assign(d.my_gpu_events.begin() + static_cast<std::ptrdiff_t>(next),
                      d.my_gpu_events.begin() + static_cast<std::ptrdiff_t>(end));
    next = end;
    const int m = static_cast<int>(ch->events.size());
    for (int e : ch->events) {
      const CaseClass &c = d.classes[e];
      batchpf_case bc;
      std::memset(&bc, 0, sizeof(bc));
      bc.case_id = e;
      bc.n_bus_updates = static_cast<int32_t>(c.bus_updates.size());
      bc.n_edge_updates = static_cast<int32_t>(c.edge_updates.size());
      bc.bus_updates = c.bus_updates.data();
      bc.edge_updates = c.edge_updates.data();
      bc.slack_bus = c.slack_bus.value;
      ch->cases.push_back(bc);
    }
    ch->outcomes.assign(m, batchpf_outcome());
    ch->v.assign(static_cast<std::size_t>(m) * n, 0.0);
    ch->theta.assign(static_cast<std::size_t>(m) * n, 0.0);
    ch->qreq.assign(static_cast<std::size_t>(m) * n, 0.0);
    ch->conv.assign(static_cast<std::size_t>(m) * n, 0);
    ch->hist.assign(static_cast<std::size_t>(m) * d.history_capacity, batchpf_mismatch_record());
    ch->hist_count.assign(m, 0);
    std::memset(&ch->batch, 0, sizeof(ch->batch));
    ch->batch.struct_size = sizeof(ch->batch);
    ch->batch.struct_version = 1;
    ch->batch.n_cases = m;
    ch->batch.cases = ch->cases.data();
    std::memset(&ch->results, 0, sizeof(ch->results));
    ch->results.struct_size = sizeof(ch->results);
    ch->results.struct_version = 1;
    ch->results.n_cases = m;
    ch->results.n_bus = n;
    ch->results.outcomes = ch->outcomes.data();
    ch->results.v = ch->v.data();
    ch->results.theta = ch->theta.data();
    ch->results.qlim_conversion = ch->conv.data();
    ch->results.q_required = ch->qreq.data();
    ch->results.history_capacity = d.history_capacity;
    ch->results.history = ch->hist.data();
    ch->results.history_count = ch->hist_count.data();
    ch->ticket = d.acc->submit(ch->batch, ch->results);
    inflight.push_back(ch);
  };
  std::string gpu_error;
  // Move finished chunks to the ready queue; keep two chunks queued
  auto poll = [&](int timeout_ms) {
    bool got = false;
    while (!inflight.empty()) {
      bool done = false;
      try {
        done = d.acc->wait(inflight.front()->ticket, got ? 0 : timeout_ms);
      } catch (const AcceleratorError &e) {
        // a failed batch: its cases go to the CPU path
        gpu_error = e.what();
        d.warn(std::string("GPU batch failed, its cases are solved by GridPACK: ") +
               e.what());
        for (batchpf_outcome &o : inflight.front()->outcomes) {
          o.status = BATCHPF_CASE_NOT_RUN;
        }
        done = true;
      }
      if (!done) break;
      std::shared_ptr<Chunk> ch = inflight.front();
      inflight.pop_front();
      const int m = static_cast<int>(ch->events.size());
      // Outcome counts of the chunk, as soon as it is done (guide 8.12)
      int st[4] = {0, 0, 0, 0}, hb[6] = {0, 0, 0, 0, 0, 0};
      for (const batchpf_outcome &o : ch->outcomes) {
        st[std::max(0, std::min(3, o.status))]++;
        for (int bit = 0; bit < 6; bit++) hb[bit] += (o.health_events >> bit) & 1;
      }
      std::ostringstream os;
      os << "GPU chunk of " << m << " cases: " << st[0] << " converged, " << st[1]
         << " diverged, " << st[2] << " flagged, " << st[3] << " not run";
      if (st[1] + st[2] > 0) {
        os << " (small pivot " << hb[0] << ", non-finite " << hb[1] << ", residual "
           << hb[2] << ", iteration limit " << hb[3] << ", mismatch growth " << hb[4]
           << ", stagnation " << hb[5] << ")";
      }
      d.info(os.str());
      for (int b = 0; b < m; b += packet_cases) {
        ready.push_back({ch, b, std::min(m, b + packet_cases)});
      }
      submitNext();
      got = true;
    }
    return got;
  };
  const double t0 = now();
  if (d.acc) {
    submitNext();
    submitNext();
  }

  // ---- Phase 1: CPU-path cases through GridPACK's TaskManager -------------
  {
    gridpack::parallel::TaskManager tm(d.world);
    tm.set(static_cast<int>(d.cpu_events.size()));
    int t = 0;
    while (tm.nextTask(&t)) {
      if (d.acc) poll(0);
      const int e = d.cpu_events[t];
      process(e, nullptr);
      OutcomeRow row;
      row.event = e;
      row.path = 1;
      row.reason = static_cast<int>(d.classes[e].reason);
      d.outcomes.push_back(row);
    }
  }
  d.t_cpu = now() - t0;

  // ---- Phase 2: report GPU results ----------------------------------------
  std::set<int> servers_all;
  for (int e : d.gpu_events) {
    if (d.owner[e] >= 0) servers_all.insert(d.owner[e]);
  }
  const bool server = servers_all.count(d.rank) > 0;
  std::vector<int> servers(servers_all.begin(), servers_all.end());
  servers.erase(std::remove(servers.begin(), servers.end(), d.rank), servers.end());
  std::set<int> pending_done;   // clients still to be told "done"
  if (server) {
    for (int r = 0; r < d.size; r++) {
      if (r != d.rank) pending_done.insert(r);
    }
  }
  std::deque<int> waiting;
  struct Send { std::vector<char> buf; MPI_Request req = MPI_REQUEST_NULL; };
  std::list<Send> sends;
  bool request_out = false;
  int current = 0;
  const int one = 1;

  auto serialize = [&](const Packet &p, std::vector<char> *buf) {
    const Chunk &c = *p.chunk;
    const int32_t head[3] = {p.end - p.begin, n, d.history_capacity};
    put(buf, head, 3);
    for (int i = p.begin; i < p.end; i++) {
      const int32_t ev = c.events[i];
      put(buf, &ev, 1);
      put(buf, &c.outcomes[i], 1);
      put(buf, &c.hist_count[i], 1);
      put(buf, c.hist.data() + static_cast<std::size_t>(i) * d.history_capacity,
          static_cast<std::size_t>(d.history_capacity));
      put(buf, c.v.data() + static_cast<std::size_t>(i) * n, n);
      put(buf, c.theta.data() + static_cast<std::size_t>(i) * n, n);
      put(buf, c.conv.data() + static_cast<std::size_t>(i) * n, n);
      put(buf, c.qreq.data() + static_cast<std::size_t>(i) * n, n);
    }
  };
  auto sendTo = [&](int dest, const Packet *p) {
    sends.emplace_back();
    Send &s = sends.back();
    if (p) {
      serialize(*p, &s.buf);
    } else {
      const int32_t head[3] = {0, n, d.history_capacity};
      put(&s.buf, head, 3);
    }
    MPI_Isend(s.buf.data(), static_cast<int>(s.buf.size()), MPI_BYTE, dest,
              kTagPacket, comm, &s.req);
  };
  auto reportLocal = [&](Packet &p) {
    const Chunk &c = *p.chunk;
    const int i = p.begin++;
    d.processGpuCase(c.events[i], c.outcomes[i], c.v.data() + static_cast<std::size_t>(i) * n,
                     c.theta.data() + static_cast<std::size_t>(i) * n,
                     c.conv.data() + static_cast<std::size_t>(i) * n,
                     c.qreq.data() + static_cast<std::size_t>(i) * n,
                     c.hist.data() + static_cast<std::size_t>(i) * d.history_capacity,
                     c.hist_count[i], process);
  };
  auto ownDone = [&]() {
    return !server || (next >= d.my_gpu_events.size() && inflight.empty() && ready.empty());
  };
  const double t1 = now();
  while (true) {
    bool progress = false;
    if (server) {
      if (!inflight.empty()) progress = poll(0) || progress;
      int flag = 1;
      while (flag) {
        MPI_Status st;
        MPI_Iprobe(MPI_ANY_SOURCE, kTagRequest, comm, &flag, &st);
        if (flag) {
          int dummy = 0;
          MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, kTagRequest, comm, MPI_STATUS_IGNORE);
          waiting.push_back(st.MPI_SOURCE);
          progress = true;
        }
      }
      while (!waiting.empty() && !ready.empty()) {
        sendTo(waiting.front(), &ready.front());
        waiting.pop_front();
        ready.pop_front();
        progress = true;
      }
      if (ownDone()) {
        for (int w : waiting) {
          sendTo(w, nullptr);
          pending_done.erase(w);
        }
        waiting.clear();
      } else if (!ready.empty() && waiting.empty()) {
        reportLocal(ready.front());
        if (ready.front().begin >= ready.front().end) ready.pop_front();
        progress = true;
      }
    }
    // client side: ask other servers for packets
    if (!servers.empty()) {
      if (!request_out) {
        MPI_Send(&one, 1, MPI_INT, servers[current % servers.size()], kTagRequest, comm);
        request_out = true;
      } else {
        const int s = servers[current % servers.size()];
        int flag = 0;
        MPI_Status st;
        MPI_Iprobe(s, kTagPacket, comm, &flag, &st);
        if (flag) {
          int bytes = 0;
          MPI_Get_count(&st, MPI_BYTE, &bytes);
          std::vector<char> buf(bytes > 0 ? bytes : 1);
          MPI_Recv(buf.data(), bytes, MPI_BYTE, s, kTagPacket, comm, MPI_STATUS_IGNORE);
          request_out = false;
          int32_t head[3];
          const char *p = get(buf.data(), head, 3);
          if (head[0] == 0) {
            servers.erase(servers.begin() +
                          static_cast<std::ptrdiff_t>(current % servers.size()));
          } else {
            const int nb = head[1], hc = head[2];
            std::vector<batchpf_mismatch_record> hist(hc);
            std::vector<double> v(nb), th(nb), q(nb);
            std::vector<int32_t> cv(nb);
            for (int i = 0; i < head[0]; i++) {
              int32_t ev = 0, hn = 0;
              batchpf_outcome o;
              p = get(p, &ev, 1);
              p = get(p, &o, 1);
              p = get(p, &hn, 1);
              p = get(p, hist.data(), hc);
              p = get(p, v.data(), nb);
              p = get(p, th.data(), nb);
              p = get(p, cv.data(), nb);
              p = get(p, q.data(), nb);
              d.processGpuCase(ev, o, v.data(), th.data(), cv.data(), q.data(),
                               hist.data(), hn, process);
            }
            current++;
          }
          progress = true;
        }
      }
    }
    // retire completed sends
    for (auto it = sends.begin(); it != sends.end();) {
      int flag = 0;
      MPI_Test(&it->req, &flag, MPI_STATUS_IGNORE);
      it = flag ? sends.erase(it) : std::next(it);
    }
    const bool finished = ownDone() && pending_done.empty() && servers.empty() &&
                          !request_out && sends.empty();
    if (finished) break;
    if (!progress) {
      if (server && !inflight.empty()) {
        poll(2);
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    }
  }
  d.t_gpu_phase = now() - t1;
  d.world.barrier();
  timer->stop(t_run);
}

// -------------------------------------------------------------------------
// Summary (R5): sidecar tables and telemetry
// -------------------------------------------------------------------------

void BatchPath::finish()
{
  Impl &d = *p_impl;
  if (!d.active) return;
  const MPI_Comm comm = static_cast<MPI_Comm>(d.world);
  auto gatherRows = [&](const auto &local, auto *all) {
    using Row = typename std::decay<decltype(local[0])>::type;
    int bytes = static_cast<int>(local.size() * sizeof(Row));
    std::vector<int> counts(d.size), offs(d.size, 0);
    MPI_Gather(&bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
    int total = 0;
    if (d.rank == 0) {
      for (int r = 0; r < d.size; r++) {
        offs[r] = total;
        total += counts[r];
      }
    }
    std::vector<char> buf(total > 0 ? total : 1);
    MPI_Gatherv(local.data(), bytes, MPI_BYTE, buf.data(), counts.data(), offs.data(),
                MPI_BYTE, 0, comm);
    if (d.rank == 0) {
      all->resize(total / sizeof(Row));
      if (total > 0) std::memcpy(all->data(), buf.data(), total);
      std::sort(all->begin(), all->end(),
                [](const Row &a, const Row &b) { return a.event < b.event; });
    }
  };
  std::vector<OutcomeRow> rows;
  std::vector<ShadowRow> shadow;
  gatherRows(d.outcomes, &rows);
  gatherRows(d.shadows, &shadow);

  // Plugin telemetry from each accelerator rank
  batchpf_diagnostics mine;
  std::memset(&mine, 0, sizeof(mine));
  int have = 0;
  if (d.acc && !d.my_gpu_events.empty()) {
    try {
      mine = d.acc->diagnostics();
      have = 1;
    } catch (const AcceleratorError &e) {
      d.warn(std::string("could not read GPU diagnostics: ") + e.what());
    }
  }
  std::vector<batchpf_diagnostics> diags(d.size);
  std::vector<int> haves(d.size, 0);
  MPI_Gather(&mine, sizeof(mine), MPI_BYTE, diags.data(), sizeof(mine), MPI_BYTE, 0, comm);
  MPI_Gather(&have, 1, MPI_INT, haves.data(), 1, MPI_INT, 0, comm);
  double gpu_phase = d.t_gpu_phase, cpu_phase = d.t_cpu;
  MPI_Allreduce(MPI_IN_PLACE, &gpu_phase, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(MPI_IN_PLACE, &cpu_phase, 1, MPI_DOUBLE, MPI_MAX, comm);
  if (d.rank != 0) return;

  const std::vector<gridpack::powerflow::Contingency> &ev = *d.events;
  auto rtrim = [](std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
    return s;
  };
  static const char *const paths[] = {"gpu", "cpu", "cpu_fallback"};
  static const char *const statuses[] = {"converged", "diverged", "flagged", "not_run"};
  {
    std::ofstream out((d.output_file + "_gpu_outcomes.csv").c_str());
    out << "event_idx,contingency,path,cpu_reason,classified_fast,gpu_status,"
           "health_events,iterations,total_iterations,controller_iterations,solves,"
           "pv_to_pq,final_tolerance\n";
    for (const OutcomeRow &r : rows) {
      out << r.event + 1 << "," << rtrim(ev[r.event].p_name) << "," << paths[r.path] << ","
          << cpuReasonName(static_cast<CpuReason>(r.reason)) << "," << r.fast << ","
          << (r.gpu_status >= 0 && r.gpu_status <= 3 ? statuses[r.gpu_status] : "") << ","
          << r.health << "," << r.iterations << "," << r.total_iterations << ","
          << r.controller_iterations << "," << r.solves << "," << r.pv_to_pq << ","
          << std::scientific << std::setprecision(6) << r.final_tolerance
          << std::defaultfloat << "\n";
    }
  }
  if (!shadow.empty()) {
    std::ofstream out((d.output_file + "_gpu_shadow.csv").c_str());
    out << "event_idx,contingency,cpu_converged,gpu_converged,max_dv_pu,max_dtheta_rad,"
           "pv_buses_cpu,pv_buses_gpu,classification_match,pv_set_match\n";
    double mdv = 0.0, mdt = 0.0;
    int agree = 0, pv_agree = 0, cls = 0;
    for (const ShadowRow &r : shadow) {
      out << r.event + 1 << "," << rtrim(ev[r.event].p_name) << "," << r.cpu_ok << ","
          << r.gpu_ok << "," << std::scientific << std::setprecision(3) << r.max_dv << ","
          << r.max_dtheta << std::defaultfloat << "," << r.pv_cpu << "," << r.pv_gpu
          << "," << r.class_match << "," << r.pv_set_match << "\n";
      if (r.cpu_ok) {
        mdv = std::max(mdv, r.max_dv);
        mdt = std::max(mdt, r.max_dtheta);
      }
      agree += (r.cpu_ok == r.gpu_ok) ? 1 : 0;
      pv_agree += r.pv_set_match;
      cls += r.class_match;
    }
    std::ostringstream os;
    os << "shadow validation: " << shadow.size() << " cases re-solved by GridPACK; "
       << "status agrees in " << agree << ", PV/PQ sets agree in " << pv_agree
       << ", classification agrees in " << cls << "; largest |dV| " << mdv
       << " pu, largest |dtheta| " << mdt << " rad (written to " << d.output_file
       << "_gpu_shadow.csv)";
    d.info(os.str());
  }
  // Path mix, fallback causes, iteration histogram (guide 8.12)
  int gpu = 0, cpu = 0, fb = 0;
  std::map<std::string, int> reasons;
  std::map<int, int> iters;
  for (const OutcomeRow &r : rows) {
    if (r.path == 0) {
      gpu++;
      iters[std::min(r.total_iterations, 11)]++;
    } else if (r.path == 1) {
      cpu++;
      reasons[cpuReasonName(static_cast<CpuReason>(r.reason))]++;
    } else {
      fb++;
      reasons[std::string("gpu_") + statuses[std::max(0, std::min(3, r.gpu_status))]]++;
    }
  }
  std::ostringstream os;
  os << "paths: " << gpu << " reported from the GPU, " << cpu << " on the CPU path, "
     << fb << " solved again on the CPU after the GPU";
  if (!reasons.empty()) {
    os << " (";
    bool first = true;
    for (const auto &r : reasons) {
      os << (first ? "" : ", ") << r.first << " " << r.second;
      first = false;
    }
    os << ")";
  }
  d.info(os.str());
  if (!iters.empty()) {
    std::ostringstream hs;
    hs << "GPU Newton iterations per case:";
    for (const auto &h : iters) {
      hs << " " << (h.first >= 11 ? std::string(">10") : std::to_string(h.first))
         << ":" << h.second;
    }
    d.info(hs.str());
  }
  for (int r = 0; r < d.size; r++) {
    if (!haves[r]) continue;
    const batchpf_diagnostics &g = diags[r];
    static const char *const names[] = {"auto", "cudss", "alg2", "cpu_reference"};
    std::ostringstream ds;
    ds << "rank " << r << " GPU: backend " << names[std::max(0, std::min(3, g.backend))]
       << ", batch size " << g.batch_size << ", " << g.cases << " cases ("
       << g.converged << " converged, " << g.diverged << " diverged, " << g.flagged
       << " flagged) in " << g.batches << " submissions, "
       << std::fixed << std::setprecision(3) << g.seconds_total << " s";
    if (g.slot_steps > 0) {
      ds << ", slot occupancy " << std::setprecision(1)
         << 100.0 * static_cast<double>(g.newton_steps) / static_cast<double>(g.slot_steps) << "%";
    }
    d.info(ds.str());
    auto bw = [](double bytes, double secs) {
      return secs > 0.0 ? bytes / secs / 1.0e9 : 0.0;
    };
    std::ostringstream ps;
    ps << std::fixed << std::setprecision(3) << "  phases (s): materialize "
       << g.seconds_materialize << ", mismatch " << g.seconds_mismatch << " ("
       << std::setprecision(1) << bw(g.bytes_mismatch, g.seconds_mismatch)
       << " GB/s), Jacobian " << std::setprecision(3) << g.seconds_jacobian << " ("
       << std::setprecision(1) << bw(g.bytes_jacobian, g.seconds_jacobian)
       << " GB/s), factor " << std::setprecision(3) << g.seconds_factor << " ("
       << std::setprecision(1) << bw(g.bytes_factor, g.seconds_factor)
       << " GB/s), solve " << std::setprecision(3) << g.seconds_solve << " ("
       << std::setprecision(1) << bw(g.bytes_solve, g.seconds_solve)
       << " GB/s), update " << std::setprecision(3) << g.seconds_update
       << ", Q limits " << g.seconds_qlim << ", exchange " << g.seconds_exchange;
    d.info(ps.str());
    std::ostringstream ls;
    ls << "  plan: " << g.jacobian_rows << " rows, " << g.jacobian_nnz
       << " Jacobian entries (standard layout " << g.minimal_nnz << ", superset overhead "
       << std::fixed << std::setprecision(1)
       << (g.minimal_nnz > 0 ? 100.0 * static_cast<double>(g.jacobian_nnz - g.minimal_nnz) /
           static_cast<double>(g.minimal_nnz) : 0.0)
       << "%), " << g.factor_nnz << " factor entries, levels " << g.levels_factor << "/"
       << g.levels_lower << "/" << g.levels_upper << ", planning "
       << std::setprecision(3) << g.plan_seconds << " s";
    d.info(ls.str());
  }
  std::ostringstream ts;
  ts << std::fixed << std::setprecision(3) << "times (s): classify " << d.t_classify
     << ", plan " << d.t_plan << ", CPU-path phase " << cpu_phase
     << ", GPU report phase " << gpu_phase << "; host memory available "
     << hostAvailableBytes() / 1.0e9 << " GB";
  d.info(ts.str());
}

}  // namespace batchpf
}  // namespace gridpack
