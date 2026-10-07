/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   engine_setup.cu
 * @date   2026-10-05
 *
 * @brief Engine planning and allocation (block B8.1, with the planner B6):
 * reference Jacobian, fixed-order LU plan, buffers for B slots, backend
 * setup, and the timing used by the batch-size sweep.
 */

#include <chrono>
#include <sstream>

#include "engine_internal.cuh"

namespace gridpack {
namespace batchpf {

namespace {

template <class T>
void uploadModel(Buffer<T> &dst, MemoryKind kind, const std::vector<T> &src,
                 cudaStream_t stream)
{
  dst.allocate(kind, src.size(), stream);
  dst.upload(src.data(), src.size(), stream);
}

}  // namespace

/**
 * Reference Jacobian: the base case at the base-case solution, with the
 * base bus roles, evaluated by the same functors on the CPU (one slot)
 */
void Engine::referenceJacobian(std::vector<double> *values,
                               std::vector<double> *rhs)
{
  const ModelHost &h = p_model;
  const int n = h.n_bus;
  std::vector<int> type = h.bus_type, conv(n, 0), one(1, 1), slack(1, h.base_slack);
  std::vector<double> g = h.g, b = h.b, p0 = h.p0, q0 = h.q0, qmax = h.qmax;
  std::vector<double> qmin = h.qmin, eg = h.eg, eb = h.eb;
  std::vector<double> v(n), th(n), thw(n), pinj(n), qinj(n), F(2 * n), X(2 * n);
  std::vector<double> qreq(n, 0.0);
  values->assign(static_cast<std::size_t>(p_pattern.nnz), 0.0);
  std::vector<unsigned long long> maxp(1, 0), maxq(1, 0);
  std::vector<int> argp(1, INT_MAX), argq(1, INT_MAX), qv(1, 0);
  ModelView m;
  m.n_bus = n;
  m.n_edge = h.n_edge;
  m.sbase = h.sbase;
  m.row_start = h.row_start.data();
  m.edge_col = h.edge_col.data();
  m.ql = h.ql.data();
  m.dg_q = h.dg_q.data();
  m.dc_q = h.dc_q.data();
  m.ip = h.ip.data();
  m.iq = h.iq.data();
  m.yp = h.yp.data();
  m.yq = h.yq.data();
  m.v_init = h.v_init.data();
  m.theta_init = h.theta_init.data();
  m.v_base = h.v_base.data();
  m.theta_base = h.theta_base.data();
  m.diag_pos = p_pattern.diag_pos.data();
  m.edge_pos = p_pattern.edge_pos.data();
  m.dc_slot = h.dc_slot.data();
  BatchView w;
  w.B = 1;
  w.type = type.data();
  w.g = g.data();
  w.b = b.data();
  w.p0 = p0.data();
  w.q0 = q0.data();
  w.qmax = qmax.data();
  w.qmin = qmin.data();
  w.eg = eg.data();
  w.eb = eb.data();
  w.v = v.data();
  w.theta = th.data();
  w.thw = thw.data();
  w.pinj = pinj.data();
  w.qinj = qinj.data();
  w.F = F.data();
  w.X = X.data();
  w.J = values->data();
  w.conv = conv.data();
  w.qreq = qreq.data();
  w.m_apply = one.data();
  w.m_qcheck = one.data();
  w.m_eval = one.data();
  w.m_fill = one.data();
  w.m_slack = slack.data();
  w.m_maxp = maxp.data();
  w.m_maxq = maxq.data();
  w.m_argp = argp.data();
  w.m_argq = argq.data();
  w.m_qviol = qv.data();
  w.warm_start = BATCHPF_WARM_START_BASE_CASE;
  const Executor host(false, nullptr, 0);
  const int slot0 = 0;
  host.run(n, InitState{m, w, &slot0, 1}, "InitState");
  host.run(n, Mismatch{m, w}, "Mismatch");
  host.run(n, JacobianDiag{m, w}, "JacobianDiag");
  host.run(h.n_edge, JacobianEdge{m, w, h.edge_row.data()}, "JacobianEdge");
  *rhs = F;
}

void Engine::plan(const batchpf_solver_params &params)
{
  const auto t0 = std::chrono::steady_clock::now();
  p_params = params;
  p_rules.tolerance = params.tolerance;
  p_rules.max_iteration = params.max_iteration;
  p_rules.pf_qlim = params.pf_qlim != 0;
  p_rules.ca_qlim = params.ca_qlim != 0;
  p_rules.max_controller_iterations = params.max_controller_iterations;
  p_rules.check_nonfinite = p_config.check_nonfinite;
  p_rules.residual_limit = p_config.residual_limit;
  p_rules.dc_lines = !p_model.dc_line.empty();
  p_rules.hvdc_tolerance = params.hvdc_tolerance;
  p_pattern = buildPattern(p_model.n_bus, p_model.row_start, p_model.edge_col);
  referenceJacobian(&p_ref_values, &p_ref_rhs);
  p_lu = planLu(p_pattern, p_ref_values, p_config.ordering,
                p_config.pivot_tolerance);
  const auto t1 = std::chrono::steady_clock::now();
  p_diag.plan_seconds = std::chrono::duration<double>(t1 - t0).count();
  p_diag.jacobian_rows = p_pattern.n_rows;
  p_diag.jacobian_nnz = p_pattern.nnz;
  p_diag.minimal_nnz = minimalNnz(p_model.n_bus, p_model.row_start,
                                  p_model.edge_col, p_model.bus_type);
  p_diag.factor_nnz = p_lu.nnz;
  p_diag.factor_flops = p_lu.flops;
  p_diag.levels_factor = static_cast<int32_t>(p_lu.lev_ptr.size()) - 1;
  p_diag.levels_lower = static_cast<int32_t>(p_lu.llev_ptr.size()) - 1;
  p_diag.levels_upper = static_cast<int32_t>(p_lu.ulev_ptr.size()) - 1;
  std::ostringstream os;
  os << "planned superset Jacobian: " << p_pattern.n_rows << " rows, "
     << p_pattern.nnz << " entries (standard layout of the base case: "
     << p_diag.minimal_nnz << "), factor entries " << p_lu.nnz << ", "
     << p_lu.flops << " multiply-adds per factorization, levels "
     << p_diag.levels_factor << "/" << p_diag.levels_lower << "/"
     << p_diag.levels_upper << " (factor/forward/backward), "
     << p_diag.plan_seconds << " s";
  p_log.info(os.str());
}

double Engine::bytesPerSlot() const
{
  const double n = p_model.n_bus;
  const double e = p_model.n_edge;
  // bus arrays (type, conv, 15 doubles), edge arrays, F/X, Jacobian,
  // factor values plus backend workspace (one more vector)
  const double bus = n * (2 * 4 + 15 * 8);
  const double edge = e * 2 * 8;
  const double vec = 2 * n * 8 * 3;
  const double jac = static_cast<double>(p_pattern.nnz) * 8;
  const double lu = static_cast<double>(p_lu.nnz) * 8;
  // dc lines: status and operating point per line, four values per
  // converter bus
  const double dc = static_cast<double>(p_model.dc_line.size()) *
                        (sizeof(int) + sizeof(HVDCSolution)) +
                    static_cast<double>(p_model.dc_bus.size()) * 4 * 8;
  return bus + edge + vec + jac + lu + dc;
}

double Engine::sharedBytes() const
{
  const double idx = static_cast<double>(p_lu.dst.size() + p_lu.seg_upos.size() * 4 +
                                         p_lu.a_to_lu.size() + p_lu.lrow_col.size() * 2 +
                                         p_lu.urow_col.size() * 2) * 4;
  const double model = (p_model.n_bus * 22.0 + p_model.n_edge * 4.0) * 8 +
                       static_cast<double>(p_model.dc_line.size()) *
                           (sizeof(HVDCLineData) + sizeof(HVDCSolution));
  return idx + model + static_cast<double>(p_pattern.nnz) * 12;
}

void Engine::allocate(int capacity)
{
  if (capacity <= 0) throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "batch capacity must be positive");
  EngineBuffers &d = *p_buf;
  const bool dev = p_config.on_device;
  const cudaStream_t st = p_config.stream;
  d.work = dev ? MemoryKind::Device : MemoryKind::Host;
  const MemoryKind mk = d.work;
  const ModelHost &h = p_model;
  if (d.row_start.size() == 0) {
    uploadModel(d.row_start, mk, h.row_start, st);
    uploadModel(d.edge_col, mk, h.edge_col, st);
    uploadModel(d.edge_row, mk, h.edge_row, st);
    uploadModel(d.base_type, mk, h.bus_type, st);
    uploadModel(d.diag_pos, mk, p_pattern.diag_pos, st);
    uploadModel(d.edge_pos, mk, p_pattern.edge_pos, st);
    uploadModel(d.jrow_ptr, mk, p_pattern.row_ptr, st);
    uploadModel(d.jcol_idx, mk, p_pattern.col_idx, st);
    uploadModel(d.ql, mk, h.ql, st);
    uploadModel(d.dg_q, mk, h.dg_q, st);
    uploadModel(d.dc_q, mk, h.dc_q, st);
    uploadModel(d.ip, mk, h.ip, st);
    uploadModel(d.iq, mk, h.iq, st);
    uploadModel(d.yp, mk, h.yp, st);
    uploadModel(d.yq, mk, h.yq, st);
    uploadModel(d.v_init, mk, h.v_init, st);
    uploadModel(d.theta_init, mk, h.theta_init, st);
    uploadModel(d.v_base, mk, h.v_base, st);
    uploadModel(d.theta_base, mk, h.theta_base, st);
    uploadModel(d.base_g, mk, h.g, st);
    uploadModel(d.base_b, mk, h.b, st);
    uploadModel(d.base_p0, mk, h.p0, st);
    uploadModel(d.base_q0, mk, h.q0, st);
    uploadModel(d.base_qmax, mk, h.qmax, st);
    uploadModel(d.base_qmin, mk, h.qmin, st);
    uploadModel(d.base_eg, mk, h.eg, st);
    uploadModel(d.base_eb, mk, h.eb, st);
    uploadModel(d.ref_values, mk, p_ref_values, st);
    uploadModel(d.ref_rhs, mk, p_ref_rhs, st);
    uploadModel(d.dc_line, mk, h.dc_line, st);
    uploadModel(d.dc_ref, mk, h.dc_ref, st);
    uploadModel(d.dc_cbus, mk, h.dc_bus, st);
    uploadModel(d.dc_slot, mk, h.dc_slot, st);
    uploadModel(d.dc_p, mk, h.dc_p, st);
  }
  // release the old backend before its buffers
  p_backend.reset();
  p_B = capacity;
  const std::size_t B = static_cast<std::size_t>(capacity);
  const std::size_t n = static_cast<std::size_t>(h.n_bus) * B;
  const std::size_t e = static_cast<std::size_t>(h.n_edge) * B;
  d.type.allocate(mk, n, st);
  d.conv.allocate(mk, n, st);
  for (Buffer<double> *x : {&d.g, &d.b, &d.p0, &d.q0, &d.qmax, &d.qmin, &d.v,
                            &d.theta, &d.thw, &d.pinj, &d.qinj, &d.qreq}) {
    x->allocate(mk, n, st);
  }
  d.eg.allocate(mk, e, st);
  d.eb.allocate(mk, e, st);
  const std::size_t nl = h.dc_line.size() * B;
  const std::size_t nc = h.dc_bus.size() * B;
  d.dc_status.allocate(mk, nl, st);
  d.dc_sol.allocate(mk, nl, st);
  for (Buffer<double> *x : {&d.dc_pw, &d.dc_qw, &d.dc_p0s, &d.dc_q0s}) {
    x->allocate(mk, nc, st);
    x->zero(st);
  }
  d.F.allocate(mk, 2 * n, st);
  d.X.allocate(mk, 2 * n, st);
  d.J.allocate(mk, static_cast<std::size_t>(p_pattern.nnz) * B, st);
  d.F.zero(st);
  d.X.zero(st);
  d.J.zero(st);
  for (Buffer<int> *x : {&d.m_apply, &d.m_qcheck, &d.m_eval, &d.m_fill,
                         &d.m_slack, &d.m_argp, &d.m_argq, &d.m_qviol,
                         &d.m_status, &d.m_dcstart, &d.m_dccheck}) {
    x->allocate(mk, B, st);
  }
  for (Buffer<unsigned long long> *x : {&d.m_maxp, &d.m_maxq, &d.m_res, &d.m_rhs,
                                        &d.m_dcchange}) {
    x->allocate(mk, B, st);
  }
  for (std::vector<int> *x : {&d.h_apply, &d.h_qcheck, &d.h_eval, &d.h_fill,
                              &d.h_slack, &d.h_argp, &d.h_argq, &d.h_qviol,
                              &d.h_status, &d.h_dcstart, &d.h_dccheck}) {
    x->assign(B, 0);
  }
  for (std::vector<unsigned long long> *x : {&d.h_maxp, &d.h_maxq, &d.h_res, &d.h_rhs,
                                             &d.h_dcchange}) {
    x->assign(B, 0);
  }
  d.gather_slots.reserve(B, dev, p_config.exchange_pinned, st);
  d.fill_slots.reserve(B, dev, p_config.exchange_pinned, st);
  const std::size_t out = static_cast<std::size_t>(h.n_bus) * B;
  d.out_v.reserve(out, dev, p_config.exchange_pinned, st);
  d.out_theta.reserve(out, dev, p_config.exchange_pinned, st);
  d.out_q.reserve(out, dev, p_config.exchange_pinned, st);
  d.out_conv.reserve(out, dev, p_config.exchange_pinned, st);
  d.u_dc_status.reserve(nl, dev, p_config.exchange_pinned, st);
  d.out_dc.reserve(nl, dev, p_config.exchange_pinned, st);
  if (dev && d.ev_start.empty()) {
    for (int i = 0; i < PH_COUNT; i++) {
      d.ev_start.emplace_back(true);
      d.ev_stop.emplace_back(true);
    }
  }
  d.host_seconds.assign(PH_COUNT, 0.0);
  d.used.assign(PH_COUNT, false);
  p_slots.assign(B, SlotState());
  p_slot_job.assign(B, nullptr);
  p_results.assign(B, StepResult());

  BackendSetup bs;
  bs.pattern = &p_pattern;
  bs.lu = &p_lu;
  bs.reference_values = &p_ref_values;
  bs.capacity = capacity;
  bs.stream = st;
  bs.device = p_config.device;
  bs.threads_per_block = p_config.threads_per_block;
  bs.refinement_steps = p_config.refinement_steps;
  bs.pivot_limit = p_config.pivot_limit;
  bs.host_solve = p_config.host_solve;
  bs.logger = p_log;
  if (p_config.backend == BATCHPF_BACKEND_ALG2) {
    p_backend = makeAlg2Backend(bs);
  } else if (p_config.backend == BATCHPF_BACKEND_CPU_REFERENCE) {
    p_backend = makeCpuReferenceBackend(bs);
  } else if (p_config.backend == BATCHPF_BACKEND_CUDSS) {
    std::string why;
    p_backend = loadPluginBackend(p_config.plugin_dir, "cudss", bs, &why);
    if (!p_backend) throw Error(BATCHPF_ERR_UNAVAILABLE, "cuDSS backend: " + why);
  } else {
    throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "no concrete backend selected");
  }
  if (dev) cudaCheck(cudaStreamSynchronize(st), "allocate");
}

double Engine::timeReferenceSolve(int repetitions)
{
  EngineBuffers &d = *p_buf;
  const Executor ex(p_config.on_device, p_config.stream, p_config.threads_per_block);
  const int64_t nnzB = p_pattern.nnz * p_B;
  const int64_t rowsB = static_cast<int64_t>(p_pattern.n_rows) * p_B;
  std::fill(d.h_eval.begin(), d.h_eval.end(), 1);
  d.m_eval.upload(d.h_eval.data(), p_B, p_config.stream);
  ex.run(nnzB, Broadcast{d.ref_values.data(), d.J.data(), p_B}, "Broadcast");
  ex.run(rowsB, Broadcast{d.ref_rhs.data(), d.F.data(), p_B}, "Broadcast");
  // warm-up, then timed repetitions
  p_backend->refactorize(d.J.span(), d.m_eval.span(), d.m_status.span());
  p_backend->solve(d.F.span(), d.X.span(), d.m_eval.span(), d.m_status.span());
  if (p_config.on_device) cudaCheck(cudaStreamSynchronize(p_config.stream), "sweep");
  const auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repetitions; r++) {
    p_backend->refactorize(d.J.span(), d.m_eval.span(), d.m_status.span());
    p_backend->solve(d.F.span(), d.X.span(), d.m_eval.span(), d.m_status.span());
  }
  if (p_config.on_device) cudaCheck(cudaStreamSynchronize(p_config.stream), "sweep");
  const auto t1 = std::chrono::steady_clock::now();
  std::fill(d.h_eval.begin(), d.h_eval.end(), 0);
  const double secs = std::chrono::duration<double>(t1 - t0).count();
  return secs / (static_cast<double>(repetitions) * p_B);
}

}  // namespace batchpf
}  // namespace gridpack
