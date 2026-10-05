/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   engine_internal.cuh
 * @date   2026-10-05
 *
 * @brief Buffers and helper kernels shared by engine.cu (batch execution)
 * and engine_setup.cu (planning and allocation). Internal to the plugin.
 */

#ifndef GRIDPACK_BATCHPF_CORE_ENGINE_INTERNAL_CUH
#define GRIDPACK_BATCHPF_CORE_ENGINE_INTERNAL_CUH

#include <algorithm>
#include <climits>
#include <cstring>
#include <vector>

#include "engine.hpp"
#include "executor.cuh"
#include "pf_kernels.cuh"

namespace gridpack {
namespace batchpf {

/// Telemetry phases (guide 8.12)
enum Phase {
  PH_MATERIALIZE, PH_MISMATCH, PH_JACOBIAN, PH_FACTOR, PH_SOLVE,
  PH_UPDATE, PH_QLIM, PH_EXCHANGE, PH_COUNT
};

inline double bitsToDouble(unsigned long long u)
{
  double d = 0.0;
  std::memcpy(&d, &u, sizeof(d));
  return d;
}

/**
 * Exchange buffer (placement class X). With pinned memory the GPU reads the
 * host copy directly (zero-copy, worthwhile on integrated GPUs, CUDA Best
 * Practices Guide 10.1.3); otherwise a device copy is kept and refreshed
 * with explicit copies at batch boundaries.
 */
template <class T>
class Exchange {
 public:
  void reserve(std::size_t count, bool on_device, bool pinned, cudaStream_t stream)
  {
    if (count <= p_capacity && on_device == p_on_device && pinned == p_pinned) {
      return;
    }
    p_capacity = std::max<std::size_t>(count, 1);
    p_on_device = on_device;
    p_pinned = pinned;
    if (!on_device) {
      p_host.allocate(MemoryKind::Host, p_capacity);
    } else if (pinned) {
      p_host.allocate(MemoryKind::Pinned, p_capacity);
    } else {
      p_host.allocate(MemoryKind::Pinned, p_capacity);
      p_dev.allocate(MemoryKind::Device, p_capacity, stream);
    }
  }
  T *host() { return p_host.data(); }
  const T *device() const
  {
    return (p_on_device && !p_pinned) ? p_dev.data() : p_host.data();
  }
  T *deviceMutable()
  {
    return (p_on_device && !p_pinned) ? p_dev.data() : p_host.data();
  }
  void toDevice(std::size_t count, cudaStream_t stream)
  {
    if (p_on_device && !p_pinned) p_dev.upload(p_host.data(), count, stream);
  }
  void toHost(std::size_t count, cudaStream_t stream)
  {
    if (p_on_device && !p_pinned) p_dev.download(p_host.data(), count, stream);
  }

 private:
  std::size_t p_capacity = 0;
  bool p_on_device = false;
  bool p_pinned = false;
  Buffer<T> p_host;
  Buffer<T> p_dev;
};

/// Copy per-slot state into a compact output block (case-major)
struct GatherState {
  BatchView w;
  int n = 0;
  const int *slots = nullptr;
  double *ov = nullptr;
  double *oth = nullptr;
  int *oconv = nullptr;
  double *oq = nullptr;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int64_t li = i / n;
    const int64_t k = i % n;
    const int64_t src = k * w.B + slots[li];
    ov[i] = w.v[src];
    oth[i] = w.theta[src];
    oconv[i] = w.conv[src];
    oq[i] = w.qreq[src];
  }
};

/// Replicate one vector into every slot (batch-size sweep)
struct Broadcast {
  const double *src = nullptr;
  double *dst = nullptr;
  int B = 0;
  __host__ __device__ void operator()(int64_t i) const { dst[i] = src[i / B]; }
};

/**
 * Linear residual |J X - F| per row and member, for the residual health
 * check (guide 8.5). Uses the undamped step.
 */
struct Residual {
  BatchView w;
  const int *row_ptr = nullptr;
  const int *col_idx = nullptr;
  unsigned long long *m_res = nullptr;
  unsigned long long *m_rhs = nullptr;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int64_t r = i / w.B;
    const int bm = static_cast<int>(i % w.B);
    if (!w.m_eval[bm]) return;
    double s = -w.F[i];
    for (int p = row_ptr[r]; p < row_ptr[r + 1]; p++) {
      s += w.J[static_cast<int64_t>(p) * w.B + bm] *
           w.X[static_cast<int64_t>(col_idx[p]) * w.B + bm];
    }
    atomicMaxBits(&m_res[bm], absBits(s));
    atomicMaxBits(&m_rhs[bm], absBits(w.F[i]));
  }
};

// -------------------------------------------------------------------------
// Buffers
// -------------------------------------------------------------------------

struct EngineBuffers {
  MemoryKind work = MemoryKind::Host;
  // model (class M)
  Buffer<int> row_start, edge_col, edge_row, base_type, diag_pos, edge_pos;
  Buffer<int> jrow_ptr, jcol_idx;
  Buffer<double> ql, ip, iq, yp, yq, v_init, theta_init, v_base, theta_base;
  Buffer<double> base_g, base_b, base_p0, base_q0, base_qmax, base_qmin;
  Buffer<double> base_eg, base_eb, ref_values, ref_rhs;
  // working (class W)
  Buffer<int> type, conv;
  Buffer<double> g, b, p0, q0, qmax, qmin, eg, eb, v, theta, thw;
  Buffer<double> pinj, qinj, F, X, J, qreq;
  // per slot, on the device
  Buffer<int> m_apply, m_qcheck, m_eval, m_fill, m_slack;
  Buffer<int> m_argp, m_argq, m_qviol, m_status;
  Buffer<unsigned long long> m_maxp, m_maxq, m_res, m_rhs;
  // per slot, host side
  std::vector<int> h_apply, h_qcheck, h_eval, h_fill, h_slack;
  std::vector<int> h_argp, h_argq, h_qviol, h_status;
  std::vector<unsigned long long> h_maxp, h_maxq, h_res, h_rhs;
  // exchange (class X)
  Exchange<int> u_bus_member, u_edge_member, gather_slots, fill_slots;
  Exchange<batchpf_bus_update> u_bus;
  Exchange<batchpf_edge_update> u_edge;
  Exchange<double> out_v, out_theta, out_q;
  Exchange<int> out_conv;
  // timing
  std::vector<Event> ev_start, ev_stop;
  std::vector<double> host_seconds;
  std::vector<bool> used;
};

/// View of the model buffers for the kernels
inline ModelView modelView(const ModelHost &h, EngineBuffers &d)
{
  ModelView m;
  m.n_bus = h.n_bus;
  m.n_edge = h.n_edge;
  m.sbase = h.sbase;
  m.row_start = d.row_start.data();
  m.edge_col = d.edge_col.data();
  m.ql = d.ql.data();
  m.ip = d.ip.data();
  m.iq = d.iq.data();
  m.yp = d.yp.data();
  m.yq = d.yq.data();
  m.v_init = d.v_init.data();
  m.theta_init = d.theta_init.data();
  m.v_base = d.v_base.data();
  m.theta_base = d.theta_base.data();
  m.base_type = d.base_type.data();
  m.base_g = d.base_g.data();
  m.base_b = d.base_b.data();
  m.base_p0 = d.base_p0.data();
  m.base_q0 = d.base_q0.data();
  m.base_qmax = d.base_qmax.data();
  m.base_qmin = d.base_qmin.data();
  m.base_eg = d.base_eg.data();
  m.base_eb = d.base_eb.data();
  m.diag_pos = d.diag_pos.data();
  m.edge_pos = d.edge_pos.data();
  return m;
}

/// View of the working buffers for the kernels
inline BatchView batchView(EngineBuffers &d, int B, const batchpf_solver_params &prm)
{
  BatchView w;
  w.B = B;
  w.type = d.type.data();
  w.g = d.g.data();
  w.b = d.b.data();
  w.p0 = d.p0.data();
  w.q0 = d.q0.data();
  w.qmax = d.qmax.data();
  w.qmin = d.qmin.data();
  w.eg = d.eg.data();
  w.eb = d.eb.data();
  w.v = d.v.data();
  w.theta = d.theta.data();
  w.thw = d.thw.data();
  w.pinj = d.pinj.data();
  w.qinj = d.qinj.data();
  w.F = d.F.data();
  w.X = d.X.data();
  w.J = d.J.data();
  w.conv = d.conv.data();
  w.qreq = d.qreq.data();
  w.m_apply = d.m_apply.data();
  w.m_qcheck = d.m_qcheck.data();
  w.m_eval = d.m_eval.data();
  w.m_fill = d.m_fill.data();
  w.m_slack = d.m_slack.data();
  w.m_maxp = d.m_maxp.data();
  w.m_maxq = d.m_maxq.data();
  w.m_argp = d.m_argp.data();
  w.m_argq = d.m_argq.data();
  w.m_qviol = d.m_qviol.data();
  w.warm_start = prm.warm_start;
  w.damping = prm.damping_factor;
  w.qlim_deadband = prm.qlim_deadband;
  return w;
}

}  // namespace batchpf
}  // namespace gridpack

#endif
