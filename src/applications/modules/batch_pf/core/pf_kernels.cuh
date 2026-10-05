/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   pf_kernels.cuh
 * @date   2026-10-05
 *
 * @brief Power flow element functions of the batch engine (blocks B8.2,
 * B8.3, B8.4, B8.6, B8.7), written once for the GPU and the CPU.
 *
 * The formulas are GridPACK's, taken from PFBus::rhsValues(),
 * PFBus::diagonalJacobianValues(), PFBranch::getPQ(),
 * PFBranch::forward/reverseJacobianValues() in the LARGE_MATRIX layout, and
 * PFBus::chkQlim(). Terms are added in the same order GridPACK adds them
 * (edges of a bus in the order of its connected branches), so the results
 * agree with GridPACK up to rounding (guide sections 5.3.2, 8.10).
 *
 * Superset rules (guide 8.3.3): every bus has a 2x2 diagonal block and
 * every edge a 2x2 off-diagonal block. Reference and isolated buses get an
 * identity block and zero coupling; a PV bus keeps its P row and gets a
 * fixed-voltage row (1 on its own V); blocks are zeroed in the Q row of a PV
 * row bus and in the V column of a PV column bus. These are all value
 * changes, so one sparsity pattern serves every case.
 *
 * Index i = item * B + b (member b of item), see executor.cuh. Arithmetic
 * is double precision throughout and no fast-math options are used
 * (precision outranks speed here, guide 8.18).
 */

#ifndef GRIDPACK_BATCHPF_CORE_PF_KERNELS_CUH
#define GRIDPACK_BATCHPF_CORE_PF_KERNELS_CUH

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "gridpack/batchpf/batchpf_plugin.h"

namespace gridpack {
namespace batchpf {

/// Read-only network data shared by all members (placement class M)
struct ModelView {
  int n_bus = 0;
  int n_edge = 0;
  double sbase = 100.0;
  const int *row_start = nullptr;
  const int *edge_col = nullptr;
  const double *ql = nullptr;
  const double *ip = nullptr;
  const double *iq = nullptr;
  const double *yp = nullptr;
  const double *yq = nullptr;
  const double *v_init = nullptr;
  const double *theta_init = nullptr;
  const double *v_base = nullptr;
  const double *theta_base = nullptr;
  const int *base_type = nullptr;
  const double *base_g = nullptr;
  const double *base_b = nullptr;
  const double *base_p0 = nullptr;
  const double *base_q0 = nullptr;
  const double *base_qmax = nullptr;
  const double *base_qmin = nullptr;
  const double *base_eg = nullptr;
  const double *base_eb = nullptr;
  const int *diag_pos = nullptr;  // 4 per bus
  const int *edge_pos = nullptr;  // 4 per edge
};

/// Per-batch working data, interleaved by member (placement class W)
struct BatchView {
  int B = 0;                 // capacity, the stride
  int *type = nullptr;       // n x B
  double *g = nullptr;       // n x B admittance diagonal
  double *b = nullptr;
  double *p0 = nullptr;      // n x B scheduled injection
  double *q0 = nullptr;
  double *qmax = nullptr;    // n x B in-service Q limit totals
  double *qmin = nullptr;
  double *eg = nullptr;      // E x B edge admittance
  double *eb = nullptr;
  double *v = nullptr;       // n x B voltage magnitude
  double *theta = nullptr;   // n x B angle as updated (like PFBus::p_a)
  double *thw = nullptr;     // n x B angle wrapped to [-pi, pi), the value
                             // neighbors see (like *p_vAng_ptr)
  double *pinj = nullptr;    // n x B computed injections (p_Pinj, p_Qinj)
  double *qinj = nullptr;
  double *F = nullptr;       // 2n x B mismatch (right-hand side)
  double *X = nullptr;       // 2n x B Newton step
  double *J = nullptr;       // nnz x B Jacobian values
  int *conv = nullptr;       // n x B Q-limit conversion: 0, +1, -1
  double *qreq = nullptr;    // n x B reactive requirement at conversion
  // per member (B entries)
  const int *m_apply = nullptr;
  const int *m_qcheck = nullptr;
  const int *m_eval = nullptr;
  const int *m_fill = nullptr;
  const int *m_slack = nullptr;
  unsigned long long *m_maxp = nullptr;   // bit pattern of max |dP|
  unsigned long long *m_maxq = nullptr;   // bit pattern of max |dQ|
  int *m_argp = nullptr;
  int *m_argq = nullptr;
  int *m_qviol = nullptr;
  int warm_start = BATCHPF_WARM_START_BASE_CASE;
  double damping = 1.0;
  double qlim_deadband = 0.1;
};

/// One case-update record, flattened over the members being filled
struct UpdateView {
  int n_bus_updates = 0;
  int n_edge_updates = 0;
  const int *bus_member = nullptr;
  const batchpf_bus_update *bus = nullptr;
  const int *edge_member = nullptr;
  const batchpf_edge_update *edge = nullptr;
};

// ---------------------------------------------------------------------
// Helpers usable on host and device
// ---------------------------------------------------------------------

/// GridPACK's angle wrap from PFBus::setValues(): result in [-pi, pi)
__host__ __device__ inline double wrapAngle(double a)
{
  const double pi = 4.0 * atan(1.0);
  return (a >= 0.0) ? fmod(a + pi, 2.0 * pi) - pi : fmod(a - pi, 2.0 * pi) + pi;
}

/// Bits of a non-negative double; their integer order is the value order,
/// so atomicMax on the bits gives the maximum value
__host__ __device__ inline unsigned long long absBits(double x)
{
  const double a = fabs(x);
  unsigned long long u = 0;
  memcpy(&u, &a, sizeof(u));
  return u;
}

__host__ __device__ inline void atomicMaxBits(unsigned long long *addr,
                                              unsigned long long val)
{
#ifdef __CUDA_ARCH__
  atomicMax(addr, val);
#else
  if (val > *addr) *addr = val;
#endif
}

__host__ __device__ inline void atomicMinInt(int *addr, int val)
{
#ifdef __CUDA_ARCH__
  atomicMin(addr, val);
#else
  if (val < *addr) *addr = val;
#endif
}

__host__ __device__ inline void atomicAddInt(int *addr, int val)
{
#ifdef __CUDA_ARCH__
  atomicAdd(addr, val);
#else
  *addr += val;
#endif
}

// ---------------------------------------------------------------------
// B8.2 Case materializer: copy the base network into the slots being
// filled, then apply each case's absolute value changes
// ---------------------------------------------------------------------

struct FillBusBase {
  ModelView m;
  BatchView w;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int64_t k = i / w.B;
    const int bm = static_cast<int>(i % w.B);
    if (!w.m_fill[bm]) return;
    w.type[i] = m.base_type[k];
    w.g[i] = m.base_g[k];
    w.b[i] = m.base_b[k];
    w.p0[i] = m.base_p0[k];
    w.q0[i] = m.base_q0[k];
    w.qmax[i] = m.base_qmax[k];
    w.qmin[i] = m.base_qmin[k];
    w.conv[i] = 0;
    w.qreq[i] = 0.0;
  }
};

struct FillEdgeBase {
  ModelView m;
  BatchView w;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int64_t e = i / w.B;
    const int bm = static_cast<int>(i % w.B);
    if (!w.m_fill[bm]) return;
    w.eg[i] = m.base_eg[e];
    w.eb[i] = m.base_eb[e];
  }
};

struct ApplyBusUpdates {
  BatchView w;
  UpdateView u;
  __host__ __device__ void operator()(int64_t i) const
  {
    const batchpf_bus_update &r = u.bus[i];
    const int64_t pos = static_cast<int64_t>(r.bus) * w.B + u.bus_member[i];
    w.type[pos] = r.type;
    w.g[pos] = r.g_diag;
    w.b[pos] = r.b_diag;
    w.p0[pos] = r.p0;
    w.q0[pos] = r.q0;
    w.qmax[pos] = r.qmax;
    w.qmin[pos] = r.qmin;
  }
};

struct ApplyEdgeUpdates {
  BatchView w;
  UpdateView u;
  __host__ __device__ void operator()(int64_t i) const
  {
    const batchpf_edge_update &r = u.edge[i];
    const int64_t pos = static_cast<int64_t>(r.edge) * w.B + u.edge_member[i];
    w.eg[pos] = r.g;
    w.eb[pos] = r.b;
  }
};

/**
 * Starting voltages. RAW start: the values GridPACK's resetVoltages()
 * restores. Base-case start (ADR-05): the base solution, except that
 * voltage-controlled and reference buses hold their specified magnitude,
 * isolated buses keep their reset values (GridPACK never updates them),
 * and angles are shifted so the case's reference bus sits at the angle
 * GridPACK gives it. The shift matters only when the slack moved.
 */
struct InitState {
  ModelView m;
  BatchView w;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int64_t k = i / w.B;
    const int bm = static_cast<int>(i % w.B);
    if (!w.m_fill[bm]) return;
    const int t = w.type[i];
    double v = m.v_init[k];
    double th = m.theta_init[k];
    if (w.warm_start == BATCHPF_WARM_START_BASE_CASE &&
        t != BATCHPF_BUS_ISOLATED && t != BATCHPF_BUS_REF) {
      const int s = w.m_slack[bm];
      const double shift = m.theta_init[s] - m.theta_base[s];
      th = m.theta_base[k] + shift;
      if (t == BATCHPF_BUS_PQ) v = m.v_base[k];
    }
    w.v[i] = v;
    w.theta[i] = th;
    w.thw[i] = wrapAngle(th);
  }
};

// ---------------------------------------------------------------------
// B8.6 State updater: subtract the Newton step, as PFBus::setValues()
// does (the step was already scaled by the damping factor)
// ---------------------------------------------------------------------

struct ApplyStep {
  BatchView w;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int64_t k = i / w.B;
    const int bm = static_cast<int>(i % w.B);
    if (!w.m_apply[bm]) return;
    const int t = w.type[i];
    if (t == BATCHPF_BUS_REF || t == BATCHPF_BUS_ISOLATED) return;
    const int64_t r = 2 * k * w.B + bm;
    w.theta[i] -= w.X[r];
    if (t == BATCHPF_BUS_PQ) w.v[i] -= w.X[r + w.B];
    w.thw[i] = wrapAngle(w.theta[i]);
  }
};

/// Scale the Newton step by the damping factor (GridPACK: X->scale())
struct ScaleStep {
  BatchView w;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int bm = static_cast<int>(i % w.B);
    if (w.m_eval[bm]) w.X[i] *= w.damping;
  }
};

// ---------------------------------------------------------------------
// B8.3 Mismatch evaluator: one worker per (bus row, member). GridPACK
// gives each bus one sum over its branches; doing the same keeps the order
// of additions, needs no atomics, and makes results reproducible.
// ---------------------------------------------------------------------

struct Mismatch {
  ModelView m;
  BatchView w;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int64_t k = i / w.B;
    const int bm = static_cast<int>(i % w.B);
    if (!w.m_eval[bm]) return;
    const int64_t rp = 2 * k * w.B + bm;
    const int t = w.type[i];
    if (t == BATCHPF_BUS_REF || t == BATCHPF_BUS_ISOLATED) {
      w.F[rp] = 0.0;
      w.F[rp + w.B] = 0.0;
      return;
    }
    const double vk = w.v[i];
    const double thk = w.thw[i];
    double P = 0.0;
    double Q = 0.0;
    for (int e = m.row_start[k]; e < m.row_start[k + 1]; e++) {
      const int64_t j = static_cast<int64_t>(m.edge_col[e]) * w.B + bm;
      const int64_t ep = static_cast<int64_t>(e) * w.B + bm;
      const double gy = w.eg[ep];
      const double by = w.eb[ep];
      const double th = thk - w.thw[j];
      const double cs = cos(th);
      const double sn = sin(th);
      const double vv = vk * w.v[j];
      P += vv * (gy * cs + by * sn);
      Q += vv * (gy * sn - by * cs);
    }
    P += vk * vk * w.g[i];
    Q += vk * vk * (-w.b[i]);
    w.pinj[i] = P;
    w.qinj[i] = Q;
    P -= w.p0[i];
    Q -= w.q0[i];
    const double pzip = m.ip[k] * vk + m.yp[k] * vk * vk;
    const double qzip = m.iq[k] * vk - m.yq[k] * vk * vk;
    P += pzip / m.sbase;
    Q += qzip / m.sbase;
    w.F[rp] = P;
    w.F[rp + w.B] = (t == BATCHPF_BUS_PQ) ? Q : 0.0;
    atomicMaxBits(&w.m_maxp[bm], absBits(P));
    if (t == BATCHPF_BUS_PQ) atomicMaxBits(&w.m_maxq[bm], absBits(Q));
  }
};

/// First bus (lowest index) holding the largest mismatch, as GridPACK's
/// per-iteration report picks it
struct MismatchArgmax {
  BatchView w;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int64_t k = i / w.B;
    const int bm = static_cast<int>(i % w.B);
    if (!w.m_eval[bm]) return;
    const int t = w.type[i];
    if (t == BATCHPF_BUS_REF || t == BATCHPF_BUS_ISOLATED) return;
    const int64_t rp = 2 * k * w.B + bm;
    if (absBits(w.F[rp]) == w.m_maxp[bm]) {
      atomicMinInt(&w.m_argp[bm], static_cast<int>(k));
    }
    if (t == BATCHPF_BUS_PQ && absBits(w.F[rp + w.B]) == w.m_maxq[bm]) {
      atomicMinInt(&w.m_argq[bm], static_cast<int>(k));
    }
  }
};

// ---------------------------------------------------------------------
// B8.4 Jacobian assembler: one worker per (block, member)
// ---------------------------------------------------------------------

struct JacobianDiag {
  ModelView m;
  BatchView w;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int64_t k = i / w.B;
    const int bm = static_cast<int>(i % w.B);
    if (!w.m_eval[bm]) return;
    const int *pos = m.diag_pos + 4 * k;
    const int t = w.type[i];
    double r0 = 1.0, r1 = 0.0, r2 = 0.0, r3 = 1.0;
    if (t == BATCHPF_BUS_PQ || t == BATCHPF_BUS_PV) {
      const double v = w.v[i];
      const double P = w.pinj[i];
      const double Q = w.qinj[i];
      const double gy = w.g[i];
      const double by = w.b[i];
      r0 = -Q - by * v * v;
      if (t == BATCHPF_BUS_PQ) {
        const double dp = m.ip[k] + 2.0 * m.yp[k] * v;
        const double dq = m.iq[k] - 2.0 * m.yq[k] * v;
        r1 = P - gy * v * v;
        r2 = P / v + gy * v;
        r3 = Q / v - by * v;
        r2 += dp / m.sbase;
        r3 += dq / m.sbase;
      }
    }
    w.J[static_cast<int64_t>(pos[0]) * w.B + bm] = r0;
    w.J[static_cast<int64_t>(pos[1]) * w.B + bm] = r1;
    w.J[static_cast<int64_t>(pos[2]) * w.B + bm] = r2;
    w.J[static_cast<int64_t>(pos[3]) * w.B + bm] = r3;
  }
};

struct JacobianEdge {
  ModelView m;
  BatchView w;
  const int *edge_row = nullptr;   // row bus of each edge
  __host__ __device__ void operator()(int64_t i) const
  {
    const int64_t e = i / w.B;
    const int bm = static_cast<int>(i % w.B);
    if (!w.m_eval[bm]) return;
    const int *pos = m.edge_pos + 4 * e;
    const int64_t kb = static_cast<int64_t>(edge_row[e]) * w.B + bm;
    const int64_t mb = static_cast<int64_t>(m.edge_col[e]) * w.B + bm;
    const int tk = w.type[kb];
    const int tm = w.type[mb];
    double r0 = 0.0, r1 = 0.0, r2 = 0.0, r3 = 0.0;
    const bool active = (tk == BATCHPF_BUS_PQ || tk == BATCHPF_BUS_PV) &&
                        (tm == BATCHPF_BUS_PQ || tm == BATCHPF_BUS_PV);
    if (active) {
      const double gy = w.eg[i];
      const double by = w.eb[i];
      const double th = w.thw[kb] - w.thw[mb];
      const double cs = cos(th);
      const double sn = sin(th);
      const double vk = w.v[kb];
      const double vm = w.v[mb];
      r0 = (gy * sn - by * cs) * (vk * vm);
      r1 = (gy * cs + by * sn) * -(vk * vm);
      r2 = (gy * cs + by * sn) * vk;
      r3 = (gy * sn - by * cs) * vk;
      if (tk == BATCHPF_BUS_PV) { r1 = 0.0; r3 = 0.0; }
      if (tm == BATCHPF_BUS_PV) { r2 = 0.0; r3 = 0.0; }
    }
    w.J[static_cast<int64_t>(pos[0]) * w.B + bm] = r0;
    w.J[static_cast<int64_t>(pos[1]) * w.B + bm] = r1;
    w.J[static_cast<int64_t>(pos[2]) * w.B + bm] = r2;
    w.J[static_cast<int64_t>(pos[3]) * w.B + bm] = r3;
  }
};

// ---------------------------------------------------------------------
// B8.7 Control loop: reactive-limit check of PFBus::chkQlim(). A PV bus
// whose requirement is outside its total limits (plus the dead band)
// becomes PQ with its generators at the limit. GridPACK's split of Q among
// generators only matters for reporting, which GridPACK redoes itself.
// ---------------------------------------------------------------------

struct QlimCheck {
  ModelView m;
  BatchView w;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int64_t k = i / w.B;
    const int bm = static_cast<int>(i % w.B);
    if (!w.m_qcheck[bm]) return;
    if (w.type[i] != BATCHPF_BUS_PV) return;
    const double v = w.v[i];
    double ql = m.ql[k];
    ql += m.iq[k] * v - m.yq[k] * v * v;
    const double q_required = w.qinj[i] * m.sbase + ql;
    int side = 0;
    if (q_required > w.qmax[i] + w.qlim_deadband) {
      side = 1;
    } else if (q_required < w.qmin[i] - w.qlim_deadband) {
      side = -1;
    }
    if (side == 0) return;
    const double qg = (side > 0) ? w.qmax[i] : w.qmin[i];
    w.type[i] = BATCHPF_BUS_PQ;
    w.q0[i] = (qg - m.ql[k]) / m.sbase;
    w.conv[i] = side;
    w.qreq[i] = q_required;
    atomicAddInt(&w.m_qviol[bm], 1);
  }
};

}  // namespace batchpf
}  // namespace gridpack

#endif
