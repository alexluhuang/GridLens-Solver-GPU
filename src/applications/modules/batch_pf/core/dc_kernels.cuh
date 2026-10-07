/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   dc_kernels.cuh
 * @date   2026-10-07
 *
 * @brief Two-terminal dc lines in the batch engine: GridPACK's sequential
 * ac/dc method (PFFactoryModule::startHVDC() and updateHVDC()), written once
 * for the GPU and the CPU like the power flow element functions.
 *
 * A dc line enters the ac equations only through the power its converters
 * draw at their buses, a value change of the scheduled injection. The
 * superset Jacobian pattern, the LU plan and the batch therefore stay as
 * they are (Zhou et al. 2017: one sparsity pattern for every member); the
 * converters are re-solved between Newton solves at the new ac voltages,
 * the sequential method (Khan and Bhowmick, ch. 3.3.2). The converter
 * equations are GridPACK's own (pf_hvdc.hpp), so both paths solve the same
 * model.
 *
 * Per member, in GridPACK's order:
 *  - at the start of each solve, lines in service with both converter
 *    buses connected start from the reference (base-case) operating point,
 *    the others are blocked (DcStart);
 *  - after each converged Newton loop, before its last step is applied,
 *    every line is solved at the voltage magnitudes of its converter buses;
 *    the largest change in converter P or Q decides whether the controller
 *    loop repeats (DcUpdate).
 * The converter power then sets the scheduled injection of its bus: P as a
 * change from the case's starting value, and Q the same way unless a
 * reactive-limit check fixed the bus's generators at a limit, in which case
 * Q is generation at the limit less the bus's demand, as in setSBus().
 *
 * Index of the per-member functions: the member (one worker each; a member
 * has only a few lines). Per-line and per-converter-bus arrays are
 * interleaved by member like the bus arrays (item * B + member).
 */

#ifndef GRIDPACK_BATCHPF_CORE_DC_KERNELS_CUH
#define GRIDPACK_BATCHPF_CORE_DC_KERNELS_CUH

#include <cuda_runtime.h>

#include <cstdint>

#include "dc_records.hpp"
#include "pf_kernels.cuh"

namespace gridpack {
namespace batchpf {

/// dc line data shared by all members (placement class M)
struct DcModelView {
  int n_line = 0;
  int n_cbus = 0;                       // converter buses
  const HVDCLineData *line = nullptr;   // converter buses are local indices
  const HVDCSolution *ref = nullptr;    // operating point a solve starts from
  const int *cbus = nullptr;            // n_cbus: bus of each converter bus
  const int *slot = nullptr;            // n_bus: position in cbus, or -1
  const double *dc_p = nullptr;         // n_bus: converter P (MW) and Q
  const double *dc_q = nullptr;         // (MVAr) in the model's p0, q0
};

/// Per-member dc line state (placement class W)
struct DcBatchView {
  int B = 0;
  int *status = nullptr;                // n_line x B: 1 in service
  HVDCSolution *sol = nullptr;          // n_line x B: operating point in use
  double *p = nullptr;                  // n_cbus x B: converter power drawn
  double *q = nullptr;                  //   now (MW, MVAr)
  double *p0s = nullptr;                // n_cbus x B: p0, q0 at case start
  double *q0s = nullptr;
  const int *m_start = nullptr;         // per member: DcStart requested
  const int *m_check = nullptr;         // per member: DcUpdate requested
  unsigned long long *m_change = nullptr;   // per member: bits of the
                                            // largest change (pu)
};

/// Larger of two values, as std::max picks it
__host__ __device__ inline double maxOf(double a, double b) { return (a < b) ? b : a; }

/// True if a converter bus of the line is outside the solved network
__host__ __device__ inline bool dcIsolated(const BatchView &w, const HVDCLineData &l, int bm)
{
  return w.type[static_cast<int64_t>(l.rect.bus) * w.B + bm] == BATCHPF_BUS_ISOLATED ||
         w.type[static_cast<int64_t>(l.inv.bus) * w.B + bm] == BATCHPF_BUS_ISOLATED;
}

/**
 * Converter power at each converter bus from the lines' operating points
 * (PFFactoryModule::applyHVDCInjections()), and the scheduled injections
 * of those buses
 */
__host__ __device__ inline void applyDcInjections(const ModelView &m, const BatchView &w,
                                                  const DcModelView &dm,
                                                  const DcBatchView &dw, int bm)
{
  for (int c = 0; c < dm.n_cbus; c++) {
    dw.p[static_cast<int64_t>(c) * dw.B + bm] = 0.0;
    dw.q[static_cast<int64_t>(c) * dw.B + bm] = 0.0;
  }
  for (int l = 0; l < dm.n_line; l++) {
    const HVDCSolution &s = dw.sol[static_cast<int64_t>(l) * dw.B + bm];
    const int64_t kr = static_cast<int64_t>(dm.slot[dm.line[l].rect.bus]) * dw.B + bm;
    const int64_t ki = static_cast<int64_t>(dm.slot[dm.line[l].inv.bus]) * dw.B + bm;
    dw.p[kr] += s.rect.p;
    dw.q[kr] += s.rect.q;
    dw.p[ki] -= s.inv.p;
    dw.q[ki] += s.inv.q;
  }
  for (int c = 0; c < dm.n_cbus; c++) {
    const int k = dm.cbus[c];
    const int64_t i = static_cast<int64_t>(k) * w.B + bm;
    const int64_t ci = static_cast<int64_t>(c) * dw.B + bm;
    w.p0[i] = dw.p0s[ci] + (dm.dc_p[k] - dw.p[ci]) / m.sbase;
    if (w.conv[i] != 0) {
      const double qg = (w.conv[i] > 0) ? w.qmax[i] : w.qmin[i];
      w.q0[i] = (qg - ((m.ql[k] - m.dg_q[k]) + dw.q[ci])) / m.sbase;
    } else {
      w.q0[i] = dw.q0s[ci] + (dm.dc_q[k] - dw.q[ci]) / m.sbase;
    }
  }
}

/// Line statuses of the members being filled; u holds n_line per member
struct FillDcStatus {
  DcModelView dm;
  DcBatchView dw;
  const int *slots = nullptr;
  int nfill = 0;
  const int *u = nullptr;
  __host__ __device__ void operator()(int64_t j) const
  {
    const int64_t l = j / nfill;
    const int64_t jj = j % nfill;
    dw.status[l * dw.B + slots[jj]] = u[jj * dm.n_line + l];
  }
};

/// Scheduled injections of the converter buses at case start
struct CaptureDcStart {
  BatchView w;
  DcModelView dm;
  DcBatchView dw;
  const int *slots = nullptr;
  int nfill = 0;
  __host__ __device__ void operator()(int64_t j) const
  {
    const int64_t c = j / nfill;
    const int bm = slots[j % nfill];
    const int64_t i = static_cast<int64_t>(dm.cbus[c]) * w.B + bm;
    dw.p0s[c * dw.B + bm] = w.p0[i];
    dw.q0s[c * dw.B + bm] = w.q0[i];
  }
};

/// Start of a solve (PFFactoryModule::startHVDC())
struct DcStart {
  ModelView m;
  BatchView w;
  DcModelView dm;
  DcBatchView dw;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int bm = static_cast<int>(i);
    if (!dw.m_start[bm]) return;
    for (int l = 0; l < dm.n_line; l++) {
      const int64_t li = static_cast<int64_t>(l) * dw.B + bm;
      const bool on = dw.status[li] != 0 && !dcIsolated(w, dm.line[l], bm);
      dw.sol[li] = on ? dm.ref[l] : gridpack::powerflow::blockedHVDCSolution();
    }
    applyDcInjections(m, w, dm, dw, bm);
  }
};

/// Sequential ac/dc step (PFFactoryModule::updateHVDC())
struct DcUpdate {
  ModelView m;
  BatchView w;
  DcModelView dm;
  DcBatchView dw;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int bm = static_cast<int>(i);
    if (!dw.m_check[bm]) return;
    double change = 0.0;
    for (int l = 0; l < dm.n_line; l++) {
      const HVDCLineData &line = dm.line[l];
      const int64_t li = static_cast<int64_t>(l) * dw.B + bm;
      HVDCSolution s = gridpack::powerflow::blockedHVDCSolution();
      if (dw.status[li] != 0 && !dcIsolated(w, line, bm)) {
        s = gridpack::powerflow::solveTwoTerminalDC(
            line, w.v[static_cast<int64_t>(line.rect.bus) * w.B + bm],
            w.v[static_cast<int64_t>(line.inv.bus) * w.B + bm]);
      }
      const HVDCSolution &o = dw.sol[li];
      const double dl = maxOf(maxOf(fabs(s.rect.p - o.rect.p), fabs(s.rect.q - o.rect.q)),
                              maxOf(fabs(s.inv.p - o.inv.p), fabs(s.inv.q - o.inv.q)));
      change = maxOf(change, dl);
      dw.sol[li] = s;
    }
    applyDcInjections(m, w, dm, dw, bm);
    dw.m_change[bm] = absBits(change / m.sbase);
  }
};

/// Final operating points of finished members, case-major
struct GatherDc {
  DcModelView dm;
  DcBatchView dw;
  const int *slots = nullptr;
  HVDCSolution *out = nullptr;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int64_t li = i / dm.n_line;
    const int64_t l = i % dm.n_line;
    out[i] = dw.sol[l * dw.B + slots[li]];
  }
};

}  // namespace batchpf
}  // namespace gridpack

#endif
