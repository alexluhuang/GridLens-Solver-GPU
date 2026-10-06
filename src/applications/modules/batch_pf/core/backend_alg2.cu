/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   backend_alg2.cu
 * @date   2026-10-05
 *
 * @brief Custom batched LU backend: Algorithm 2 of Zhou et al. (guide
 * appendix B.3, ADR-06).
 *
 * All members share the sparsity pattern and the fixed row/column order
 * from the planner, so one plan serves the whole batch:
 *
 *  - Refactorization is left-looking, column by column. Columns are
 *    grouped in dependency levels; a level is one kernel launch. Inside a
 *    launch a thread block handles one column, and thread t of the block
 *    handles member t. Because the values of one entry for all members sit
 *    next to each other (case-interleaved layout), the threads of a warp
 *    read consecutive addresses, and because every member has the same
 *    structure they never take different branches.
 *  - The triangular solves use the same arrangement, by rows, with one
 *    launch per level (ADR-04: solve on the GPU by default).
 *
 * No pivoting happens during refactorization. A pivot that is tiny compared
 * with its column in the reference matrix, or not finite, marks the member
 * so that the engine sends it to GridPACK's CPU path (guide 8.5).
 *
 * With GPUBatch/solvePlacement = host the factors are kept in pinned host
 * memory and the solves run on the CPU (an experiment for PERF-2).
 */

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "backend.hpp"
#include "executor.cuh"

namespace gridpack {
namespace batchpf {

namespace {

/// Plan arrays on the device
struct Alg2Plan {
  int n = 0;
  int B = 0;
  int64_t nnz_a = 0;
  const int *a_to_lu = nullptr;
  const int *col_ptr = nullptr;
  const int *diag = nullptr;
  const int *seg_ptr = nullptr;
  const int *seg_upos = nullptr;
  const int *seg_lbeg = nullptr;
  const int *seg_len = nullptr;
  const int *seg_dst = nullptr;
  const int *dst = nullptr;
  const int *lrow_ptr = nullptr;
  const int *lrow_col = nullptr;
  const int *lrow_pos = nullptr;
  const int *urow_ptr = nullptr;
  const int *urow_col = nullptr;
  const int *urow_pos = nullptr;
  const int *P = nullptr;
  const int *Q = nullptr;
  const double *col_scale = nullptr;
  double *lu = nullptr;      // nnz_lu x B
  double *y = nullptr;       // n x B work vector
  double pivot_limit = 0.0;
};

/// Clear the factor storage of active members
struct ZeroFactor {
  Alg2Plan p;
  const int *mask;
  __host__ __device__ void operator()(int64_t i) const
  {
    if (mask[i % p.B]) p.lu[i] = 0.0;
  }
};

/// Place the Jacobian values into the factor storage
struct ScatterValues {
  Alg2Plan p;
  const double *values;
  const int *mask;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int b = static_cast<int>(i % p.B);
    if (!mask[b]) return;
    const int64_t a = i / p.B;
    p.lu[static_cast<int64_t>(p.a_to_lu[a]) * p.B + b] = values[i];
  }
};

/// One column of the left-looking factorization for one member
__host__ __device__ inline void factorColumn(const Alg2Plan &p, int j, int b)
{
  const int64_t B = p.B;
  double *lu = p.lu;
  for (int s = p.seg_ptr[j]; s < p.seg_ptr[j + 1]; s++) {
    const double ukj = lu[p.seg_upos[s] * B + b];
    const int lb = p.seg_lbeg[s];
    const int len = p.seg_len[s];
    const int *dst = p.dst + p.seg_dst[s];
    for (int t = 0; t < len; t++) {
      lu[dst[t] * B + b] -= lu[(lb + t) * B + b] * ukj;
    }
  }
  const double piv = lu[p.diag[j] * B + b];
  for (int q = p.diag[j] + 1; q < p.col_ptr[j + 1]; q++) {
    lu[q * B + b] /= piv;
  }
}

/// Kernel for one level: block = (column, chunk of members)
__global__ void factorLevelKernel(Alg2Plan p, const int *cols, int ncols,
                                  int chunks, const int *mask)
{
  const int c = blockIdx.x / chunks;
  const int b = (blockIdx.x % chunks) * blockDim.x + threadIdx.x;
  if (c >= ncols || b >= p.B || !mask[b]) return;
  factorColumn(p, cols[c], b);
}

/**
 * Kernel for one level with L threads per member, for levels with few
 * columns (the last levels of the elimination, where columns are long and
 * one thread per member would work through them alone). A block of L warps
 * takes one column for 32 consecutive members: warp w applies every L-th
 * entry of each update segment, and the block synchronizes between
 * segments. Every factor entry receives the same operations in the same
 * order as in factorColumn(), so the factors are bitwise identical.
 */
template <int L>
__global__ void __launch_bounds__(32 * L)
factorLevelWideKernel(Alg2Plan p, const int *cols, int groups, const int *mask)
{
  const int c = blockIdx.x / groups;
  const int b = (blockIdx.x % groups) * 32 + static_cast<int>(threadIdx.x % 32);
  const int lane = static_cast<int>(threadIdx.x / 32);
  const bool active = b < p.B && mask[b] != 0;
  if (!__syncthreads_or(active)) return;   // same answer for the whole block
  const int j = cols[c];
  const int64_t B = p.B;
  double *lu = p.lu;
  for (int s = p.seg_ptr[j]; s < p.seg_ptr[j + 1]; s++) {
    if (active) {
      const double ukj = lu[p.seg_upos[s] * B + b];
      const int lb = p.seg_lbeg[s];
      const int len = p.seg_len[s];
      const int *dst = p.dst + p.seg_dst[s];
      // Targets of one segment are distinct, so their updates are
      // independent: read four before writing any
      int t = lane;
      for (; t + 3 * L < len; t += 4 * L) {
        const int64_t d0 = dst[t] * B + b, d1 = dst[t + L] * B + b;
        const int64_t d2 = dst[t + 2 * L] * B + b, d3 = dst[t + 3 * L] * B + b;
        const double l0 = lu[(lb + t) * B + b], l1 = lu[(lb + t + L) * B + b];
        const double l2 = lu[(lb + t + 2 * L) * B + b], l3 = lu[(lb + t + 3 * L) * B + b];
        const double x0 = lu[d0], x1 = lu[d1], x2 = lu[d2], x3 = lu[d3];
        lu[d0] = x0 - l0 * ukj;
        lu[d1] = x1 - l1 * ukj;
        lu[d2] = x2 - l2 * ukj;
        lu[d3] = x3 - l3 * ukj;
      }
      for (; t < len; t += L) {
        lu[dst[t] * B + b] -= lu[(lb + t) * B + b] * ukj;
      }
    }
    __syncthreads();
  }
  if (!active) return;
  const double piv = lu[p.diag[j] * B + b];
  for (int q = p.diag[j] + 1 + lane; q < p.col_ptr[j + 1]; q += L) {
    lu[q * B + b] /= piv;
  }
}

// Many rows can flag the same member. Preserve the strongest failure
// without racing between threads (non-finite takes precedence).
__host__ __device__ void recordFailure(int *status, int cause)
{
#ifdef __CUDA_ARCH__
  atomicMax(status, cause);
#else
  *status = std::max(*status, cause);
#endif
}

/// Pivot health: tiny relative to the reference column, or not finite
struct PivotCheck {
  Alg2Plan p;
  const int *mask;
  int *status;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int b = static_cast<int>(i % p.B);
    if (!mask[b]) return;
    const int64_t j = i / p.B;
    const double piv = p.lu[static_cast<int64_t>(p.diag[j]) * p.B + b];
    if (!isfinite(piv)) {
      recordFailure(status + b, BATCHPF_MEMBER_NONFINITE);
    } else if (fabs(piv) <= p.pivot_limit * p.col_scale[j]) {
      recordFailure(status + b, BATCHPF_MEMBER_SMALL_PIVOT);
    }
  }
};

/// y = rhs(P)
struct PermuteIn {
  Alg2Plan p;
  const double *rhs;
  const int *mask;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int b = static_cast<int>(i % p.B);
    if (!mask[b]) return;
    const int64_t r = i / p.B;
    p.y[i] = rhs[static_cast<int64_t>(p.P[r]) * p.B + b];
  }
};

/// x(Q) = y, with a finiteness check
struct PermuteOut {
  Alg2Plan p;
  double *x;
  const int *mask;
  int *status;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int b = static_cast<int>(i % p.B);
    if (!mask[b]) return;
    const int64_t r = i / p.B;
    const double v = p.y[i];
    x[static_cast<int64_t>(p.Q[r]) * p.B + b] = v;
    if (!isfinite(v)) recordFailure(status + b, BATCHPF_MEMBER_NONFINITE);
  }
};

/// Forward substitution, one row of a level: y_i -= sum L(i,k) y_k
struct ForwardRows {
  Alg2Plan p;
  const int *rows;
  const int *mask;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int b = static_cast<int>(i % p.B);
    if (!mask[b]) return;
    const int r = rows[i / p.B];
    const int64_t B = p.B;
    double s = p.y[r * B + b];
    // Same order of subtractions as a plain loop; unrolled so that the
    // loads of several terms are in flight at once
#ifdef __CUDA_ARCH__
#pragma unroll 4
#endif
    for (int q = p.lrow_ptr[r]; q < p.lrow_ptr[r + 1]; q++) {
      s -= p.lu[p.lrow_pos[q] * B + b] * p.y[p.lrow_col[q] * B + b];
    }
    p.y[r * B + b] = s;
  }
};

/// Back substitution, one row of a level: y_i = (y_i - sum U(i,j) y_j) / U(i,i)
struct BackwardRows {
  Alg2Plan p;
  const int *rows;
  const int *mask;
  __host__ __device__ void operator()(int64_t i) const
  {
    const int b = static_cast<int>(i % p.B);
    if (!mask[b]) return;
    const int r = rows[i / p.B];
    const int64_t B = p.B;
    double s = p.y[r * B + b];
    // Same order of subtractions as a plain loop; unrolled so that the
    // loads of several terms are in flight at once
#ifdef __CUDA_ARCH__
#pragma unroll 4
#endif
    for (int q = p.urow_ptr[r]; q < p.urow_ptr[r + 1]; q++) {
      s -= p.lu[p.urow_pos[q] * B + b] * p.y[p.urow_col[q] * B + b];
    }
    p.y[r * B + b] = s / p.lu[p.diag[r] * B + b];
  }
};

class Alg2Backend : public SolverBackend {
 public:
  explicit Alg2Backend(const BackendSetup &setup)
      : p_setup(setup), p_lu(*setup.lu),
        p_factor_graph(setup.launch_graphs), p_solve_graph(setup.launch_graphs)
  {
    const cudaStream_t st = setup.stream;
    const MemoryKind mk = MemoryKind::Device;
    auto up = [&](Buffer<int> &dst, const std::vector<int> &src) {
      dst.allocate(mk, src.size(), st);
      dst.upload(src.data(), src.size(), st);
    };
    up(p_a_to_lu, p_lu.a_to_lu);
    up(p_col_ptr, p_lu.col_ptr);
    up(p_diag, p_lu.diag);
    up(p_seg_ptr, p_lu.seg_ptr);
    up(p_seg_upos, p_lu.seg_upos);
    up(p_seg_lbeg, p_lu.seg_lbeg);
    up(p_seg_len, p_lu.seg_len);
    up(p_seg_dst, p_lu.seg_dst);
    up(p_dst, p_lu.dst);
    up(p_lev_cols, p_lu.lev_cols);
    up(p_lrow_ptr, p_lu.lrow_ptr);
    up(p_lrow_col, p_lu.lrow_col);
    up(p_lrow_pos, p_lu.lrow_pos);
    up(p_urow_ptr, p_lu.urow_ptr);
    up(p_urow_col, p_lu.urow_col);
    up(p_urow_pos, p_lu.urow_pos);
    up(p_llev_rows, p_lu.llev_rows);
    up(p_ulev_rows, p_lu.ulev_rows);
    up(p_P, p_lu.P);
    up(p_Q, p_lu.Q);
    p_col_scale.allocate(mk, p_lu.col_scale.size(), st);
    p_col_scale.upload(p_lu.col_scale.data(), p_lu.col_scale.size(), st);
    const std::size_t B = static_cast<std::size_t>(setup.capacity);
    // Factors in pinned host memory only when the solve runs on the CPU
    p_values.allocate(setup.host_solve ? MemoryKind::Pinned : mk,
                      static_cast<std::size_t>(p_lu.nnz) * B, st);
    p_y.allocate(setup.host_solve ? MemoryKind::Pinned : mk,
                 static_cast<std::size_t>(p_lu.n) * B, st);
    p_values.zero(st);
    // Block size: one thread per member, a multiple of 32 (CUDA Best
    // Practices Guide 11.3), at most 256 so a column's members fit in a
    // few warps
    const int want = setup.threads_per_block > 0 ? setup.threads_per_block : 256;
    p_threads = std::min<int>(want, static_cast<int>(((B + 31) / 32) * 32));
    p_threads = std::max(p_threads, 32);
    p_chunks = static_cast<int>((B + p_threads - 1) / p_threads);
    chooseLanes(setup);
    cudaCheck(cudaStreamSynchronize(st), "alg2 setup");
  }

  BackendCaps caps() const override
  {
    BackendCaps c;
    c.name = "alg2";
    c.version = "batched left-looking LU, levels " +
                std::to_string(p_lu.lev_ptr.size() - 1);
    c.on_device = true;
    c.validated_batch = kAlg2ValidatedBatch;
    return c;
  }

  void refactorize(gsl::span<const double> values, gsl::span<const int> mask,
                   gsl::span<int> member_status) override
  {
    Expects(values.size() == static_cast<std::size_t>(p_setup.pattern->nnz) * p_setup.capacity);
    Expects(mask.size() == static_cast<std::size_t>(p_setup.capacity));
    Expects(member_status.size() == mask.size());
    p_factor_graph.run(p_setup.stream,
                       {values.data(), mask.data(), member_status.data(), nullptr},
                       [&] { launchFactor(values.data(), mask.data(), member_status.data()); });
  }

  void solve(gsl::span<const double> rhs, gsl::span<double> x,
             gsl::span<const int> mask, gsl::span<int> member_status) override
  {
    Expects(rhs.size() == static_cast<std::size_t>(p_lu.n) * p_setup.capacity);
    Expects(x.size() == rhs.size());
    Expects(mask.size() == static_cast<std::size_t>(p_setup.capacity));
    Expects(member_status.size() == mask.size());
    if (p_setup.host_solve) {
      solveOnHost(rhs.data(), x.data(), mask.data(), member_status.data());
      return;
    }
    p_solve_graph.run(p_setup.stream,
                      {rhs.data(), x.data(), mask.data(), member_status.data()},
                      [&] { launchSolve(rhs.data(), x.data(), mask.data(), member_status.data()); });
  }

 private:
  /// The launches of one factorization (ZeroFactor to PivotCheck)
  void launchFactor(const double *values, const int *mask, int *member_status)
  {
    const Alg2Plan p = view();
    const Executor ex(true, p_setup.stream, p_setup.threads_per_block);
    const int64_t B = p_setup.capacity;
    ex.run(p_lu.nnz * B, ZeroFactor{p, mask}, "ZeroFactor");
    ex.run(p_setup.pattern->nnz * B, ScatterValues{p, values, mask}, "ScatterValues");
    const int groups = static_cast<int>((B + 31) / 32);
    for (std::size_t l = 0; l + 1 < p_lu.lev_ptr.size(); l++) {
      const int beg = p_lu.lev_ptr[l];
      const int ncols = p_lu.lev_ptr[l + 1] - beg;
      const int *cols = p_lev_cols.data() + beg;
      const int lanes = p_lanes[l];
      if (lanes == 1) {
        const unsigned grid = static_cast<unsigned>(ncols) * p_chunks;
        factorLevelKernel<<<grid, p_threads, 0, p_setup.stream>>>(
            p, cols, ncols, p_chunks, mask);
      } else {
        const unsigned grid = static_cast<unsigned>(ncols) * groups;
        const unsigned threads = 32U * static_cast<unsigned>(lanes);
        const cudaStream_t st = p_setup.stream;
        switch (lanes) {
          case 2: factorLevelWideKernel<2><<<grid, threads, 0, st>>>(p, cols, groups, mask); break;
          case 4: factorLevelWideKernel<4><<<grid, threads, 0, st>>>(p, cols, groups, mask); break;
          case 8: factorLevelWideKernel<8><<<grid, threads, 0, st>>>(p, cols, groups, mask); break;
          case 16: factorLevelWideKernel<16><<<grid, threads, 0, st>>>(p, cols, groups, mask); break;
          default: factorLevelWideKernel<32><<<grid, threads, 0, st>>>(p, cols, groups, mask); break;
        }
      }
      launchCheck("factorLevelKernel");
    }
    ex.run(static_cast<int64_t>(p_lu.n) * B,
           PivotCheck{p, mask, member_status}, "PivotCheck");
  }

  /// The launches of one solve (PermuteIn to PermuteOut)
  void launchSolve(const double *rhs, double *x, const int *mask, int *member_status)
  {
    const Alg2Plan p = view();
    const int64_t B = p_setup.capacity;
    const int64_t nB = static_cast<int64_t>(p_lu.n) * B;
    const Executor ex(true, p_setup.stream, p_setup.threads_per_block);
    ex.run(nB, PermuteIn{p, rhs, mask}, "PermuteIn");
    for (std::size_t l = 0; l + 1 < p_lu.llev_ptr.size(); l++) {
      const int beg = p_lu.llev_ptr[l];
      const int cnt = p_lu.llev_ptr[l + 1] - beg;
      ex.run(static_cast<int64_t>(cnt) * B,
             ForwardRows{p, p_llev_rows.data() + beg, mask}, "ForwardRows");
    }
    for (std::size_t l = 0; l + 1 < p_lu.ulev_ptr.size(); l++) {
      const int beg = p_lu.ulev_ptr[l];
      const int cnt = p_lu.ulev_ptr[l + 1] - beg;
      ex.run(static_cast<int64_t>(cnt) * B,
             BackwardRows{p, p_ulev_rows.data() + beg, mask}, "BackwardRows");
    }
    ex.run(nB, PermuteOut{p, x, mask, member_status}, "PermuteOut");
  }

  /**
   * Solve on the CPU (GPUBatch/solvePlacement = host). The factors were
   * written by the GPU into pinned memory; right-hand side, mask and
   * status are staged through pinned buffers because device allocations
   * are not readable by the CPU on every platform (DGX Spark included).
   */
  void solveOnHost(const double *rhs, double *x, const int *mask,
                   int *member_status)
  {
    const cudaStream_t st = p_setup.stream;
    const int64_t B = p_setup.capacity;
    const std::size_t nB = static_cast<std::size_t>(p_lu.n) * B;
    p_hrhs.allocate(MemoryKind::Pinned, nB);
    p_hx.allocate(MemoryKind::Pinned, nB);
    p_hmask.assign(B, 0);
    p_hstatus.assign(B, 0);
    cudaCheck(cudaMemcpyAsync(p_hrhs.data(), rhs, nB * sizeof(double),
                              cudaMemcpyDeviceToHost, st), "stage rhs");
    cudaCheck(cudaMemcpyAsync(p_hmask.data(), mask, B * sizeof(int),
                              cudaMemcpyDeviceToHost, st), "stage mask");
    cudaCheck(cudaMemcpyAsync(p_hstatus.data(), member_status, B * sizeof(int),
                              cudaMemcpyDeviceToHost, st), "stage status");
    cudaCheck(cudaStreamSynchronize(st), "host solve");
    Alg2Plan h = view();
    h.P = p_lu.P.data();
    h.Q = p_lu.Q.data();
    h.diag = p_lu.diag.data();
    h.lrow_ptr = p_lu.lrow_ptr.data();
    h.lrow_col = p_lu.lrow_col.data();
    h.lrow_pos = p_lu.lrow_pos.data();
    h.urow_ptr = p_lu.urow_ptr.data();
    h.urow_col = p_lu.urow_col.data();
    h.urow_pos = p_lu.urow_pos.data();
    const Executor ex(false, nullptr, 0);
    const int *hm = p_hmask.data();
    ex.run(static_cast<int64_t>(nB), PermuteIn{h, p_hrhs.data(), hm}, "PermuteIn");
    for (std::size_t l = 0; l + 1 < p_lu.llev_ptr.size(); l++) {
      const int beg = p_lu.llev_ptr[l];
      const int cnt = p_lu.llev_ptr[l + 1] - beg;
      ex.run(static_cast<int64_t>(cnt) * B,
             ForwardRows{h, p_lu.llev_rows.data() + beg, hm}, "ForwardRows");
    }
    for (std::size_t l = 0; l + 1 < p_lu.ulev_ptr.size(); l++) {
      const int beg = p_lu.ulev_ptr[l];
      const int cnt = p_lu.ulev_ptr[l + 1] - beg;
      ex.run(static_cast<int64_t>(cnt) * B,
             BackwardRows{h, p_lu.ulev_rows.data() + beg, hm}, "BackwardRows");
    }
    ex.run(static_cast<int64_t>(nB),
           PermuteOut{h, p_hx.data(), hm, p_hstatus.data()}, "PermuteOut");
    cudaCheck(cudaMemcpyAsync(x, p_hx.data(), nB * sizeof(double),
                              cudaMemcpyHostToDevice, st), "return x");
    cudaCheck(cudaMemcpyAsync(member_status, p_hstatus.data(), B * sizeof(int),
                              cudaMemcpyHostToDevice, st), "return status");
    cudaCheck(cudaStreamSynchronize(st), "host solve");
  }

  /**
   * Threads per member for each factorization level. A level with few
   * columns gives the GPU too little work at one thread per member: it is
   * then spread over more warps, up to about sixteen per streaming
   * multiprocessor, but not past the longest update in the level. (On
   * GB10, 4, 16 and 32 warps per multiprocessor were within 10% of each
   * other at batch sizes 128 to 2048; 16 was best or close to it.)
   * BackendSetup::factor_lanes forces one value for all levels (tests).
   */
  void chooseLanes(const BackendSetup &setup)
  {
    const std::size_t nlev = p_lu.lev_ptr.size() - 1;
    p_lanes.assign(nlev, 1);
    if (setup.factor_lanes > 0) {
      int forced = 1;
      while (forced < setup.factor_lanes && forced < kMaxLanes) forced *= 2;
      p_lanes.assign(nlev, forced);
      return;
    }
    int sms = 1;
    cudaCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, setup.device),
              "multiprocessor count");
    const int64_t groups = (static_cast<int64_t>(setup.capacity) + 31) / 32;
    const int64_t target = static_cast<int64_t>(sms) * kWarpsPerSm;
    for (std::size_t l = 0; l < nlev; l++) {
      int longest = 0;
      for (int c = p_lu.lev_ptr[l]; c < p_lu.lev_ptr[l + 1]; c++) {
        const int j = p_lu.lev_cols[c];
        for (int s = p_lu.seg_ptr[j]; s < p_lu.seg_ptr[j + 1]; s++) {
          longest = std::max(longest, p_lu.seg_len[s]);
        }
      }
      const int64_t warps = (p_lu.lev_ptr[l + 1] - p_lu.lev_ptr[l]) * groups;
      int lanes = 1;
      while (lanes < kMaxLanes && warps * lanes < target && lanes < longest) lanes *= 2;
      p_lanes[l] = lanes;
    }
  }

  Alg2Plan view()
  {
    Alg2Plan p;
    p.n = p_lu.n;
    p.B = p_setup.capacity;
    p.nnz_a = p_setup.pattern->nnz;
    p.a_to_lu = p_a_to_lu.data();
    p.col_ptr = p_col_ptr.data();
    p.diag = p_diag.data();
    p.seg_ptr = p_seg_ptr.data();
    p.seg_upos = p_seg_upos.data();
    p.seg_lbeg = p_seg_lbeg.data();
    p.seg_len = p_seg_len.data();
    p.seg_dst = p_seg_dst.data();
    p.dst = p_dst.data();
    p.lrow_ptr = p_lrow_ptr.data();
    p.lrow_col = p_lrow_col.data();
    p.lrow_pos = p_lrow_pos.data();
    p.urow_ptr = p_urow_ptr.data();
    p.urow_col = p_urow_col.data();
    p.urow_pos = p_urow_pos.data();
    p.P = p_P.data();
    p.Q = p_Q.data();
    p.col_scale = p_col_scale.data();
    p.lu = p_values.data();
    p.y = p_y.data();
    p.pivot_limit = p_setup.pivot_limit;
    return p;
  }

  BackendSetup p_setup;
  const LuPlan &p_lu;
  static constexpr int kMaxLanes = 32;
  static constexpr int kWarpsPerSm = 16;
  int p_threads = 128;
  int p_chunks = 1;
  std::vector<int> p_lanes;             // threads per member, per level
  LaunchGraph p_factor_graph, p_solve_graph;
  Buffer<int> p_a_to_lu, p_col_ptr, p_diag, p_seg_ptr, p_seg_upos, p_seg_lbeg;
  Buffer<int> p_seg_len, p_seg_dst, p_dst, p_lev_cols;
  Buffer<int> p_lrow_ptr, p_lrow_col, p_lrow_pos, p_urow_ptr, p_urow_col, p_urow_pos;
  Buffer<int> p_llev_rows, p_ulev_rows, p_P, p_Q;
  Buffer<double> p_col_scale, p_values, p_y;
  Buffer<double> p_hrhs, p_hx;          // host-solve staging
  std::vector<int> p_hmask, p_hstatus;
};

}  // namespace

std::unique_ptr<SolverBackend> makeAlg2Backend(const BackendSetup &setup)
{
  return std::make_unique<Alg2Backend>(setup);
}

}  // namespace batchpf
}  // namespace gridpack
