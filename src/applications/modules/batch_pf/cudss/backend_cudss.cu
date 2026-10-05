/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   backend_cudss.cu
 * @date   2026-10-05
 *
 * @brief cuDSS uniform-batch backend plugin (I-11, guide 5.3.5, ADR-06).
 *
 * cuDSS factors a "uniform batch" of matrices that share one sparsity
 * pattern. It is NVIDIA's supported sparse direct solver on Arm servers,
 * Jetson and DGX Spark, and replaces the deprecated cuSolverRF (T-4).
 *
 * The plugin is its own shared object so that ca.x and the core plugin
 * work where cuDSS is not installed. Analysis (reordering and symbolic
 * factorization) runs once at setup on the reference matrix; every Newton
 * step then refactorizes and solves. cuDSS wants each member's values and
 * vectors stored one after another, while the engine stores the members of
 * each entry side by side, so small transpose kernels convert both ways.
 * Members switched off in the mask are skipped through cuDSS's uniform
 * batch mask; their value slots hold the reference matrix so that they stay
 * well defined.
 *
 * Per-member health: where cuDSS provides the factor diagonal, a pivot
 * that is not finite or tiny relative to the largest one marks that
 * member; cuDSS 0.8 does not provide it for uniform batches, so there the
 * finiteness check of each solution and the engine's linear residual check
 * (GPUBatch/health/residualLimit) do this job. cuDSS 0.8.0 had a reported defect above
 * about 160 members per uniform batch (risk R-2), so the validated batch
 * size reported to the core plugin is 128.
 *
 * Every CUDA and cuDSS call is checked; errors are returned as status
 * codes with a message, never thrown across the C boundary.
 */

#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gridpack/batchpf/batchpf_backend.h"

namespace {

const int kValidatedBatch = 128;

/// Failure inside the plugin, carrying a status code
class Failure : public std::runtime_error {
 public:
  Failure(batchpf_status code, const std::string &what)
      : std::runtime_error(what), p_code(code) {}
  batchpf_status code() const noexcept { return p_code; }

 private:
  batchpf_status p_code;
};

void cudaCall(cudaError_t e, const char *what)
{
  if (e != cudaSuccess) {
    throw Failure(BATCHPF_ERR_CUDA, std::string(what) + ": " + cudaGetErrorString(e));
  }
}

void cudssCall(cudssStatus_t s, const char *what)
{
  if (s != CUDSS_STATUS_SUCCESS) {
    throw Failure(BATCHPF_ERR_BACKEND, std::string(what) + " failed (cuDSS status " +
                                           std::to_string(static_cast<int>(s)) + ")");
  }
}

/// Owner of a device allocation
struct DeviceArray {
  void *ptr = nullptr;
  DeviceArray() = default;
  explicit DeviceArray(std::size_t bytes) { cudaCall(cudaMalloc(&ptr, bytes ? bytes : 1), "cudaMalloc"); }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  DeviceArray(DeviceArray &&o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
  DeviceArray &operator=(DeviceArray &&o) noexcept
  {
    std::swap(ptr, o.ptr);
    return *this;
  }
  ~DeviceArray() { if (ptr) cudaFree(ptr); }
  template <class T> T *as() const { return static_cast<T *>(ptr); }
};

// ---- layout conversion kernels ------------------------------------------

/// interleaved [p * B + b] -> member-major [b * len + p]; inactive members
/// get fill[p] (or 0 if fill is null)
__global__ void toMemberMajor(const double *in, double *out, const int *mask,
                              const double *fill, int64_t len, int B)
{
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= len * B) return;
  const int64_t p = i / B;
  const int b = static_cast<int>(i % B);
  const double v = mask[b] ? in[i] : (fill ? fill[p] : 0.0);
  out[static_cast<int64_t>(b) * len + p] = v;
}

/// member-major -> interleaved, active members only; flags non-finite
__global__ void toInterleaved(const double *in, double *out, const int *mask,
                              int *status, int64_t len, int B)
{
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= len * B) return;
  const int64_t p = i / B;
  const int b = static_cast<int>(i % B);
  if (!mask[b]) return;
  const double v = in[static_cast<int64_t>(b) * len + p];
  out[i] = v;
  if (!isfinite(v)) atomicMax(status + b, BATCHPF_MEMBER_NONFINITE);
}

/// Per-member pivot check on the factor diagonal (member-major)
__global__ void checkDiagonal(const double *diag, const int *mask, int *status,
                              int n, int B, double limit)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B || !mask[b]) return;
  double mx = 0.0, mn = INFINITY;
  bool finite = true;
  for (int i = 0; i < n; i++) {
    const double a = fabs(diag[static_cast<int64_t>(b) * n + i]);
    finite = finite && isfinite(a);
    mx = fmax(mx, a);
    mn = fmin(mn, a);
  }
  if (!finite) {
    status[b] = BATCHPF_MEMBER_NONFINITE;
  } else if (mn <= limit * mx) {
    status[b] = BATCHPF_MEMBER_SMALL_PIVOT;
  }
}

unsigned blocksFor(int64_t n, int t) { return static_cast<unsigned>((n + t - 1) / t); }

}  // namespace

/// Backend object behind the opaque C handle
struct batchpf_backend {
  int n = 0;
  int B = 0;
  int64_t nnz = 0;
  cudaStream_t stream = nullptr;
  double pivot_limit = 0.0;
  bool factored = false;
  bool diag_available = true;
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t data = nullptr;
  cudssMatrix_t A = nullptr, X = nullptr, Bv = nullptr;
  DeviceArray row_ptr, col_idx, values, ref, rhs, sol, diag, mask_dev;
  std::vector<int> mask_host, mask_prev;
  std::string error;

  ~batchpf_backend()
  {
    if (A) cudssMatrixDestroy(A);
    if (X) cudssMatrixDestroy(X);
    if (Bv) cudssMatrixDestroy(Bv);
    if (data) cudssDataDestroy(handle, data);
    if (config) cudssConfigDestroy(config);
    if (handle) cudssDestroy(handle);
  }

  void setup(const batchpf_backend_plan &p)
  {
    n = p.n;
    B = p.batch_capacity;
    nnz = p.nnz;
    stream = static_cast<cudaStream_t>(p.stream);
    pivot_limit = p.pivot_limit;
    cudaCall(cudaSetDevice(p.device), "cudaSetDevice");
    row_ptr = DeviceArray((n + 1) * sizeof(int));
    col_idx = DeviceArray(nnz * sizeof(int));
    values = DeviceArray(static_cast<std::size_t>(nnz) * B * sizeof(double));
    ref = DeviceArray(nnz * sizeof(double));
    rhs = DeviceArray(static_cast<std::size_t>(n) * B * sizeof(double));
    sol = DeviceArray(static_cast<std::size_t>(n) * B * sizeof(double));
    diag = DeviceArray(static_cast<std::size_t>(n) * B * sizeof(double));
    mask_dev = DeviceArray(B * sizeof(int));
    mask_host.assign(B, 1);
    cudaCall(cudaMemcpyAsync(row_ptr.ptr, p.row_ptr, (n + 1) * sizeof(int),
                             cudaMemcpyHostToDevice, stream), "copy pattern");
    cudaCall(cudaMemcpyAsync(col_idx.ptr, p.col_idx, nnz * sizeof(int),
                             cudaMemcpyHostToDevice, stream), "copy pattern");
    cudaCall(cudaMemcpyAsync(ref.ptr, p.reference_values, nnz * sizeof(double),
                             cudaMemcpyHostToDevice, stream), "copy reference");
    cudaCall(cudaMemcpyAsync(mask_dev.ptr, mask_host.data(), B * sizeof(int),
                             cudaMemcpyHostToDevice, stream), "copy mask");
    // Every slot starts with the reference matrix (mask all zero = fill)
    const int t = 256;
    std::vector<int> zeros(B, 0);
    cudaCall(cudaMemcpyAsync(mask_dev.ptr, zeros.data(), B * sizeof(int),
                             cudaMemcpyHostToDevice, stream), "copy mask");
    toMemberMajor<<<blocksFor(nnz * B, t), t, 0, stream>>>(
        nullptr, values.as<double>(), mask_dev.as<int>(), ref.as<double>(), nnz, B);
    cudaCall(cudaGetLastError(), "fill reference");
    cudaCall(cudaMemsetAsync(rhs.ptr, 0, static_cast<std::size_t>(n) * B * sizeof(double),
                             stream), "memset");
    cudaCall(cudaStreamSynchronize(stream), "setup");

    cudssCall(cudssCreate(&handle), "cudssCreate");
    cudssCall(cudssSetStream(handle, stream), "cudssSetStream");
    cudssCall(cudssConfigCreate(&config), "cudssConfigCreate");
    cudssCall(cudssDataCreate(handle, &data), "cudssDataCreate");
    int ub = B;
    cudssCall(cudssConfigSet(config, CUDSS_CONFIG_UBATCH_SIZE, &ub, sizeof(ub)),
              "set uniform batch size");
    if (p.refinement_steps > 0) {
      int steps = p.refinement_steps;
      cudssCall(cudssConfigSet(config, CUDSS_CONFIG_IR_N_STEPS, &steps, sizeof(steps)),
                "set refinement steps");
    }
    cudssCall(cudssMatrixCreateCsr(&A, n, n, nnz, row_ptr.ptr, nullptr, col_idx.ptr,
                                   values.ptr, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                                   CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
              "create matrix");
    cudssCall(cudssMatrixCreateDn(&Bv, n, 1, n, rhs.ptr, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
              "create rhs");
    cudssCall(cudssMatrixCreateDn(&X, n, 1, n, sol.ptr, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
              "create solution");
    // One-time analysis on the reference matrix in every slot
    cudssCall(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, A, X, Bv),
              "analysis");
    cudaCall(cudaStreamSynchronize(stream), "analysis");
  }

  void setMask(const int *mask)
  {
    cudaCall(cudaMemcpyAsync(mask_host.data(), mask, B * sizeof(int),
                             cudaMemcpyDeviceToHost, stream), "read mask");
    cudaCall(cudaStreamSynchronize(stream), "read mask");
    if (mask_host != mask_prev) {
      cudssCall(cudssDataSet(handle, data, CUDSS_DATA_UBATCH_MASK, mask_host.data(),
                             B * sizeof(int)), "set batch mask");
      mask_prev = mask_host;
    }
  }

  void refactorize(const double *vals, const int *mask, int *status)
  {
    const int t = 256;
    toMemberMajor<<<blocksFor(nnz * B, t), t, 0, stream>>>(
        vals, values.as<double>(), mask, ref.as<double>(), nnz, B);
    cudaCall(cudaGetLastError(), "toMemberMajor");
    setMask(mask);
    cudssCall(cudssExecute(handle, factored ? CUDSS_PHASE_REFACTORIZATION
                                            : CUDSS_PHASE_FACTORIZATION,
                           config, data, A, X, Bv),
              "factorization");
    factored = true;
    // Factor diagonal for the pivot check, where cuDSS provides it. cuDSS
    // 0.8 does not for uniform batches; then tiny pivots are caught by the
    // finiteness check of the solution and the engine's residual check.
    if (diag_available) {
      std::size_t written = 0;
      const cudssStatus_t s =
          cudssDataGet(handle, data, CUDSS_DATA_DIAG, diag.ptr,
                       static_cast<std::size_t>(n) * B * sizeof(double), &written);
      if (s != CUDSS_STATUS_SUCCESS ||
          written < static_cast<std::size_t>(n) * B * sizeof(double)) {
        diag_available = false;
      } else {
        checkDiagonal<<<blocksFor(B, 128), 128, 0, stream>>>(
            diag.as<double>(), mask, status, n, B, pivot_limit);
        cudaCall(cudaGetLastError(), "checkDiagonal");
      }
    }
  }

  void solve(const double *in, double *out, const int *mask, int *status)
  {
    const int t = 256;
    toMemberMajor<<<blocksFor(static_cast<int64_t>(n) * B, t), t, 0, stream>>>(
        in, rhs.as<double>(), mask, nullptr, n, B);
    cudaCall(cudaGetLastError(), "toMemberMajor");
    cudssCall(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A, X, Bv), "solve");
    toInterleaved<<<blocksFor(static_cast<int64_t>(n) * B, t), t, 0, stream>>>(
        sol.as<double>(), out, mask, status, n, B);
    cudaCall(cudaGetLastError(), "toInterleaved");
  }
};

namespace {

template <class F>
batchpf_status guarded(batchpf_backend *b, char *err, size_t err_size, F &&f)
{
  try {
    f();
    return BATCHPF_OK;
  } catch (const Failure &e) {
    if (b) b->error = e.what();
    if (err && err_size) {
      std::strncpy(err, e.what(), err_size - 1);
      err[err_size - 1] = '\0';
    }
    return e.code();
  } catch (const std::exception &e) {
    if (b) b->error = e.what();
    if (err && err_size) {
      std::strncpy(err, e.what(), err_size - 1);
      err[err_size - 1] = '\0';
    }
    return BATCHPF_ERR_INTERNAL;
  }
}

batchpf_status capabilities(batchpf_backend_caps *caps)
{
  if (!caps || caps->struct_size < sizeof(batchpf_backend_caps)) {
    return BATCHPF_ERR_INVALID_ARGUMENT;
  }
  const uint32_t size = caps->struct_size;
  std::memset(caps, 0, sizeof(*caps));
  caps->struct_size = size;
  caps->struct_version = 1;
  caps->max_batch = 0;
  caps->validated_batch = kValidatedBatch;
  caps->member_masking = 1;
  caps->iterative_refinement = 1;
  caps->failed_member_reporting = 1;
  caps->memory_kinds = BATCHPF_MEMKIND_DEVICE;
  std::strncpy(caps->name, "cudss", sizeof(caps->name) - 1);
  int major = 0, minor = 0, patch = 0;
  cudssGetProperty(MAJOR_VERSION, &major);
  cudssGetProperty(MINOR_VERSION, &minor);
  cudssGetProperty(PATCH_LEVEL, &patch);
  const std::string v = "cuDSS " + std::to_string(major) + "." + std::to_string(minor) +
                        "." + std::to_string(patch) + " uniform batch";
  std::strncpy(caps->version, v.c_str(), sizeof(caps->version) - 1);
  return BATCHPF_OK;
}

batchpf_status setup(const batchpf_backend_plan *plan, batchpf_backend **out,
                     char *err, size_t err_size)
{
  if (!plan || !out || plan->struct_size < sizeof(batchpf_backend_plan) ||
      plan->memory_kind != BATCHPF_MEMKIND_DEVICE || plan->n <= 0 ||
      plan->batch_capacity <= 0) {
    return BATCHPF_ERR_INVALID_ARGUMENT;
  }
  auto b = std::make_unique<batchpf_backend>();
  const batchpf_status st = guarded(b.get(), err, err_size, [&] { b->setup(*plan); });
  if (st == BATCHPF_OK) *out = b.release();
  return st;
}

batchpf_status refactorize(batchpf_backend *b, const double *values, const int32_t *mask,
                           int32_t *status)
{
  if (!b || !values || !mask || !status) return BATCHPF_ERR_INVALID_ARGUMENT;
  return guarded(b, nullptr, 0, [&] { b->refactorize(values, mask, status); });
}

batchpf_status solve(batchpf_backend *b, const double *rhs, double *x, const int32_t *mask,
                     int32_t *status)
{
  if (!b || !rhs || !x || !mask || !status) return BATCHPF_ERR_INVALID_ARGUMENT;
  return guarded(b, nullptr, 0, [&] { b->solve(rhs, x, mask, status); });
}

void teardown(batchpf_backend *b)
{
  std::unique_ptr<batchpf_backend> owned(b);   // ownership returns here
}

const char *lastError(const batchpf_backend *b) { return b ? b->error.c_str() : ""; }

}  // namespace

extern "C" __attribute__((visibility("default")))
batchpf_status batchpf_get_backend_api(uint32_t requested_major, batchpf_backend_api *api)
{
  if (!api || api->struct_size < offsetof(batchpf_backend_api, name)) {
    return BATCHPF_ERR_INVALID_ARGUMENT;
  }
  batchpf_backend_api full;
  std::memset(&full, 0, sizeof(full));
  full.struct_size = sizeof(full);
  full.api_major = BATCHPF_BACKEND_API_MAJOR;
  full.api_minor = BATCHPF_BACKEND_API_MINOR;
  full.name = "cudss";
  full.capabilities = capabilities;
  full.setup = setup;
  full.refactorize = refactorize;
  full.solve = solve;
  full.teardown = teardown;
  full.last_error = lastError;
  const uint32_t size = api->struct_size;
  std::memcpy(api, &full, std::min<std::size_t>(size, sizeof(full)));
  api->struct_size = std::min<uint32_t>(size, sizeof(full));
  if (requested_major != BATCHPF_BACKEND_API_MAJOR) return BATCHPF_ERR_VERSION;
  return BATCHPF_OK;
}
