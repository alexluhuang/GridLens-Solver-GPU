/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   executor.cuh
 * @date   2026-10-05
 *
 * @brief Runs a per-element function over an index range, either as a CUDA
 * kernel or as a loop on the CPU.
 *
 * The power flow element formulas (mismatch, Jacobian blocks, updates) are
 * written once as functors with __host__ __device__ call operators. The GPU
 * engine launches them as kernels; the CPU reference engine runs the same
 * functors in a loop. This avoids keeping two copies of the formulas and
 * lets the unit tests compare CPU and GPU results of the same code (CUDA
 * Best Practices Guide 7.1.2).
 *
 * The index i encodes (item, member) as i = item * B + member, where B is
 * the batch capacity. Neighboring threads therefore work on neighboring
 * members of the same item and read neighboring addresses in the
 * case-interleaved layout (guide section 8.4.1).
 */

#ifndef GRIDPACK_BATCHPF_CORE_EXECUTOR_CUH
#define GRIDPACK_BATCHPF_CORE_EXECUTOR_CUH

#include <cuda_runtime.h>

#include <cstdint>

#include "common.hpp"

namespace gridpack {
namespace batchpf {

template <class F>
__global__ void forEachKernel(int64_t n, F f)
{
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) f(i);
}

/**
 * Chooses where functors run. On the GPU the launch configuration is the
 * configured block size (a multiple of 32, CUDA Best Practices Guide 11.3)
 * or, when automatic, the size the occupancy calculator suggests for each
 * kernel (cached per functor type).
 */
class Executor {
 public:
  Executor() = default;
  Executor(bool on_device, cudaStream_t stream, int threads_per_block)
      : p_device(on_device), p_stream(stream),
        p_threads(threads_per_block) {}

  bool onDevice() const noexcept { return p_device; }
  cudaStream_t stream() const noexcept { return p_stream; }

  template <class F>
  void run(int64_t n, const F &f, const char *name) const
  {
    if (n <= 0) return;
    if (!p_device) {
      for (int64_t i = 0; i < n; i++) f(i);
      return;
    }
    const int threads = blockSize<F>();
    const int64_t blocks = (n + threads - 1) / threads;
    forEachKernel<F><<<static_cast<unsigned int>(blocks), threads, 0,
                       p_stream>>>(n, f);
    launchCheck(name);
  }

 private:
  template <class F>
  int blockSize() const
  {
    if (p_threads > 0) return p_threads;
    // One value per functor type, computed once (thread-safe static init)
    static const int cached = occupancyBlockSize<F>();
    return cached;
  }

  template <class F>
  static int occupancyBlockSize()
  {
    int min_grid = 0;
    int block = 0;
    cudaCheck(cudaOccupancyMaxPotentialBlockSize(&min_grid, &block,
                                                 forEachKernel<F>, 0, 0),
              "cudaOccupancyMaxPotentialBlockSize");
    return (block >= 32) ? (block / 32) * 32 : 128;
  }

  bool p_device = false;
  cudaStream_t p_stream = nullptr;
  int p_threads = 0;
};

}  // namespace batchpf
}  // namespace gridpack

#endif
