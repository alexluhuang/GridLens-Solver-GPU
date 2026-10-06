/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   backend.hpp
 * @date   2026-10-05
 *
 * @brief Batch linear solver backends (the I-8 contract of guide 5.3.5).
 *
 * The engine assembles a batch of Jacobians that share one sparsity
 * pattern and asks a backend to factor them and solve for the Newton
 * steps. Three implementations exist:
 *   - Alg2Backend: hand-written batched LU (Zhou et al., guide appendix B),
 *     built into the core plugin;
 *   - CpuReferenceBackend: KLU on each member in turn, for tests and for
 *     machines without a GPU;
 *   - PluginBackend: an adapter over a backend plugin (cuDSS), loaded at
 *     run time through the C interface in batchpf_backend.h.
 * The backend is chosen at run time from GPUBatch/backend (RT-1).
 *
 * Buffers use the case-interleaved layout (entry p of member b at
 * p * capacity + b). A member is processed only if mask[b] != 0.
 */

#ifndef GRIDPACK_BATCHPF_CORE_BACKEND_HPP
#define GRIDPACK_BATCHPF_CORE_BACKEND_HPP

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "gridpack/batchpf/batchpf_backend.h"
#include "planner.hpp"

namespace gridpack {
namespace batchpf {

// Largest Algorithm 2 batch exercised by the 10k validation study.
constexpr int kAlg2ValidatedBatch = 2048;

/// What a backend can do (subset of batchpf_backend_caps)
struct BackendCaps {
  std::string name;
  std::string version;
  int max_batch = 0;          // 0 = no fixed limit
  int validated_batch = 0;    // 0 = no recorded validation limit
  bool member_masking = true;
  bool iterative_refinement = false;
  bool on_device = true;      // false: works on host memory
};

/// Everything a backend needs at setup (I-5)
struct BackendSetup {
  const JacobianPattern *pattern = nullptr;
  const LuPlan *lu = nullptr;          // fill-reducing order and factor plan
  const std::vector<double> *reference_values = nullptr;
  int capacity = 0;                    // batch capacity B (buffer stride)
  cudaStream_t stream = nullptr;
  int device = 0;
  int threads_per_block = 0;
  int refinement_steps = 0;
  double pivot_limit = 0.0;
  bool host_solve = false;             // GPUBatch/solvePlacement = host
  int factor_lanes = 0;                // Alg2: threads per member in every
                                       // factorization level; 0 = per level
  bool launch_graphs = true;           // Alg2: replay launches as CUDA graphs
  Logger logger;
};

/**
 * Interface of a batch linear solver backend. Implementations are not
 * copyable; calls are ordered on the stream given at setup.
 */
class SolverBackend {
 public:
  SolverBackend() = default;
  SolverBackend(const SolverBackend &) = delete;
  SolverBackend &operator=(const SolverBackend &) = delete;
  SolverBackend(SolverBackend &&) = delete;
  SolverBackend &operator=(SolverBackend &&) = delete;
  virtual ~SolverBackend() = default;

  virtual BackendCaps caps() const = 0;

  /**
   * Factor the active members.
   * @param values nnz * capacity Jacobian values (pattern order)
   * @param mask capacity flags (device-accessible)
   * @param member_status capacity BATCHPF_MEMBER_* (device-accessible)
   */
  virtual void refactorize(gsl::span<const double> values, gsl::span<const int> mask,
                           gsl::span<int> member_status) = 0;

  /**
   * Solve A x = rhs for the active members with the last factorization.
   * rhs and x hold n * capacity entries and must not alias.
   */
  virtual void solve(gsl::span<const double> rhs, gsl::span<double> x,
                     gsl::span<const int> mask, gsl::span<int> member_status) = 0;
};

/// Create a built-in backend. Throws Error if it cannot be used.
std::unique_ptr<SolverBackend> makeAlg2Backend(const BackendSetup &setup);
std::unique_ptr<SolverBackend> makeCpuReferenceBackend(const BackendSetup &setup);

/**
 * Load a backend plugin (e.g. libgridpack_batchpf_cudss.so) from dir.
 * Returns nullptr, with the reason in why, if it is missing or unusable.
 */
std::unique_ptr<SolverBackend> loadPluginBackend(const std::string &dir,
                                                 const std::string &name,
                                                 const BackendSetup &setup,
                                                 std::string *why);

/// Capabilities of a backend plugin without setting it up
bool probePluginBackend(const std::string &dir, const std::string &name,
                        BackendCaps *caps, std::string *why);

}  // namespace batchpf
}  // namespace gridpack

#endif
