/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   engine.hpp
 * @date   2026-10-05
 *
 * @brief Batched Newton-Raphson engine (block B8).
 *
 * The engine keeps B "slots". Each slot holds one contingency case and
 * walks it through exactly the sequence GridPACK's PFAppModule::solve()
 * and the contingency driver follow (guide 5.3.3, 6.2):
 *
 *   for each controller iteration (bounded):
 *     evaluate mismatch and Jacobian, solve, then repeat
 *       { apply step; evaluate; solve } while mismatch > tolerance,
 *       stopping on the iteration limit, on growth past 100x the starting
 *       mismatch, or on stagnation (with an early Q-limit check);
 *     check reactive limits (PV to PQ conversion) and repeat if any
 *     converted; otherwise apply the last step and stop.
 *   If the solve succeeded and Contingency_analysis/qlim is on, check
 *   limits once more and, if any bus converts, solve again.
 *
 * All slots advance one step at a time. A step runs, for the slots that
 * need it, in this order: apply the previous step, reactive-limit check,
 * evaluate (mismatch, Jacobian, factor, solve). The per-slot decisions are
 * made on the host from a handful of numbers read back after the step.
 * Finished slots are refilled from the queue when backfill is on,
 * otherwise the next wave starts when all slots are done.
 */

#ifndef GRIDPACK_BATCHPF_CORE_ENGINE_HPP
#define GRIDPACK_BATCHPF_CORE_ENGINE_HPP

#include <cuda_runtime.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "backend.hpp"
#include "common.hpp"
#include "engine_control.hpp"
#include "gridpack/batchpf/batchpf_plugin.h"
#include "planner.hpp"

namespace gridpack {
namespace batchpf {

struct EngineBuffers;   // defined in engine.cu

/// Host copy of the superset model (I-4)
struct ModelHost {
  int n_bus = 0;
  int n_edge = 0;
  double sbase = 100.0;
  int base_slack = -1;
  std::vector<int> bus_type, row_start, edge_col, edge_mate, edge_row;
  std::vector<double> g, b, p0, q0, v_init, theta_init, v_base, theta_base;
  std::vector<double> ql, ip, iq, yp, yq, qmax, qmin, eg, eb;
};

/// Copy and check a model record from ca.x
ModelHost copyModel(const batchpf_model &model);

/// Engine options resolved by the session
struct EngineConfig {
  bool on_device = true;
  cudaStream_t stream = nullptr;
  int device = 0;
  int threads_per_block = 0;
  int backend = BATCHPF_BACKEND_ALG2;   // a concrete backend, not AUTO
  std::string plugin_dir;
  int ordering = 0;                     // KLU code: 0 AMD, 1 COLAMD
  double pivot_tolerance = 0.001;
  int refinement_steps = 0;
  double residual_limit = 0.0;          // 0 = no residual check
  double pivot_limit = 0.0;
  bool check_nonfinite = true;
  bool backfill = false;
  bool host_solve = false;
  bool exchange_pinned = true;          // false: device buffers + copies
  int telemetry = BATCHPF_TELEMETRY_SUMMARY;
  bool profiler_ranges = false;
};

class Engine {
 public:
  Engine(ModelHost model, EngineConfig config, Logger logger);
  ~Engine();
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;
  Engine(Engine &&) = delete;
  Engine &operator=(Engine &&) = delete;

  /**
   * Build the superset pattern, the reference Jacobian (base case) and the
   * fixed-order LU plan. Independent of the batch size.
   */
  void plan(const batchpf_solver_params &params);

  /// Bytes of GPU working memory one slot needs (after plan())
  double bytesPerSlot() const;

  /// Bytes of GPU memory shared by all slots (after plan())
  double sharedBytes() const;

  /**
   * Allocate buffers for B slots and set up the backend. May be called
   * again with another size (e.g. during the batch-size sweep).
   */
  void allocate(int capacity);

  int capacity() const noexcept { return p_B; }

  /// Select the backend used by the next allocate() (BATCHPF_BACKEND_*)
  void setBackend(int backend) { p_config.backend = backend; }

  /**
   * Time one factorization and solve of the reference Jacobian in every
   * slot; returns seconds per slot. Used to find where throughput levels
   * off (guide 8.7).
   */
  double timeReferenceSolve(int repetitions);

  /// Solve every case of a batch, writing into the caller's buffers
  void run(const batchpf_batch &batch, batchpf_results &results);

  /// Totals for telemetry
  const batchpf_diagnostics &diagnostics() const noexcept { return p_diag; }

  /// Name and version of the backend in use
  BackendCaps backendCaps() const;

 private:
  void runImpl(const batchpf_batch &batch, batchpf_results &results);
  void fillSlots(const batchpf_batch &batch, const std::vector<MemberIndex> &slots,
                 const std::vector<CaseIndex> &cases);
  void step();
  void finishSlots(const batchpf_batch &batch, batchpf_results &results,
                   const std::vector<MemberIndex> &slots);
  void referenceJacobian(std::vector<double> *values,
                         std::vector<double> *rhs);
  double elapsed(int phase) const;

  ModelHost p_model;
  EngineConfig p_config;
  Logger p_log;
  batchpf_solver_params p_params{};
  JacobianPattern p_pattern;
  LuPlan p_lu;
  std::vector<double> p_ref_values;
  std::vector<double> p_ref_rhs;
  int p_B = 0;
  std::unique_ptr<EngineBuffers> p_buf;
  std::unique_ptr<SolverBackend> p_backend;
  ControlRules p_rules;
  std::vector<SlotState> p_slots;
  std::vector<StepResult> p_results;
  batchpf_diagnostics p_diag{};
};

}  // namespace batchpf
}  // namespace gridpack

#endif
