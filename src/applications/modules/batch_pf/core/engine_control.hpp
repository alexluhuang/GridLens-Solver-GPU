/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   engine_control.hpp
 * @date   2026-10-05
 *
 * @brief Per-case control logic of the batch engine (blocks B8.7, B8.8).
 *
 * This is GridPACK's PFAppModule::solve() loop, plus the contingency
 * driver's extra Q-limit check, rewritten as a state machine so that many
 * cases can share one GPU step at a time. It is plain host code with no GPU
 * dependence, so it can be tested on its own.
 *
 * GridPACK's sequence for one case (guide 5.3.3, 6.2):
 *
 *   solve():
 *     repeat (controller iterations, bounded):
 *       F0 = mismatch, J0, X = J0 \ F0; tol0 = |F0|; iter = 0
 *       while |F| > tolerance and iter < limit:
 *         x -= X; F, J at x; X = J \ F; iter++
 *         stagnation (|F| unchanged 5 times, Q limits on): check Q limits;
 *           if any bus converts, leave the loop ("early")
 *         if |F| > 100 tol0: diverged
 *       iter == limit: diverged
 *       if converged: check Q limits (unless early); repeat if a bus
 *         converted and the controller limit allows
 *       if not repeating: x -= X (the last step is applied)
 *   driver: if solve() succeeded and Contingency_analysis/qlim is on,
 *     check Q limits at the final state; if any bus converts, solve again.
 *
 * Each call to advanceSlot() consumes the results of the step just run
 * for the slot and sets which actions the next step must run for it.
 */

#ifndef GRIDPACK_BATCHPF_CORE_ENGINE_CONTROL_HPP
#define GRIDPACK_BATCHPF_CORE_ENGINE_CONTROL_HPP

#include <vector>

namespace gridpack {
namespace batchpf {

/// Rules taken from GridPACK's settings (batchpf_solver_params)
struct ControlRules {
  double tolerance = 1.0e-6;
  int max_iteration = 50;
  bool pf_qlim = true;            // Powerflow/qlim
  bool ca_qlim = true;            // Contingency_analysis/qlim
  int max_controller_iterations = 10;
  bool check_nonfinite = true;
  double residual_limit = 0.0;    // 0 = no residual check
};

/// What the last step produced for one slot
struct StepResult {
  double maxp = 0.0;              // largest |P mismatch| (pu)
  double maxq = 0.0;              // largest |Q mismatch| (pu)
  int argp = -1;                  // bus of maxp
  int argq = -1;
  int qviol = 0;                  // buses converted by a Q-limit check
  int member_status = 0;          // BATCHPF_MEMBER_* from the backend
  double residual = 0.0;          // |J X - F| (if checked)
  double rhs_norm = 0.0;          // |F|
};

/// One entry of GridPACK's per-iteration convergence history
struct IterationRecord {
  int argp = -1;
  int argq = -1;
  double maxp = 0.0;
  double maxq = 0.0;
};

/// State of one slot
struct SlotState {
  enum class Stage { Free, StartCtrl, NewtonIter, StagnationQcheck,
                     CtrlQcheck, FinalApply, CaQcheck, Done };
  Stage stage = Stage::Free;
  int case_idx = -1;
  // actions of the next step, run in this order
  bool act_apply = false;         // x -= X
  bool act_qcheck = false;        // reactive-limit check
  bool act_eval = false;          // mismatch, Jacobian, factor, solve
  int solve_no = 1;               // 2 after the driver's extra check
  int ctrl_iter = 0;
  int iter = 0;
  int stagnant = 0;
  double tol = 0.0, tol_org = 0.0, tol_prev = 0.0;
  bool ret = true, early = false, repeat = false;
  // GridPACK's convergence record of the last Newton loop
  int rec_iter = 0;
  double rec_tol = 0.0, rec_maxp = 0.0, rec_maxq = 0.0;
  int rec_argp = -1, rec_argq = -1;
  std::vector<IterationRecord> history;   // cleared per controller iteration
  // outcome
  int status = 3;                 // BATCHPF_CASE_NOT_RUN
  int health = 0;                 // BATCHPF_HEALTH_* bits
  int total_iters = 0, ctrl_total = 0, pv_to_pq = 0;
};

/// Put a newly filled slot at the start of its first controller iteration
void startSlot(SlotState &s, int case_idx);

/// Consume a step's results and choose the slot's next actions
void advanceSlot(SlotState &s, const StepResult &r, const ControlRules &rules);

}  // namespace batchpf
}  // namespace gridpack

#endif
