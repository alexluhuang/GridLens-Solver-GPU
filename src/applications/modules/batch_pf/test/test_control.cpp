/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   test_control.cpp
 * @date   2026-10-05
 *
 * @brief batchpf.unit.control: the per-case control logic of the batch
 * engine follows GridPACK's PFAppModule::solve() and the contingency
 * driver step for step.
 *
 * Each scenario scripts the mismatch norms the "GPU" would produce and the
 * number of PV to PQ conversions each Q-limit check finds, runs the state
 * machine, and compares the sequence of actions and the outcome with what
 * GridPACK's code does for the same numbers. No GPU is needed.
 */

#include <iostream>
#include <cstddef>
#include "gridpack/batchpf/index.hpp"
#include <string>
#include <vector>

#include "../core/engine_control.hpp"
#include "gridpack/batchpf/batchpf_backend.h"
#include "gridpack/batchpf/batchpf_plugin.h"

using namespace gridpack::batchpf;

namespace {

/// Scripted inputs for one case
struct Script {
  std::vector<double> tol;      // mismatch norm of each evaluation, in order
  std::vector<int> qviol;       // conversions found by each Q-limit check
  int bad_eval = -1;            // evaluation that reports a tiny pivot
  std::vector<double> dc{};     // converter change found by each dc step (pu)
};

struct Run {
  std::string actions;          // one letter group per step: A(pply) Q
                                // S(tart dc lines) D(c step) E(val)
  SlotState s;
};

Run simulate(const Script &sc, const ControlRules &rules)
{
  Run run;
  startSlot(run.s, CaseIndex{0});
  std::size_t ne = 0, nq = 0, nd = 0;
  for (int step = 0; step < 1000 && run.s.stage != SlotState::Stage::Done; step++) {
    StepResult r;
    std::string a;
    if (run.s.act_apply) a += "A";
    if (run.s.act_qcheck) {
      a += "Q";
      r.qviol = nq < sc.qviol.size() ? sc.qviol[nq] : 0;
      nq++;
    }
    if (run.s.act_dcstart) a += "S";
    if (run.s.act_dccheck) {
      a += "D";
      r.dc_change = nd < sc.dc.size() ? sc.dc[nd] : 0.0;
      nd++;
    }
    if (run.s.act_eval) {
      a += "E";
      const double t = ne < sc.tol.size() ? sc.tol[ne] : sc.tol.back();
      r.maxp = t;
      r.maxq = t / 2;
      r.argp = 3;
      r.argq = 4;
      if (static_cast<int>(ne) == sc.bad_eval) r.member_status = BATCHPF_MEMBER_SMALL_PIVOT;
      ne++;
    }
    run.actions += (run.actions.empty() ? "" : " ") + a;
    advanceSlot(run.s, r, rules);
  }
  return run;
}

ControlRules rules(bool pf_qlim, bool ca_qlim, int maxit = 50, int max_ctrl = 10)
{
  ControlRules r;
  r.tolerance = 1.0e-6;
  r.max_iteration = maxit;
  r.pf_qlim = pf_qlim;
  r.ca_qlim = ca_qlim;
  r.max_controller_iterations = max_ctrl;
  return r;
}

/// The same rules for a network with two-terminal dc lines
ControlRules dcRules(bool pf_qlim, bool ca_qlim, int max_ctrl = 10)
{
  ControlRules r = rules(pf_qlim, ca_qlim, 50, max_ctrl);
  r.dc_lines = true;
  r.hvdc_tolerance = 1.0e-4;
  return r;
}

}  // namespace

int main()
{
  int failures = 0;
  const auto check = [&](bool ok, const std::string &what) {
    if (!ok) {
      std::cout << "FAILED: " << what << "\n";
      failures++;
    }
  };
  // 1. Plain convergence: GridPACK evaluates, then applies and evaluates
  //    until the norm is small, then applies the last step once more.
  {
    const Run r = simulate({{1e-1, 1e-3, 1e-8}, {}}, rules(false, false));
    check(r.actions == "E AE AE A", "plain convergence actions: " + r.actions);
    check(r.s.status == BATCHPF_CASE_CONVERGED, "plain convergence status");
    check(r.s.rec_iter == 2, "plain convergence counts 2 iterations");
    check(r.s.history.size() == 2, "history has one record per iteration");
  }
  // 2. Already converged at the first evaluation: no iteration
  {
    const Run r = simulate({{1e-9}, {}}, rules(false, false));
    check(r.actions == "E A", "converged at start: " + r.actions);
    check(r.s.rec_iter == 0 && r.s.status == BATCHPF_CASE_CONVERGED, "zero iterations");
  }
  // 3. Iteration limit: iter == maxIteration means not converged
  {
    const Run r = simulate({{1e-1}, {}}, rules(false, false, 3));
    check(r.actions == "E AE AE AE A", "iteration limit actions: " + r.actions);
    check(r.s.status == BATCHPF_CASE_DIVERGED, "iteration limit is diverged");
    check((r.s.health & BATCHPF_HEALTH_ITERATION_LIMIT) != 0, "iteration limit flag");
  }
  // 4. Growth past 100x the starting mismatch stops the loop
  {
    const Run r = simulate({{1e-1, 20.0}, {}}, rules(false, false));
    check(r.actions == "E AE A", "growth actions: " + r.actions);
    check(r.s.status == BATCHPF_CASE_DIVERGED, "growth is diverged");
    check((r.s.health & BATCHPF_HEALTH_MISMATCH_GROWTH) != 0, "growth flag");
  }
  // 5. Q limits inside solve(): checked before the last step is applied; a
  //    conversion starts another controller iteration from the same state
  {
    const Run r = simulate({{1e-1, 1e-8, 1e-2, 1e-9}, {1, 0}}, rules(true, false));
    check(r.actions == "E AE Q E AE Q A", "Q-limit re-solve actions: " + r.actions);
    check(r.s.pv_to_pq == 1 && r.s.ctrl_total == 2, "one conversion, two controller iterations");
    check(r.s.status == BATCHPF_CASE_CONVERGED, "Q-limit re-solve converged");
    check(r.s.rec_iter == 1, "record keeps the last Newton loop only");
  }
  // 6. Controller limit: a conversion in the last allowed iteration does
  //    not trigger another solve
  {
    const Run r = simulate({{1e-1, 1e-8, 1e-2, 1e-9}, {1, 1}}, rules(true, false, 50, 2));
    check(r.actions == "E AE Q E AE Q A", "controller limit actions: " + r.actions);
    check(r.s.pv_to_pq == 2 && r.s.ctrl_total == 2, "conversions counted, no third iteration");
  }
  // 7. The driver's extra Q-limit check runs after the last step; a
  //    conversion means a complete second solve
  {
    const Run r = simulate({{1e-1, 1e-8, 1e-3, 1e-9}, {2}}, rules(false, true));
    check(r.actions == "E AE AQ E AE A", "driver re-solve actions: " + r.actions);
    check(r.s.solve_no == 2 && r.s.pv_to_pq == 2, "second solve after two conversions");
    check(r.s.status == BATCHPF_CASE_CONVERGED, "driver re-solve converged");
  }
  // 8. A failed second solve goes to the CPU path instead of being accepted
  {
    const Run r = simulate({{1e-1, 1e-8, 1e-3, 1.0}, {1}}, rules(false, true));
    check(r.s.status == BATCHPF_CASE_FLAGGED, "failed second solve is flagged");
  }
  // 9. Stagnation with Q limits on: five unchanged norms trigger an early
  //    Q-limit check; a conversion ends the loop and repeats the iteration
  {
    const Run r = simulate({{1e-1, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-2, 1e-9}, {1, 0}},
                           rules(true, false));
    check(r.actions == "E AE AE AE AE AE AE Q E AE Q A", "stagnation actions: " + r.actions);
    check((r.s.health & BATCHPF_HEALTH_STAGNATION) != 0, "stagnation flag");
    check(r.s.status == BATCHPF_CASE_CONVERGED, "stagnation then converged");
  }
  // 10. Health: a tiny pivot stops the case on the GPU at once
  {
    Script sc{{1e-1, 1e-3, 1e-8}, {}};
    sc.bad_eval = 1;
    const Run r = simulate(sc, rules(false, false));
    check(r.actions == "E AE", "health stop actions: " + r.actions);
    check(r.s.status == BATCHPF_CASE_FLAGGED, "tiny pivot is flagged");
    check((r.s.health & BATCHPF_HEALTH_SMALL_PIVOT) != 0, "pivot flag");
  }
  // 11. dc lines (sequential ac/dc, PFAppModule::solve() check 5): after a
  //     converged Newton loop and before its last step, a converter change
  //     above hvdcTolerance repeats the controller iteration
  {
    Script sc{{1e-1, 1e-8, 1e-2, 1e-9}, {}};
    sc.dc = {1e-3, 1e-8};
    const Run r = simulate(sc, dcRules(false, false));
    check(r.actions == "E AE D E AE D A", "dc re-solve actions: " + r.actions);
    check(r.s.ctrl_total == 2 && r.s.status == BATCHPF_CASE_CONVERGED, "dc re-solve converged");
  }
  // 12. Reactive limits and dc lines are checked in the same step, limits
  //     first; either one repeats the iteration
  {
    Script sc{{1e-1, 1e-8, 1e-2, 1e-9}, {1, 0}};
    sc.dc = {1e-8, 1e-8};
    const Run r = simulate(sc, dcRules(true, false));
    check(r.actions == "E AE QD E AE QD A", "Q-limit and dc actions: " + r.actions);
    check(r.s.pv_to_pq == 1 && r.s.ctrl_total == 2, "conversion repeats with dc lines");
  }
  // 13. The driver's second solve starts the dc lines again (startHVDC)
  {
    Script sc{{1e-1, 1e-8, 1e-3, 1e-9}, {2}};
    sc.dc = {1e-8, 1e-8};
    const Run r = simulate(sc, dcRules(false, true));
    check(r.actions == "E AE D AQ SE AE D A", "second solve restarts dc: " + r.actions);
    check(r.s.solve_no == 2 && r.s.status == BATCHPF_CASE_CONVERGED, "second solve converged");
  }
  // 14. The controller limit also ends repeated dc changes
  {
    Script sc{{1e-1, 1e-8, 1e-2, 1e-9}, {}};
    sc.dc = {1.0, 1.0};
    const Run r = simulate(sc, dcRules(false, false, 2));
    check(r.actions == "E AE D E AE D A", "dc controller limit actions: " + r.actions);
    check(r.s.ctrl_total == 2, "no third controller iteration");
  }
  // 15. A failed Newton loop is not followed by a dc step
  {
    const Run r = simulate({{1e-1, 20.0}, {}}, dcRules(true, false));
    check(r.actions == "E AE A", "no dc step after divergence: " + r.actions);
    check(r.s.status == BATCHPF_CASE_DIVERGED, "diverged with dc lines");
  }
  // 16. A stagnation conversion still runs the dc step and repeats
  {
    Script sc{{1e-1, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-2, 1e-9}, {1}};
    sc.dc = {1e-8, 1e-8};
    const Run r = simulate(sc, dcRules(true, false));
    check(r.actions == "E AE AE AE AE AE AE Q D E AE QD A", "stagnation with dc: " + r.actions);
    check(r.s.status == BATCHPF_CASE_CONVERGED, "stagnation with dc converged");
  }
  if (failures == 0) {
    std::cout << "No errors detected\n";
    return 0;
  }
  std::cout << failures << " failure detected\n";
  return 1;
}
