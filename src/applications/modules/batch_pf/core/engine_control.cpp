/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   engine_control.cpp
 * @date   2026-10-05
 *
 * @brief Per-case control logic of the batch engine. See
 * engine_control.hpp for the GridPACK sequence it reproduces.
 *
 * Two places differ from GridPACK on purpose, both recorded in the
 * validation notes (guide 8.10):
 *  - When a Newton loop ends before its first iteration, GridPACK's
 *    convergence record keeps the mismatch summary of whatever it solved
 *    before (another case, on that rank). The record here uses the
 *    mismatch of this case's first evaluation instead.
 *  - GridPACK ignores the result of the driver's second solve. If that
 *    second solve fails here, the case is flagged so GridPACK's CPU path
 *    solves it, rather than reporting a failed GPU solve as converged.
 */

#include "engine_control.hpp"

#include <algorithm>
#include <cmath>

#include "gridpack/batchpf/batchpf_backend.h"
#include "gridpack/batchpf/batchpf_plugin.h"

namespace gridpack {
namespace batchpf {

namespace {

using Stage = SlotState::Stage;

void startCtrl(SlotState &s)
{
  s.ctrl_iter++;
  s.ctrl_total++;
  s.stage = Stage::StartCtrl;
  s.act_eval = true;
}

void finish(SlotState &s, int status)
{
  s.status = status;
  s.stage = Stage::Done;
}

/// After a controller iteration: start another, or apply the last step
void endCtrl(SlotState &s, const ControlRules &rules)
{
  if (s.repeat) {
    startCtrl(s);
    return;
  }
  s.act_apply = true;
  if (s.solve_no == 1 && s.ret && rules.ca_qlim) {
    s.stage = Stage::CaQcheck;
    s.act_qcheck = true;   // runs after the step is applied
  } else {
    s.stage = Stage::FinalApply;
  }
}

/// After the Newton loop of a controller iteration
void endNewton(SlotState &s, const ControlRules &rules)
{
  if (s.iter >= rules.max_iteration) {
    s.ret = false;
    s.health |= BATCHPF_HEALTH_ITERATION_LIMIT;
  }
  s.rec_iter = s.iter;
  s.rec_tol = s.tol;
  if (!s.ret) {
    s.repeat = false;
    endCtrl(s, rules);
  } else if (rules.pf_qlim && !s.early) {
    s.stage = Stage::CtrlQcheck;   // limits checked before the last step
    s.act_qcheck = true;
  } else if (s.early) {
    s.repeat = s.ctrl_iter < rules.max_controller_iterations;
    endCtrl(s, rules);
  } else {
    s.repeat = false;
    endCtrl(s, rules);
  }
}

/// Loop condition of the Newton loop
void newtonCheck(SlotState &s, const ControlRules &rules)
{
  if (s.tol > rules.tolerance && s.iter < rules.max_iteration) {
    s.stage = Stage::NewtonIter;
    s.act_apply = true;
    s.act_eval = true;
  } else {
    endNewton(s, rules);
  }
}

/// End of a Newton iteration after the stagnation test
void afterStagnation(SlotState &s, const ControlRules &rules)
{
  s.tol_prev = s.tol;
  if (s.tol > 100.0 * s.tol_org) {
    s.ret = false;
    s.health |= BATCHPF_HEALTH_MISMATCH_GROWTH;
    endNewton(s, rules);
  } else {
    newtonCheck(s, rules);
  }
}

void record(SlotState &s, const StepResult &r)
{
  s.rec_maxp = r.maxp;
  s.rec_maxq = r.maxq;
  s.rec_argp = r.argp;
  s.rec_argq = r.argq;
}

}  // namespace

void startSlot(SlotState &s, int case_idx)
{
  s = SlotState();
  s.case_idx = case_idx;
  s.solve_no = 1;
  startCtrl(s);
}

void advanceSlot(SlotState &s, const StepResult &r, const ControlRules &rules)
{
  if (s.stage == Stage::Done || s.stage == Stage::Free) return;
  const bool evaluated = s.act_eval;
  s.act_apply = s.act_qcheck = s.act_eval = false;
  const double tol = std::max(r.maxp, r.maxq);

  // Health checks (B8.8): any failure stops the case on the GPU
  if (evaluated) {
    bool bad = false;
    if (r.member_status == BATCHPF_MEMBER_SMALL_PIVOT) {
      s.health |= BATCHPF_HEALTH_SMALL_PIVOT;
      bad = true;
    }
    if (r.member_status == BATCHPF_MEMBER_NONFINITE ||
        (rules.check_nonfinite && !std::isfinite(tol))) {
      s.health |= BATCHPF_HEALTH_NONFINITE;
      bad = true;
    }
    if (rules.residual_limit > 0.0 && r.rhs_norm > 0.0 &&
        !(r.residual <= rules.residual_limit * r.rhs_norm)) {
      s.health |= BATCHPF_HEALTH_RESIDUAL;
      bad = true;
    }
    if (bad) {
      finish(s, BATCHPF_CASE_FLAGGED);
      return;
    }
  }

  switch (s.stage) {
    case Stage::StartCtrl:
      // first evaluation of a controller iteration
      s.tol = tol;
      s.tol_org = tol;
      s.tol_prev = tol;
      s.iter = 0;
      s.stagnant = 0;
      s.ret = true;
      s.early = false;
      s.history.clear();
      record(s, r);
      newtonCheck(s, rules);
      break;
    case Stage::NewtonIter:
      s.tol = tol;
      s.iter++;
      s.total_iters++;
      record(s, r);
      s.history.push_back({r.argp, r.argq, r.maxp, r.maxq});
      if (rules.pf_qlim && std::fabs(s.tol - s.tol_prev) < 1.0e-10) {
        s.stagnant++;
        if (s.stagnant >= 5) {
          s.stage = Stage::StagnationQcheck;
          s.act_qcheck = true;
          break;
        }
      } else {
        s.stagnant = 0;
      }
      afterStagnation(s, rules);
      break;
    case Stage::StagnationQcheck:
      if (r.qviol > 0) {
        s.pv_to_pq += r.qviol;
        s.early = true;
        s.health |= BATCHPF_HEALTH_STAGNATION;
        endNewton(s, rules);
      } else {
        s.stagnant = 0;
        afterStagnation(s, rules);
      }
      break;
    case Stage::CtrlQcheck:
      if (r.qviol > 0) {
        s.pv_to_pq += r.qviol;
        s.repeat = s.ctrl_iter < rules.max_controller_iterations;
      } else {
        s.repeat = false;
      }
      endCtrl(s, rules);
      break;
    case Stage::CaQcheck:
      if (r.qviol > 0) {
        s.pv_to_pq += r.qviol;
        s.solve_no = 2;
        s.ctrl_iter = 0;
        startCtrl(s);
      } else {
        finish(s, BATCHPF_CASE_CONVERGED);
      }
      break;
    case Stage::FinalApply:
      if (s.solve_no == 1) {
        finish(s, s.ret ? BATCHPF_CASE_CONVERGED : BATCHPF_CASE_DIVERGED);
      } else {
        finish(s, s.ret ? BATCHPF_CASE_CONVERGED : BATCHPF_CASE_FLAGGED);
      }
      break;
    default:
      break;
  }
}

}  // namespace batchpf
}  // namespace gridpack
