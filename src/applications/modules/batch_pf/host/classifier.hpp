/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   classifier.hpp
 * @date   2026-10-05
 *
 * @brief Contingency classifier and value-change builder (blocks B4.3,
 * B4.4).
 *
 * Each case is sorted into one of two paths (guide 6.4, 8.6.1):
 *  - GPU: GridPACK would solve it and every change it makes is a value
 *    change in the superset pattern (outages, lone-bus isolation, slack
 *    moves, PV to PQ changes);
 *  - CPU: everything else, solved or reported by GridPACK's own loop. That
 *    includes cases GridPACK reports without solving (ISLANDED, NO_SLACK),
 *    studies with controls the GPU does not reproduce (switched shunts,
 *    tap changers, area interchange, remote voltage regulation), and
 *    elements that cannot be found.
 *
 * GridPACK semantics are kept by construction: a case is classified by
 * applying it with GridPACK's setContingency(), reading what it did, and
 * restoring the network. Most N-1 cases cannot change the topology, so a
 * fast path handles them without the full routine, under conditions that
 * guarantee the same answer:
 *  - one branch element whose loss cannot disconnect anything (the branch
 *    keeps another element in service, or it is not a bridge of the
 *    network graph): no lone bus, one island, no slack move;
 *  - one branch element that is a bridge: if one end has no other branch,
 *    that end becomes a lone bus, which GridPACK isolates before looking
 *    for islands, so the rest stays one island and the case is solved with
 *    that bus isolated (unless the lone bus is the slack); otherwise the
 *    network splits into two islands of two or more buses and GridPACK
 *    reports ISLANDED without solving;
 *  - one generator not at the reference bus, or at a reference bus that
 *    keeps another unit online: no topology or slack change.
 * Before the fast path is used, the unmodified network is checked with
 * GridPACK's own routines (one island, no lone bus, slack has a unit); if
 * that fails, every case takes the full path. With shadow validation on,
 * sampled fast-path results are compared against the full path.
 */

#ifndef GRIDPACK_BATCHPF_HOST_CLASSIFIER_HPP
#define GRIDPACK_BATCHPF_HOST_CLASSIFIER_HPP

#include <string>
#include <vector>

#include "gridpack/applications/modules/powerflow/pf_app_module.hpp"
#include "gridpack/batchpf/batchpf_plugin.h"
#include "gridpack/batchpf/index.hpp"

namespace gridpack {
namespace batchpf {

/// Path a case takes (lifecycle "Classified", guide 6.6)
enum class CasePath { Gpu = 0, Cpu = 1 };

/// Why a case goes to the CPU path
enum class CpuReason {
  None = 0,
  StudyControls,      // switched shunts, LTC or area interchange enabled
  RemoteRegulation,   // remote voltage regulation would act
  NotFound,           // element or slack missing: GridPACK reports NO_SLACK
  Islanded,           // more than one island: GridPACK reports ISLANDED
  NoSlack,
  Other
};
const char *cpuReasonName(CpuReason r);

/// Classification record of one case (I-3)
struct CaseClass {
  CaseIndex event;
  CasePath path = CasePath::Cpu;
  CpuReason reason = CpuReason::None;
  bool fast = false;                 // classified by the fast path
  int island_count = 1;
  bool lone_bus = false;
  bool slack_transferred = false;
  BusIndex slack_bus;                // local index
  std::vector<batchpf_bus_update> bus_updates;
  std::vector<batchpf_edge_update> edge_updates;
};

class Classifier {
 public:
  /**
   * @param pf_app power flow module with the network in its contingency
   *        start state (after the base solve and clearQlimViolations)
   * @param model the exported superset model of that state
   * @param study_controls true if switched shunts, LTC or area interchange
   *        are enabled (then every case is a CPU case)
   */
  Classifier(gridpack::powerflow::PFAppModule &pf_app,
             boost::shared_ptr<gridpack::powerflow::PFNetwork> network,
             const gridpack::powerflow::SupersetModel &model, bool study_controls);

  /// Classify one case; the network is restored before returning
  CaseClass classify(CaseIndex event, gridpack::powerflow::Contingency &c);

  /// Classify with the full GridPACK routine only (for cross-checks)
  CaseClass classifyFull(CaseIndex event, gridpack::powerflow::Contingency &c);

  /// Whether the fast path is enabled, with the reason if not
  bool fastPathEnabled() const { return p_fast_ok; }
  const std::string &fastPathNote() const { return p_fast_note; }

 private:
  bool classifyFastBranch(CaseIndex event, gridpack::powerflow::Contingency &c,
                          CaseClass *out);
  bool classifyFastGenerator(CaseIndex event, gridpack::powerflow::Contingency &c,
                             CaseClass *out);
  void finishUpdates(const gridpack::powerflow::SupersetCaseState &state,
                     CaseClass *out) const;
  void findBridges();

  gridpack::powerflow::PFAppModule &p_app;
  boost::shared_ptr<gridpack::powerflow::PFNetwork> p_network;
  const gridpack::powerflow::SupersetModel &p_model;
  bool p_study_controls;
  bool p_fast_ok = false;
  std::string p_fast_note;
  int p_base_slack = -1;
  std::vector<char> p_bridge;        // per local branch: removal disconnects
  std::vector<int> p_degree;         // per bus: in-service branch objects
};

}  // namespace batchpf
}  // namespace gridpack

#endif
