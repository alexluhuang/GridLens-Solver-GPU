/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   pf_superset_model.hpp
 * @date   2026-10-05
 *
 * @brief Plain data exported from a power flow network for the GPU batch
 * contingency path.
 *
 * The batch path solves many contingency cases at once with one shared
 * Jacobian sparsity pattern (the "superset" pattern: a 2x2 block for every
 * bus and for every bus pair joined by a branch, kept even when the branch
 * is switched off). This header describes what GridPACK hands over:
 *
 *  - SupersetModel: the network as it stands at the start of every
 *    contingency solve (statuses, admittances, scheduled injections, bus
 *    types), plus the base-case solution used as a warm start.
 *  - SupersetCaseState: the absolute values that change when one
 *    contingency is applied, read back from GridPACK's own components so
 *    that the batch path sees exactly what GridPACK's solver would see.
 *  - ContingencyEffects: what setContingency() did to the topology.
 *
 * Nothing here changes GridPACK behavior; the structs are filled by
 * PFFactoryModule export/capture calls that only read component state.
 */
// -------------------------------------------------------------

#ifndef _pf_superset_model_h_
#define _pf_superset_model_h_

#include <vector>

namespace gridpack {
namespace powerflow {

/// Role of a bus in the power flow equations, at the start of a solve
enum SupersetBusType {
  SUPERSET_PQ = 1,        // P and Q specified
  SUPERSET_PV = 2,        // P and V specified (voltage controlled)
  SUPERSET_REF = 3,       // reference (slack) bus
  SUPERSET_ISOLATED = 4   // removed from the equations
};

/// One bus of the exported network (local index order)
struct SupersetBus {
  int original_index;      // bus number in the RAW file
  int type;                // SupersetBusType
  double g_diag, b_diag;   // diagonal entry of the admittance matrix (pu)
  double p0, q0;           // scheduled net injection (pu), from setSBus()
  double v_init, theta_init;     // values resetVoltage() restores (pu, rad)
  double v_solved, theta_solved; // values at export time (base solution)
  double pl, ql, ip, iq, yp, yq; // in-service load totals (RAW units)
  double dg_q;             // in-service distributed generation Q on the
                           // loads (MVAr), not included in ql
  double dc_p, dc_q;       // dc converter power drawn (MW, MVAr), which p0
                           // and q0 include
  double qmax, qmin;       // in-service generator Q limit totals (MVAr);
                           // infinite for a bus without generators
  bool remote_regulation;  // remote voltage regulation would adjust this bus
  bool switched_shunt;     // bus has an active switched shunt
};

/**
 * Network exported for the batch path. Off-diagonal admittances are stored
 * per directed bus pair ("edge"): row bus k, column bus m. Several GridPACK
 * branch objects joining the same pair are summed into one edge.
 */
struct SupersetModel {
  double sbase;                       // system MVA base
  std::vector<SupersetBus> buses;
  std::vector<int> row_start;         // edges of bus k: [row_start[k], row_start[k+1])
  std::vector<int> edge_col;          // column bus (local index) of each edge
  std::vector<double> edge_g, edge_b; // off-diagonal admittance of each edge
  std::vector<int> edge_mate;         // index of the reverse edge (m, k)
  std::vector<int> branch_edge;       // per local branch: edge (bus1, bus2), -1 if none
  bool has_remote_regulation;         // any bus with remote_regulation set
  bool has_switched_shunt;            // any bus with switched_shunt set
  bool has_ltc;                       // any branch with tap changer control
};

/// Absolute bus values after a contingency is applied
struct SupersetBusUpdate {
  int bus;                 // local index
  int type;                // SupersetBusType
  double g_diag, b_diag;
  double p0, q0;
  double qmax, qmin;
  bool remote_regulation;
};

/// Absolute edge admittance after a contingency is applied
struct SupersetEdgeUpdate {
  int edge;
  double g, b;
};

/// Everything a contingency changes, as absolute values
struct SupersetCaseState {
  std::vector<SupersetBusUpdate> buses;
  std::vector<SupersetEdgeUpdate> edges;
};

/// What the last setContingency() did to topology and the slack bus
struct ContingencyEffects {
  int island_count;                // from detectIslands()
  bool lone_bus;                   // checkLoneBus() isolated a bus
  bool slack_transferred;          // checkAndTransferSlack() moved the slack
  int slack_bus;                   // local index of the reference bus now
  std::vector<int> isolated_buses; // local indices isolated by this contingency
};

} // powerflow
} // gridpack
#endif
