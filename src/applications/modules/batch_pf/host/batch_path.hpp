/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   batch_path.hpp
 * @date   2026-10-05
 *
 * @brief The batch-aware part of the contingency driver (blocks B1, B5,
 * B7, B10, B11; GridPACK extension E3).
 *
 * The contingency driver keeps doing everything it did: read the RAW file,
 * solve the base case, build the contingency list, and report each case
 * with GridPACK's checks and writers. This class adds, only when the
 * GPUBatch block asks for it:
 *
 *   start-up (R0)   settings, accelerator roles, plugin loading
 *   prepare  (R1)   export the network (B5), classify every case (B4),
 *                   plan the batched factorization (B6, in the plugin)
 *   run      (R2)   CPU-path cases through GridPACK's loop on all ranks;
 *                   GPU cases in batches on the accelerator ranks; their
 *                   results handed to any rank that asks, injected into
 *                   that rank's network copy and reported by GridPACK
 *                   (B11); cases the GPU flags or cannot solve go back to
 *                   GridPACK's CPU solve (B10, R3)
 *   finish  (R5)   GPU outcome table, shadow validation table, telemetry
 *
 * The driver supplies one function that processes a case: with a GPU
 * result it injects and reports it; without one it runs GridPACK's normal
 * per-case solve. Without a GPUBatch block nothing here runs (RT-4).
 */

#ifndef GRIDPACK_BATCHPF_HOST_BATCH_PATH_HPP
#define GRIDPACK_BATCHPF_HOST_BATCH_PATH_HPP

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "gridpack/applications/modules/powerflow/pf_app_module.hpp"
#include "gridpack/applications/modules/powerflow/pf_hvdc.hpp"
#include "gridpack/parallel/communicator.hpp"
#include "gridpack/utilities/results_exporter.hpp"

namespace gridpack {
namespace utility {
class Configuration;
}
namespace batchpf {

/// A GPU solution handed to the driver for reporting (I-9)
struct GpuCaseResult {
  std::vector<double> v;           // per local bus (pu)
  std::vector<double> theta;       // per local bus (rad)
  std::vector<int> qlim_conversion;
  std::vector<double> q_required;
  std::vector<gridpack::powerflow::HVDCSolution> dc;   // per dc line, final
  gridpack::utility::ConvergenceSummary convergence;
  // The classifier showed the case leaves one island and the unmodified
  // network has no lone buses, islands or slack without a unit, so the
  // reporter may use PFAppModule's known-topology shortcuts (P2)
  bool known_topology = false;
};

enum class ReportStatus { Ok, Islanded, NoSlack, SlackOverload, Diverged,
                          NumericalFailure, Count };

/// State captured by the reporter before it restores the network (guide 8.11).
struct CaseReport {
  ReportStatus status = ReportStatus::Diverged;
  int iterations = -1;          // no solve record for unsolved cases or exceptions
  double final_tolerance = 0.0;
  int pv_buses = 0;
  int pq_buses = 0;
};

/// processCase(event index, GPU result or nullptr), returning the reported state.
using ProcessCase = std::function<CaseReport(int, const GpuCaseResult *)>;

class BatchPath {
 public:
  /**
   * Resolve the settings and, when the path is requested, choose the ranks
   * that drive GPUs and load the plugin on them. Collective over world.
   * Prints a clear message and stops the run for invalid settings, or when
   * enabled=on with onUnavailable=error and no GPU can be used.
   */
  BatchPath(gridpack::utility::Configuration *config,
            gridpack::parallel::Communicator &world);
  ~BatchPath();
  BatchPath(const BatchPath &) = delete;
  BatchPath &operator=(const BatchPath &) = delete;
  BatchPath(BatchPath &&) = delete;
  BatchPath &operator=(BatchPath &&) = delete;

  /// True if the batch path will run (some rank has a working accelerator)
  bool active() const;

  /**
   * Export, classify and plan. Call after the base case is solved and the
   * Q-limit changes are cleared, right before the contingency loop.
   * Collective over world.
   */
  void prepare(gridpack::powerflow::PFAppModule &pf_app,
               boost::shared_ptr<gridpack::powerflow::PFNetwork> network,
               std::vector<gridpack::powerflow::Contingency> &events,
               bool ca_qlim, const std::string &output_file);

  /// Solve and report every case. Collective over world.
  void run(const ProcessCase &process);

  /// Write the GPU sidecar tables and the telemetry summary. Collective.
  void finish();

 private:
  struct Impl;
  std::unique_ptr<Impl> p_impl;
};

}  // namespace batchpf
}  // namespace gridpack

#endif
