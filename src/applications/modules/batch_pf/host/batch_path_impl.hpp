/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   batch_path_impl.hpp
 * @date   2026-10-05
 *
 * @brief Internal state of BatchPath, shared by batch_path.cpp (start-up,
 * preparation) and batch_run.cpp (execution, reporting, summary).
 */

#ifndef GRIDPACK_BATCHPF_HOST_BATCH_PATH_IMPL_HPP
#define GRIDPACK_BATCHPF_HOST_BATCH_PATH_IMPL_HPP

#include <mpi.h>

#include <memory>
#include <string>
#include <vector>

#include "accelerator.hpp"
#include "batch_path.hpp"
#include "classifier.hpp"
#include "host_platform.hpp"
#include "settings.hpp"

namespace gridpack {
namespace batchpf {

/// Arrays behind a batchpf_model record (kept alive while it is used)
struct ModelArrays {
  std::vector<int32_t> bus_type, row_start, edge_col, edge_mate;
  std::vector<double> g, b, p0, q0, v_init, theta_init, v_base, theta_base;
  std::vector<double> ql, ip, iq, yp, yq, qmax, qmin, eg, eb;
  batchpf_model record() const;
};

/// One row of the GPU outcome table (<outputFile>_gpu_outcomes.csv)
struct OutcomeRow {
  int event = 0;
  int path = 0;                // 0 gpu, 1 cpu, 2 cpu after GPU fallback
  int reason = 0;              // CpuReason
  int fast = 0;                // classified by the fast path
  int gpu_status = -1;         // BATCHPF_CASE_*, -1 if not on the GPU
  int health = 0;
  int iterations = 0;
  int total_iterations = 0;
  int controller_iterations = 0;
  int solves = 0;
  int pv_to_pq = 0;
  double final_tolerance = 0.0;
};

/// One row of the shadow validation table (<outputFile>_gpu_shadow.csv)
struct ShadowRow {
  int event = 0;
  int cpu_ok = 0;
  int gpu_ok = 0;
  double max_dv = 0.0;
  double max_dtheta = 0.0;
  int pv_cpu = 0;
  int pv_gpu = 0;
  int class_match = 1;         // fast-path classification equals full routine
};

struct BatchPath::Impl {
  // MPI layout
  gridpack::parallel::Communicator world;
  int rank = 0;
  int size = 1;
  MPI_Comm node_comm = MPI_COMM_NULL;
  int local_rank = 0;
  int local_size = 1;

  // start-up
  ResolvedSettings settings;
  bool requested = false;      // GPUBatch present and not off
  bool active = false;         // batch path will run
  std::unique_ptr<HostLogger> log;
  std::vector<std::string> plugin_dirs;
  CpuTopology topology;
  bool is_accelerator = false; // this rank was chosen to drive a GPU
  int device = 0;
  std::unique_ptr<Accelerator> acc;
  std::vector<int> accelerator_ranks;   // world ranks with a session

  // preparation
  gridpack::powerflow::PFAppModule *app = nullptr;
  boost::shared_ptr<gridpack::powerflow::PFNetwork> network;
  std::vector<gridpack::powerflow::Contingency> *events = nullptr;
  bool ca_qlim = true;
  std::string output_file;
  gridpack::powerflow::SupersetModel model;
  std::unique_ptr<Classifier> classifier;
  std::vector<CaseClass> classes;      // one per event
  std::vector<int> cpu_events;         // events on the CPU path
  std::vector<int> gpu_events;         // events on the GPU path
  std::vector<int> owner;              // per event: accelerator rank, or -1
  std::vector<int> my_gpu_events;      // GPU events this rank solves
  int capacity = 0;
  int history_capacity = 0;
  batchpf_solver_params params{};

  // results
  std::vector<OutcomeRow> outcomes;
  std::vector<ShadowRow> shadows;
  double t_classify = 0.0, t_plan = 0.0, t_cpu = 0.0, t_gpu_phase = 0.0;
  double t_report = 0.0, t_start = 0.0;
  int reported_gpu = 0, fallback = 0;
  double replica_bytes = 0.0;

  // helpers (batch_path.cpp)
  batchpf_settings pluginSettings() const;
  void info(const std::string &msg) const { if (log) log->log(BATCHPF_LOG_INFO, msg); }
  void warn(const std::string &msg) const { if (log) log->log(BATCHPF_LOG_WARN, msg); }
  bool shadowSelected(int event) const;
  // helpers (batch_run.cpp)
  void processGpuCase(int event, const batchpf_outcome &o, const double *v,
                      const double *theta, const int32_t *conv, const double *qreq,
                      const batchpf_mismatch_record *hist, int hist_count,
                      const ProcessCase &process);
  void shadowCompare(int event, const GpuCaseResult &res);
};

}  // namespace batchpf
}  // namespace gridpack

#endif
