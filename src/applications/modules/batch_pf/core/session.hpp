/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   session.hpp
 * @date   2026-10-05
 *
 * @brief One accelerator session of the core plugin.
 *
 * A session owns the GPU stream, the engine and a worker thread. Batches
 * are queued by submit() and solved by the worker, so the calling rank
 * stays free to report finished cases or solve CPU-path cases while the
 * GPU works (guide 6.2, 8.15: reporting overlaps GPU batches). The worker
 * never calls MPI or GridPACK; it only touches the plugin's own data and
 * the caller's result buffers of the batch it is solving.
 */

#ifndef GRIDPACK_BATCHPF_CORE_SESSION_HPP
#define GRIDPACK_BATCHPF_CORE_SESSION_HPP

#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "common.hpp"
#include "engine.hpp"
#include "gridpack/batchpf/batchpf_plugin.h"

namespace gridpack {
namespace batchpf {

class Session {
 public:
  explicit Session(const batchpf_settings &settings);
  ~Session();
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  batchpf_device_info deviceInfo() const;
  void setModel(const batchpf_model &model);
  int plan(const batchpf_solver_params &params, int64_t expected_cases);
  int64_t submit(const batchpf_batch &batch, batchpf_results &results);
  /// OK when done, BATCHPF_ERR_TIMEOUT if still running, or the error
  batchpf_status wait(int64_t ticket, int timeout_ms);
  batchpf_diagnostics diagnostics() const;
  std::string lastError() const;
  void setLastError(const std::string &msg);

 private:
  struct Job {
    int64_t ticket = 0;
    const batchpf_batch *batch = nullptr;
    batchpf_results *results = nullptr;
  };
  struct Done {
    batchpf_status status = BATCHPF_OK;
    std::string message;
  };
  void workerLoop();
  int chooseBackend(int requested);
  int chooseCapacity(int64_t expected_cases);

  batchpf_settings p_settings;
  std::string p_plugin_dir;
  Logger p_log;
  bool p_on_device = true;
  Stream p_stream;
  batchpf_device_info p_info{};
  int p_profile = BATCHPF_MEMORY_AUTO;
  std::unique_ptr<ModelHost> p_model;
  std::unique_ptr<Engine> p_engine;
  int p_backend = BATCHPF_BACKEND_AUTO;
  int p_capacity = 0;

  mutable std::mutex p_mutex;
  std::condition_variable p_cv;
  std::deque<Job> p_queue;
  std::map<int64_t, Done> p_done;
  int64_t p_next_ticket = 1;
  bool p_stop = false;
  std::string p_last_error;
  std::thread p_worker;   // last member: started after everything else
};

}  // namespace batchpf
}  // namespace gridpack

#endif
