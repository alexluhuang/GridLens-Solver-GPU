/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   accelerator.hpp
 * @date   2026-10-05
 *
 * @brief Accelerator loader (block B13) and the C++ side of the plugin
 * interface I-10.
 *
 * ca.x links no CUDA library. When the GPU path is enabled, the loader
 * finds the core plugin, checks that it speaks the same interface major
 * version, and creates a session (guide 6.0). Any failure is reported with
 * a reason so the caller can fall back to GridPACK's CPU path. The class
 * owns the library handle and the session (RAII) and turns plugin status
 * codes into AcceleratorError exceptions.
 */

#ifndef GRIDPACK_BATCHPF_HOST_ACCELERATOR_HPP
#define GRIDPACK_BATCHPF_HOST_ACCELERATOR_HPP

#include <memory>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "gridpack/batchpf/batchpf_plugin.h"

namespace gridpack {
namespace batchpf {

/// A plugin call failed
class AcceleratorError : public std::runtime_error {
 public:
  AcceleratorError(batchpf_status code, const std::string &msg)
      : std::runtime_error(msg), p_code(code) {}
  batchpf_status code() const noexcept { return p_code; }

 private:
  batchpf_status p_code;
};

/**
 * Log sink shared by ca.x and the plugin. Messages from the plugin's worker
 * thread arrive concurrently with the main thread, so writes are
 * serialized.
 */
class HostLogger {
 public:
  HostLogger(int rank, int level) : p_rank(rank), p_level(level) {}
  void log(int level, const std::string &msg);
  int level() const noexcept { return p_level; }
  bool enabled(int level) const noexcept { return level <= p_level; }
  /// C callback for the plugin; user is a HostLogger
  static void callback(void *user, int32_t level, const char *message);

 private:
  int p_rank;
  int p_level;
  std::mutex p_mutex;
};

/// Directory of the running executable (for plugin paths relative to ca.x)
std::string executableDirectory();

/// Plugin directories to try, in order: setting, then relative to ca.x
std::vector<std::string> pluginSearchPath(const std::string &configured,
                                          const std::string &exe_dir);

class Accelerator {
  struct ConstructionKey {};
 public:
  explicit Accelerator(ConstructionKey);
  ~Accelerator();
  Accelerator(const Accelerator &) = delete;
  Accelerator &operator=(const Accelerator &) = delete;
  Accelerator(Accelerator &&) = delete;
  Accelerator &operator=(Accelerator &&) = delete;

  /**
   * Load the core plugin from the first directory that has it, check the
   * interface version and probe the GPU (device index `device`). Returns
   * nullptr with the reason in *why if the plugin cannot be loaded. A
   * failed GPU probe is not an error here (the CPU reference backend works
   * without a GPU); see probeOk().
   */
  static std::unique_ptr<Accelerator> load(const std::vector<std::string> &dirs,
                                           int device, std::string *why);

  /// Number of visible GPUs, by loading the plugin and probing (0 if none)
  static int countDevices(const std::vector<std::string> &dirs, std::string *why);

  /// Whether the GPU probe succeeded, and the reason if not
  bool probeOk() const noexcept { return p_probe_ok; }
  const std::string &probeMessage() const noexcept { return p_probe_msg; }
  const batchpf_device_info &probeInfo() const noexcept { return p_probe; }

  /**
   * Create the session. Returns false with the reason in *why if the
   * plugin refuses (no GPU, missing backend, ...).
   */
  bool createSession(batchpf_settings settings, std::string *why);

  batchpf_device_info deviceInfo();
  void setModel(const batchpf_model &model);
  int plan(const batchpf_solver_params &params, int64_t expected_cases);
  int64_t submit(const batchpf_batch &batch, batchpf_results &results);
  /// True when the batch is finished; false on timeout. Throws on failure.
  bool wait(int64_t ticket, int timeout_ms);
  batchpf_diagnostics diagnostics();

  const std::string &pluginFile() const noexcept { return p_file; }
  const std::string &pluginVersion() const noexcept { return p_version; }
  const std::string &buildInfo() const noexcept { return p_build; }

 private:
  struct Library;
  void check(batchpf_status st, const char *what);

  std::unique_ptr<Library> p_lib;
  batchpf_api p_api{};
  batchpf_session *p_session = nullptr;   // owned; destroyed in ~Accelerator
  std::string p_dir;                      // owned copy of plugin_dir
  std::string p_file;
  std::string p_version;
  std::string p_build;
  batchpf_device_info p_probe{};
  bool p_probe_ok = false;
  std::string p_probe_msg;
};

}  // namespace batchpf
}  // namespace gridpack

#endif
