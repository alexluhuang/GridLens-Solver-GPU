/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   settings.hpp
 * @date   2026-10-05
 *
 * @brief Settings resolver for the GPU batch path (block B14).
 *
 * Every choice that can be made at run time is a setting (guide 2.6, 8.16,
 * appendix G). The resolver reads the optional GPUBatch and Execution blocks
 * inside Contingency_analysis, takes deployment-level values from the
 * environment where the plan defines one, and records where each effective
 * value came from:
 *
 *   configuration.xml  >  environment variable  >  detected  >  default
 *
 * Invalid values stop the run at start-up with a message naming the key
 * (RT-5). Without a GPUBatch block the accelerator is off and ca.x behaves
 * exactly like stock GridPACK (RT-4).
 */

#ifndef GRIDPACK_BATCHPF_HOST_SETTINGS_HPP
#define GRIDPACK_BATCHPF_HOST_SETTINGS_HPP

#include <stdexcept>
#include <string>
#include <vector>

#include "gridpack/batchpf/batchpf_plugin.h"

namespace gridpack {
namespace utility {
class Configuration;
}
namespace batchpf {

/// Where an effective value came from (RT-3)
enum class Source { Default, Xml, Environment, Detected };
const char *sourceName(Source s);

template <class T>
struct Setting {
  T value;
  Source source = Source::Default;
};

/// GPUBatch/enabled
enum class Enabled { Off, Auto, On };
/// GPUBatch/onUnavailable
enum class OnUnavailable { Fallback, Error };
/// Execution/cpuBinding
enum class CpuBinding { Auto, None, PerformanceFirst };

/// Thrown for an invalid setting; the message names the key and the value
class SettingsError : public std::runtime_error {
 public:
  explicit SettingsError(const std::string &msg) : std::runtime_error(msg) {}
};

struct GpuBatchSettings {
  bool block_present = false;
  Setting<Enabled> enabled{Enabled::Off};
  Setting<OnUnavailable> on_unavailable{OnUnavailable::Fallback};
  Setting<std::string> plugin_path{""};          // "" = next to ca.x
  Setting<int> backend{BATCHPF_BACKEND_AUTO};
  Setting<int> device{0};
  Setting<int> batch_size{0};                    // 0 = automatic
  Setting<int> max_validated_batch{0};           // 0 = backend default
  Setting<bool> backfill{true};
  Setting<int> threads_per_block{0};             // 0 = automatic
  Setting<int> memory_profile{BATCHPF_MEMORY_AUTO};
  Setting<double> memory_headroom_gb{-1.0};      // < 0 = by memory profile
  Setting<double> max_memory_gb{0.0};            // 0 = no cap
  Setting<std::string> formulation{"superset"};
  Setting<int> planner_ordering{BATCHPF_ORDERING_AMD};
  Setting<double> pivot_tolerance{0.001};
  Setting<int> solve_placement{BATCHPF_SOLVE_GPU};
  Setting<int> warm_start{BATCHPF_WARM_START_BASE_CASE};
  Setting<int> refinement_steps{0};
  Setting<double> residual_limit{1.0e-6};
  Setting<double> pivot_limit{1.0e-12};
  Setting<bool> check_nonfinite{true};
  Setting<double> shadow_fraction{0.0};
  Setting<int> telemetry{BATCHPF_TELEMETRY_SUMMARY};
  Setting<bool> profiler_ranges{false};
};

struct ExecutionSettings {
  Setting<std::vector<int>> accelerator_ranks{{}};   // empty = automatic
  Setting<CpuBinding> cpu_binding{CpuBinding::Auto};
  Setting<int> log_level{BATCHPF_LOG_INFO};
};

struct ResolvedSettings {
  GpuBatchSettings gpu;
  ExecutionSettings exec;
  std::string cuda_visible_devices;   // for the log; "" if unset
  std::string jit_cache_path;         // CUDA_CACHE_PATH, for the log
};

/**
 * Read, merge and check the settings. Must be called while the main
 * configuration file is the open one (before the contingency list is
 * opened).
 * @throws SettingsError for any invalid value or unknown key
 */
ResolvedSettings resolveSettings(gridpack::utility::Configuration *config);

/// One line per effective value with its source, for the start-up log
std::vector<std::string> describeSettings(const ResolvedSettings &s);

}  // namespace batchpf
}  // namespace gridpack

#endif
