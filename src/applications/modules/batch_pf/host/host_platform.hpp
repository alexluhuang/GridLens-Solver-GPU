/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   host_platform.hpp
 * @date   2026-10-05
 *
 * @brief Host side of the platform layer (block B9-H): CPU topology,
 * placement of ranks on cores, host memory figures. No CUDA here.
 *
 * Core types are discovered at run time, never hard-coded (guide 7.1.2,
 * 8.9). On a DGX Spark the 10 Cortex-X925 performance cores have a 2 MB L2
 * and the 10 Cortex-A725 efficiency cores 512 KB; on other Arm systems the
 * cores may all be alike. The placement rules are those of guide 7.1.2:
 * GPU-driving ranks on performance cores (two cores each: one for the
 * rank, one for the plugin's worker thread), CPU-path ranks next on
 * performance cores and then on efficiency cores, and one core left for
 * the operating system when there are enough.
 */

#ifndef GRIDPACK_BATCHPF_HOST_HOST_PLATFORM_HPP
#define GRIDPACK_BATCHPF_HOST_HOST_PLATFORM_HPP

#include <string>
#include <vector>

#include "settings.hpp"

namespace gridpack {
namespace batchpf {

struct CpuCore {
  int id = 0;
  bool performance = true;
  long l2_bytes = 0;
  long l3_bytes = 0;
  int capacity = 0;          // /sys/.../cpu_capacity, 0 if unknown
  std::string part;          // "CPU part" from /proc/cpuinfo
};

struct CpuTopology {
  std::vector<CpuCore> cores;    // online cores
  int performance_count = 0;
  std::string basis;             // how the core classes were told apart
};

/// Read the online cores and classify them
CpuTopology discoverTopology();

/// One line for the log
std::string describeTopology(const CpuTopology &t);

/**
 * Bind the calling process to cores according to its role. Returns a log
 * line describing what was done (or why nothing was).
 * @param policy Execution/cpuBinding
 * @param local_rank rank among the ranks on this node
 * @param local_size number of ranks on this node
 * @param accelerator_local local ranks that drive a GPU on this node
 */
std::string applyBinding(const CpuTopology &t, CpuBinding policy, int local_rank,
                         int local_size, const std::vector<int> &accelerator_local);

/// OS-reported available memory (MemAvailable), bytes; 0 if unknown
double hostAvailableBytes();

/// Resident memory of this process, bytes
double processResidentBytes();

}  // namespace batchpf
}  // namespace gridpack

#endif
