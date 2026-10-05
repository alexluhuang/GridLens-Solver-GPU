/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   profiler.hpp
 * @date   2026-10-05
 *
 * @brief Named ranges for Nsight Systems (GPUBatch/profilerRanges).
 *
 * NVTX is header-only in CUDA 12 and later, so this adds no library
 * dependency. A range costs nothing when profilerRanges is false.
 */

#ifndef GRIDPACK_BATCHPF_CORE_PROFILER_HPP
#define GRIDPACK_BATCHPF_CORE_PROFILER_HPP

#include <nvtx3/nvToolsExt.h>

namespace gridpack {
namespace batchpf {

/// Push a named range for the lifetime of the object (RAII)
class ProfilerRange {
 public:
  ProfilerRange(bool enabled, const char *name) : p_enabled(enabled)
  {
    if (p_enabled) nvtxRangePushA(name);
  }
  ProfilerRange(const ProfilerRange &) = delete;
  ProfilerRange &operator=(const ProfilerRange &) = delete;
  ~ProfilerRange()
  {
    if (p_enabled) nvtxRangePop();
  }

 private:
  bool p_enabled;
};

}  // namespace batchpf
}  // namespace gridpack

#endif
