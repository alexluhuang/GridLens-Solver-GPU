/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   platform.hpp
 * @date   2026-10-05
 *
 * @brief Device side of the platform layer (block B9-D): GPU capability
 * probe, memory profile and placement, memory budget, code-path check.
 *
 * Nothing about the memory system is assumed at build time (ADR-13). The
 * probe reads the attributes that tell integrated, hardware-coherent and
 * discrete systems apart (guide 8.8.1); the placement rules of guide 8.8.2
 * then follow from the selected profile:
 *
 *   profile     model/working buffers   exchange buffers
 *   unified     device allocation       pinned host, read by the GPU
 *   coherent    device allocation       pinned host, read by the GPU
 *   discrete    device allocation       device copy + explicit transfers
 */

#ifndef GRIDPACK_BATCHPF_CORE_PLATFORM_HPP
#define GRIDPACK_BATCHPF_CORE_PLATFORM_HPP

#include <string>

#include "common.hpp"
#include "gridpack/batchpf/batchpf_plugin.h"

namespace gridpack {
namespace batchpf {

/**
 * Probe device `device`. Starts with cudaGetDeviceCount(), which fails with
 * cudaErrorNoDevice or cudaErrorInsufficientDriver when no usable GPU is
 * present (CUDA Best Practices Guide 17.1); that case returns
 * BATCHPF_ERR_NO_DEVICE without throwing.
 */
batchpf_status probeDevice(int device, batchpf_device_info *info,
                           std::string *why);

/// Memory profile from the probe, unless the setting forces one
int selectMemoryProfile(const batchpf_device_info &info, int requested);

const char *memoryProfileName(int profile);

/**
 * Memory the plugin may use for batch buffers. On a unified-memory system
 * the GPU and the CPU share one pool, and cudaMemGetInfo() does not count
 * memory the OS could reclaim, so the budget starts from the OS-reported
 * available memory instead, and keeps the configured headroom (guide
 * 7.1.3, 8.7). On other profiles it is the free GPU memory minus headroom.
 */
double memoryBudget(const batchpf_device_info &info, int profile,
                    const batchpf_settings &settings);

/**
 * Check that the compiled kernels can run on the current device: either
 * native code for its architecture or PTX the driver can compile is in the
 * plugin (CUDA Best Practices Guide 17.3). Throws if not.
 */
void verifyKernelsLoad();

}  // namespace batchpf
}  // namespace gridpack

#endif
