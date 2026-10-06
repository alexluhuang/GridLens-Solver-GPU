/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   platform.cu
 * @date   2026-10-05
 *
 * @brief Device side of the platform layer (block B9-D). See platform.hpp.
 */

#include "platform.hpp"

#include <cudaTypedefs.h>

#include <algorithm>
#include <cstring>

namespace gridpack {
namespace batchpf {

namespace {

int attribute(cudaDeviceAttr attr, int device)
{
  int v = 0;
  const auto status = cudaDeviceGetAttribute(&v, attr, device);
  if (status == cudaErrorInvalidValue || status == cudaErrorNotSupported) {
    // Older drivers can lack optional attributes. Clear only this
    // expected failure; preserve unrelated asynchronous errors.
    const auto previous = cudaGetLastError();
    if (previous != cudaSuccess && previous != status) {
      cudaCheck(previous, "optional device attribute");
    }
    return -1;
  }
  cudaCheck(status, "cudaDeviceGetAttribute");
  return v;
}

int dmaBufSupported(int device)
{
  void *address = nullptr;
  cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSymbolNotFound;
  const auto status = cudaGetDriverEntryPointByVersion(
      "cuDeviceGetAttribute", &address, CUDA_VERSION, cudaEnableDefault, &query);
  if (status == cudaErrorInvalidValue || status == cudaErrorNotSupported) return -1;
  cudaCheck(status, "query driver attribute entry point");
  if (query != cudaDriverEntryPointSuccess || !address) return -1;
  // EX-CG-04: the runtime returns an untyped driver entry point; the
  // versioned CUDA typedef fixes its signature without linking libcuda.
  const auto getAttribute = reinterpret_cast<PFN_cuDeviceGetAttribute_v2000>(address);
  int supported = 0;
  const auto result = getAttribute(&supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, device);
  if (result == CUDA_ERROR_INVALID_VALUE || result == CUDA_ERROR_NOT_SUPPORTED) return -1;
  if (result != CUDA_SUCCESS) {
    throw Error(BATCHPF_ERR_CUDA, "DMA-BUF attribute query failed (driver status " +
                std::to_string(static_cast<int>(result)) + ")");
  }
  return supported;
}

__global__ void probeKernel(int *out) { *out = 1; }

}  // namespace

batchpf_status probeDevice(int device, batchpf_device_info *info,
                           std::string *why)
{
  const uint32_t size = info->struct_size;
  std::memset(info, 0, std::min<std::size_t>(size, sizeof(*info)));
  info->struct_size = size;
  info->struct_version = 1;
  int count = 0;
  const cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    if (why) {
      *why = (err != cudaSuccess)
          ? std::string("no usable GPU: ") + cudaGetErrorName(err) + " (" +
                cudaGetErrorString(err) + ")"
          : std::string("no GPU visible (check CUDA_VISIBLE_DEVICES and the "
                        "container's GPU access)");
    }
    cudaGetLastError();
    return BATCHPF_ERR_NO_DEVICE;
  }
  info->device_count = count;
  if (device < 0 || device >= count) {
    if (why) {
      *why = "GPUBatch/device " + std::to_string(device) + " is not one of the " +
             std::to_string(count) + " visible devices";
    }
    return BATCHPF_ERR_INVALID_ARGUMENT;
  }
  info->device = device;
  cudaDeviceProp prop{};
  cudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  copyMessage(prop.name, info->name, sizeof(info->name));
  info->cc_major = attribute(cudaDevAttrComputeCapabilityMajor, device);
  info->cc_minor = attribute(cudaDevAttrComputeCapabilityMinor, device);
  info->integrated = attribute(cudaDevAttrIntegrated, device);
  info->concurrent_managed_access =
      attribute(cudaDevAttrConcurrentManagedAccess, device);
  info->pageable_memory_access =
      attribute(cudaDevAttrPageableMemoryAccess, device);
  info->pageable_uses_host_page_tables =
      attribute(cudaDevAttrPageableMemoryAccessUsesHostPageTables, device);
  info->host_native_atomics =
      attribute(cudaDevAttrHostNativeAtomicSupported, device);
  info->gpudirect_rdma = attribute(cudaDevAttrGPUDirectRDMASupported, device);
  info->dmabuf = dmaBufSupported(device);
  info->multiprocessors = attribute(cudaDevAttrMultiProcessorCount, device);
  cudaCheck(cudaDriverGetVersion(&info->driver_version), "cudaDriverGetVersion");
  cudaCheck(cudaRuntimeGetVersion(&info->runtime_version), "cudaRuntimeGetVersion");
  info->total_memory_bytes = static_cast<double>(prop.totalGlobalMem);
  cudaCheck(cudaSetDevice(device), "cudaSetDevice");
  std::size_t free_b = 0, total_b = 0;
  cudaCheck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
  info->free_memory_bytes = static_cast<double>(free_b);
  info->memory_profile = selectMemoryProfile(*info, BATCHPF_MEMORY_AUTO);
  return BATCHPF_OK;
}

int selectMemoryProfile(const batchpf_device_info &info, int requested)
{
  if (requested != BATCHPF_MEMORY_AUTO) return requested;
  if (info.integrated == 1) return BATCHPF_MEMORY_UNIFIED;
  if (info.pageable_memory_access == 1 &&
      info.pageable_uses_host_page_tables == 1) {
    return BATCHPF_MEMORY_COHERENT;
  }
  return BATCHPF_MEMORY_DISCRETE;
}

const char *memoryProfileName(int profile)
{
  switch (profile) {
    case BATCHPF_MEMORY_UNIFIED: return "unified";
    case BATCHPF_MEMORY_COHERENT: return "coherent";
    case BATCHPF_MEMORY_DISCRETE: return "discrete";
    default: return "auto";
  }
}

double memoryBudget(const batchpf_device_info &info, int profile,
                    const batchpf_settings &s)
{
  // Automatic headroom: on unified memory 10% of the shared pool, protecting
  // the OS from the host stalls reported under memory pressure; elsewhere 5%
  // of GPU memory
  double headroom = s.memory_headroom_bytes;
  if (headroom < 0.0) {
    headroom = (profile == BATCHPF_MEMORY_UNIFIED)
                   ? 0.10 * info.total_memory_bytes
                   : 0.05 * info.total_memory_bytes;
  }
  double budget = 0.0;
  if (profile == BATCHPF_MEMORY_UNIFIED && s.host_available_bytes > 0.0) {
    budget = s.host_available_bytes - headroom - s.host_reserved_bytes;
  } else {
    budget = info.free_memory_bytes - headroom;
  }
  if (s.max_memory_bytes > 0.0) budget = std::min(budget, s.max_memory_bytes);
  return std::max(budget, 0.0);
}

void verifyKernelsLoad()
{
  cudaFuncAttributes attr{};
  const cudaError_t err = cudaFuncGetAttributes(&attr, probeKernel);
  if (err != cudaSuccess) {
    cudaGetLastError();
    throw Error(BATCHPF_ERR_NO_DEVICE,
                std::string("the plugin has no code for this GPU (") +
                    cudaGetErrorName(err) + "); rebuild with its architecture "
                    "in CMAKE_CUDA_ARCHITECTURES");
  }
  Buffer<int> flag(MemoryKind::Device, 1);
  flag.zero(nullptr);
  probeKernel<<<1, 1>>>(flag.data());
  launchCheck("probeKernel");
  int host = 0;
  cudaCheck(cudaMemcpy(&host, flag.data(), sizeof(int), cudaMemcpyDeviceToHost),
            "probe copy");
  if (host != 1) throw Error(BATCHPF_ERR_CUDA, "probe kernel did not run");
}

}  // namespace batchpf
}  // namespace gridpack
