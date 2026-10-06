/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   common.hpp
 * @date   2026-10-05
 *
 * @brief Shared helpers for the core plugin: error types, CUDA error
 * checking, owners for GPU resources, logging.
 *
 * Error strategy (ADR-19): code inside the plugin throws Error when it
 * cannot do its job; the C entry points catch everything and turn it into a
 * status code. Whether a case converges is never an error; it is a status
 * in the case outcome. Every CUDA call is checked, and every kernel launch
 * is followed by cudaGetLastError(), because launch errors otherwise only
 * show up at a later synchronization (CUDA Best Practices Guide 17.2).
 */

#ifndef GRIDPACK_BATCHPF_CORE_COMMON_HPP
#define GRIDPACK_BATCHPF_CORE_COMMON_HPP

#include <cuda_runtime.h>
#include <gsl/assert>
#include <gsl/narrow>
#include <gsl/span>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gridpack/batchpf/batchpf_plugin.h"

namespace gridpack {
namespace batchpf {

/// Failure of an operation, carrying the status code for the C boundary
class Error : public std::runtime_error {
 public:
  Error(batchpf_status code, const std::string &what)
      : std::runtime_error(what), p_code(code) {}
  batchpf_status code() const noexcept { return p_code; }

 private:
  batchpf_status p_code;
};

/// Throw an Error with BATCHPF_ERR_CUDA if a CUDA call failed
inline void cudaCheck(cudaError_t err, const char *what)
{
  if (err != cudaSuccess) {
    throw Error(err == cudaErrorMemoryAllocation ? BATCHPF_ERR_OUT_OF_MEMORY : BATCHPF_ERR_CUDA,
                std::string(what) + ": " +
                                      cudaGetErrorName(err) + " (" +
                                      cudaGetErrorString(err) + ")");
  }
}

/// Check for an error from the kernel launch just issued
inline void launchCheck(const char *kernel)
{
  cudaCheck(cudaGetLastError(), kernel);
}

// Destructors cannot throw while unwinding another failure. Still record
// release errors so delayed CUDA failures do not disappear (STD-2).
inline void cudaCleanup(cudaError_t status, const char *what) noexcept
{
  if (status == cudaSuccess) return;
  std::fputs("[gpu-batch] cleanup failure in ", stderr);
  std::fputs(what, stderr);
  std::fputs(": ", stderr);
  std::fputs(cudaGetErrorString(status), stderr);
  std::fputc('\n', stderr);
}

inline bool memoryPoolsSupported()
{
  int device = 0, supported = 0;
  cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
  const auto status = cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device);
  if (status == cudaErrorInvalidValue || status == cudaErrorNotSupported) return false;
  cudaCheck(status, "cudaDevAttrMemoryPoolsSupported");
  return supported == 1;
}

/// Logger forwarding to the callback given by ca.x
class Logger {
 public:
  Logger() = default;
  Logger(batchpf_log_fn fn, void *user, int32_t level)
      : p_fn(fn), p_user(user), p_level(level) {}
  bool enabled(int32_t level) const { return p_fn && level <= p_level; }
  void log(int32_t level, const std::string &msg) const
  {
    if (enabled(level)) p_fn(p_user, level, msg.c_str());
  }
  void error(const std::string &m) const { log(BATCHPF_LOG_ERROR, m); }
  void warn(const std::string &m) const { log(BATCHPF_LOG_WARN, m); }
  void info(const std::string &m) const { log(BATCHPF_LOG_INFO, m); }
  void debug(const std::string &m) const { log(BATCHPF_LOG_DEBUG, m); }

 private:
  batchpf_log_fn p_fn = nullptr;
  void *p_user = nullptr;
  int32_t p_level = BATCHPF_LOG_INFO;
};

/// Where a buffer lives (placement classes of guide section 8.8.2)
enum class MemoryKind {
  Host,        // ordinary host memory (CPU-only execution)
  Device,      // stream-ordered device pool (legacy allocation if unsupported)
  Pinned,      // page-locked host memory the GPU can read directly
};

namespace detail {
struct DeviceFree {
  cudaStream_t stream = nullptr;
  bool pooled = false;
  void operator()(void *p) const noexcept
  {
    cudaCleanup(pooled ? cudaFreeAsync(p, stream) : cudaFree(p), "device allocation");
  }
};
struct PinnedFree {
  void operator()(void *p) const noexcept { cudaCleanup(cudaFreeHost(p), "cudaFreeHost"); }
};
}  // namespace detail

/**
 * Owner of a typed array in one memory kind. Host arrays use std::vector,
 * device and pinned arrays are released by their deleters (RAII, no naked
 * frees). Copying is disabled; moving transfers ownership.
 */
template <class T>
class Buffer {
  // Elements are moved with memcpy-style copies between memory kinds
  static_assert(std::is_trivially_copyable<T>::value,
                "Buffer<T> needs a trivially copyable element type");

 public:
  Buffer() = default;
  Buffer(MemoryKind kind, std::size_t count, cudaStream_t stream = nullptr)
  {
    allocate(kind, count, stream);
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&) noexcept = default;
  Buffer &operator=(Buffer &&) noexcept = default;
  ~Buffer() = default;

  void allocate(MemoryKind kind, std::size_t count, cudaStream_t stream = nullptr)
  {
    p_kind = kind;
    p_count = count;
    p_host.clear();
    p_device.reset();
    p_pinned.reset();
    p_ptr = nullptr;
    const std::size_t allocation_bytes = (count > 0 ? count : 1) * sizeof(T);
    if (kind == MemoryKind::Host) {
      p_host.assign(count > 0 ? count : 1, T());
      p_ptr = p_host.data();
    } else if (kind == MemoryKind::Device) {
      void *raw = nullptr;
      const bool pooled = memoryPoolsSupported();
      cudaCheck(pooled ? cudaMallocAsync(&raw, allocation_bytes, stream) : cudaMalloc(&raw, allocation_bytes),
                "device allocation");
      p_device = std::unique_ptr<void, detail::DeviceFree>(raw, {stream, pooled});
      p_ptr = static_cast<T *>(raw);
      // Callers that omit a stream retain the old immediate-allocation
      // contract; production buffers allocate on their execution stream.
      if (pooled && stream == nullptr) {
        cudaCheck(cudaStreamSynchronize(stream), "default-stream allocation");
      }
    } else {
      void *raw = nullptr;
      cudaCheck(cudaMallocHost(&raw, allocation_bytes), "cudaMallocHost");
      p_pinned.reset(raw);
      p_ptr = static_cast<T *>(raw);
    }
  }

  T *data() noexcept { return p_ptr; }
  const T *data() const noexcept { return p_ptr; }
  std::size_t size() const noexcept { return p_count; }
  std::size_t bytes() const noexcept { return p_count * sizeof(T); }
  MemoryKind kind() const noexcept { return p_kind; }

  /// Copy count elements from host memory into this buffer
  void upload(const T *src, std::size_t count, cudaStream_t stream)
  {
    upload(gsl::span<const T>(src, gsl::narrow<gsl::index>(count)), stream);
  }

  void upload(gsl::span<const T> src, cudaStream_t stream)
  {
    Expects(src.size() <= p_count);
    const auto count = src.size();
    if (count == 0) return;
    if (p_kind == MemoryKind::Device) {
      cudaCheck(cudaMemcpyAsync(p_ptr, src.data(), count * sizeof(T),
                                cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync H2D");
    } else {
      std::copy(src.begin(), src.end(), p_ptr);
    }
  }

  /// Copy count elements from this buffer into host memory
  void download(T *dst, std::size_t count, cudaStream_t stream) const
  {
    Expects(count <= p_count);
    if (count == 0) return;
    if (p_kind == MemoryKind::Device) {
      cudaCheck(cudaMemcpyAsync(dst, p_ptr, count * sizeof(T),
                                cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync D2H");
    } else {
      std::copy(p_ptr, p_ptr + count, dst);
    }
  }

  /// Set every byte to zero
  void zero(cudaStream_t stream)
  {
    if (p_kind == MemoryKind::Device) {
      cudaCheck(cudaMemsetAsync(p_ptr, 0, bytes(), stream), "cudaMemsetAsync");
    } else {
      std::fill(p_ptr, p_ptr + p_count, T());
    }
  }

 private:
  MemoryKind p_kind = MemoryKind::Host;
  std::size_t p_count = 0;
  T *p_ptr = nullptr;
  std::vector<T> p_host;
  std::unique_ptr<void, detail::DeviceFree> p_device;
  std::unique_ptr<void, detail::PinnedFree> p_pinned;
};

/// Owner of a CUDA stream
class Stream {
 public:
  Stream() = default;
  explicit Stream(bool create)
  {
    if (create) {
      cudaStream_t s = nullptr;
      cudaCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),
                "cudaStreamCreate");
      p_stream.reset(s);
    }
  }
  cudaStream_t get() const noexcept { return p_stream.get(); }
  void synchronize() const
  {
    if (p_stream) cudaCheck(cudaStreamSynchronize(p_stream.get()),
                            "cudaStreamSynchronize");
  }

 private:
  struct Destroy {
    void operator()(cudaStream_t s) const noexcept { cudaCleanup(cudaStreamDestroy(s), "cudaStreamDestroy"); }
  };
  std::unique_ptr<CUstream_st, Destroy> p_stream;
};

/// Owner of a CUDA event used for phase timing
class Event {
 public:
  Event() = default;
  explicit Event(bool create)
  {
    if (create) {
      cudaEvent_t e = nullptr;
      cudaCheck(cudaEventCreate(&e), "cudaEventCreate");
      p_event.reset(e);
    }
  }
  cudaEvent_t get() const noexcept { return p_event.get(); }

 private:
  struct Destroy {
    void operator()(cudaEvent_t e) const noexcept { cudaCleanup(cudaEventDestroy(e), "cudaEventDestroy"); }
  };
  std::unique_ptr<CUevent_st, Destroy> p_event;
};

/// Copy a C string into a fixed buffer, always terminated
inline void copyMessage(const std::string &msg, char *dst, std::size_t size)
{
  if (dst == nullptr || size == 0) return;
  const std::size_t n = (msg.size() < size - 1) ? msg.size() : size - 1;
  std::copy(msg.begin(), msg.begin() + static_cast<std::ptrdiff_t>(n), dst);
  dst[n] = '\0';
}

}  // namespace batchpf
}  // namespace gridpack

#endif
