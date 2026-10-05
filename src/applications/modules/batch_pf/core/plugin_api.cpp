/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   plugin_api.cpp
 * @date   2026-10-05
 *
 * @brief C entry points of the core plugin (I-10).
 *
 * This is the only file that sees both sides of the boundary. Each entry
 * point checks its arguments, calls into the C++ session, and turns any
 * exception into a status code with a message, so no exception ever
 * reaches ca.x (ADR-19). batchpf_get_api() is the only exported symbol; the
 * build hides everything else (CUDA Best Practices Guide 16.4.1.4).
 */

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>

#include "common.hpp"
#include "gridpack/batchpf/batchpf_plugin.h"
#include "platform.hpp"
#include "session.hpp"

using gridpack::batchpf::Error;
using gridpack::batchpf::Session;

/// Definition of the opaque C handle
struct batchpf_session {
  std::unique_ptr<Session> impl;
  std::string error;
};

namespace {

const char *const kPluginVersion = "gridpack_batchpf_core 1.0.0";

std::string buildInfo()
{
  constexpr int rt = CUDART_VERSION;
  return "CUDA runtime " + std::to_string(rt / 1000) + "." +
         std::to_string((rt % 1000) / 10) + " (static)";
}

/// Run f, converting exceptions to a status and storing the message
template <class F>
batchpf_status guarded(batchpf_session *s, char *err, size_t err_size, F &&f)
{
  try {
    std::forward<F>(f)();
    return BATCHPF_OK;
  } catch (const Error &e) {
    if (s) s->error = e.what();
    gridpack::batchpf::copyMessage(e.what(), err, err_size);
    return e.code();
  } catch (const std::bad_alloc &) {
    if (s) s->error = "out of host memory";
    gridpack::batchpf::copyMessage("out of host memory", err, err_size);
    return BATCHPF_ERR_OUT_OF_MEMORY;
  } catch (const std::exception &e) {
    if (s) s->error = e.what();
    gridpack::batchpf::copyMessage(e.what(), err, err_size);
    return BATCHPF_ERR_INTERNAL;
  } catch (...) {
    if (s) s->error = "unknown error";
    gridpack::batchpf::copyMessage("unknown error", err, err_size);
    return BATCHPF_ERR_INTERNAL;
  }
}

batchpf_status probe(int32_t device, batchpf_device_info *info, char *error,
                     size_t error_size)
{
  if (!info || info->struct_size < sizeof(batchpf_device_info)) {
    return BATCHPF_ERR_INVALID_ARGUMENT;
  }
  batchpf_status st = BATCHPF_OK;
  const batchpf_status g = guarded(nullptr, error, error_size, [&] {
    std::string why;
    st = gridpack::batchpf::probeDevice(device, info, &why);
    if (st != BATCHPF_OK) gridpack::batchpf::copyMessage(why, error, error_size);
  });
  return (g != BATCHPF_OK) ? g : st;
}

batchpf_status sessionCreate(const batchpf_settings *settings,
                             batchpf_session **session, char *error,
                             size_t error_size)
{
  if (!settings || !session || settings->struct_size < sizeof(batchpf_settings)) {
    gridpack::batchpf::copyMessage("invalid settings record", error, error_size);
    return BATCHPF_ERR_INVALID_ARGUMENT;
  }
  *session = nullptr;
  return guarded(nullptr, error, error_size, [&] {
    auto s = std::make_unique<batchpf_session>();
    s->impl = std::make_unique<Session>(*settings);
    *session = s.release();
  });
}

void sessionDestroy(batchpf_session *session)
{
  // The handle was created by sessionCreate; ownership returns here
  std::unique_ptr<batchpf_session> owned(session);
}

const char *lastError(const batchpf_session *session)
{
  if (!session) return "no session";
  return session->error.c_str();
}

batchpf_status getDeviceInfo(batchpf_session *s, batchpf_device_info *info)
{
  if (!s || !info || info->struct_size < sizeof(batchpf_device_info)) {
    return BATCHPF_ERR_INVALID_ARGUMENT;
  }
  return guarded(s, nullptr, 0, [&] {
    const uint32_t size = info->struct_size;
    *info = s->impl->deviceInfo();
    info->struct_size = size;
  });
}

batchpf_status setModel(batchpf_session *s, const batchpf_model *model)
{
  if (!s || !model) return BATCHPF_ERR_INVALID_ARGUMENT;
  return guarded(s, nullptr, 0, [&] { s->impl->setModel(*model); });
}

batchpf_status plan(batchpf_session *s, const batchpf_solver_params *params,
                    int64_t expected_cases, int32_t *capacity)
{
  if (!s || !params || !capacity) return BATCHPF_ERR_INVALID_ARGUMENT;
  return guarded(s, nullptr, 0, [&] {
    *capacity = s->impl->plan(*params, expected_cases);
  });
}

batchpf_status submit(batchpf_session *s, const batchpf_batch *batch,
                      batchpf_results *results, int64_t *ticket)
{
  if (!s || !batch || !results || !ticket) return BATCHPF_ERR_INVALID_ARGUMENT;
  return guarded(s, nullptr, 0, [&] {
    *ticket = s->impl->submit(*batch, *results);
  });
}

batchpf_status wait(batchpf_session *s, int64_t ticket, int32_t timeout_ms)
{
  if (!s) return BATCHPF_ERR_INVALID_ARGUMENT;
  batchpf_status st = BATCHPF_OK;
  const batchpf_status g = guarded(s, nullptr, 0, [&] {
    st = s->impl->wait(ticket, timeout_ms);
    if (st != BATCHPF_OK && st != BATCHPF_ERR_TIMEOUT) {
      s->error = s->impl->lastError();
    }
  });
  return (g != BATCHPF_OK) ? g : st;
}

batchpf_status getDiagnostics(batchpf_session *s, batchpf_diagnostics *d)
{
  if (!s || !d || d->struct_size < sizeof(batchpf_diagnostics)) {
    return BATCHPF_ERR_INVALID_ARGUMENT;
  }
  return guarded(s, nullptr, 0, [&] {
    const uint32_t size = d->struct_size;
    *d = s->impl->diagnostics();
    d->struct_size = size;
  });
}

}  // namespace

extern "C" __attribute__((visibility("default")))
batchpf_status batchpf_get_api(uint32_t requested_major, batchpf_api *api)
{
  if (!api || api->struct_size < offsetof(batchpf_api, plugin_version)) {
    return BATCHPF_ERR_INVALID_ARGUMENT;
  }
  return guarded(nullptr, nullptr, 0, [&] {
    static const std::string info = buildInfo();
    batchpf_api full{};
    full.struct_size = sizeof(full);
    full.api_major = BATCHPF_API_MAJOR;
    full.api_minor = BATCHPF_API_MINOR;
    full.plugin_version = kPluginVersion;
    full.build_info = info.c_str();
    full.probe = probe;
    full.session_create = sessionCreate;
    full.session_destroy = sessionDestroy;
    full.last_error = lastError;
    full.get_device_info = getDeviceInfo;
    full.set_model = setModel;
    full.plan = plan;
    full.submit = submit;
    full.wait = wait;
    full.get_diagnostics = getDiagnostics;
    // Older callers see only the entries their table can hold.
    const uint32_t size = api->struct_size;
    std::memcpy(api, &full, std::min<std::size_t>(size, sizeof(full)));
    api->struct_size = std::min<uint32_t>(size, sizeof(full));
    if (requested_major != BATCHPF_API_MAJOR) throw Error(BATCHPF_ERR_VERSION, "API major mismatch");
  });
}
