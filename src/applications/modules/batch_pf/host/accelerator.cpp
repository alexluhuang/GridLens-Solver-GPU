/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   accelerator.cpp
 * @date   2026-10-05
 *
 * @brief Accelerator loader (block B13). See accelerator.hpp.
 */

#include "accelerator.hpp"

#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <memory>

namespace gridpack {
namespace batchpf {

namespace {

const char *const kCoreLibrary = "libgridpack_batchpf_core.so";

const char *levelName(int level)
{
  switch (level) {
    case BATCHPF_LOG_ERROR: return "ERROR";
    case BATCHPF_LOG_WARN: return "WARNING";
    case BATCHPF_LOG_DEBUG: return "debug";
    default: return "info";
  }
}

struct DlClose {
  void operator()(void *h) const noexcept
  {
    if (h) dlclose(h);
  }
};

/**
 * Open the core plugin in dir and fill its function table. The conversion
 * of dlsym's untyped address to a function pointer is unavoidable with
 * POSIX dynamic loading; it is done once, here (exception EX-CG-04).
 */
bool loadTable(const std::string &dir, std::unique_ptr<void, DlClose> *handle,
               batchpf_api *api, std::string *file, std::string *why)
{
  *file = dir + "/" + kCoreLibrary;
  if (access(file->c_str(), R_OK) != 0) {
    *why = file->c_str();
    *why += " not found";
    return false;
  }
  handle->reset(dlopen(file->c_str(), RTLD_NOW | RTLD_LOCAL));
  if (!*handle) {
    const char *e = dlerror();
    *why = "cannot load " + *file + ": " + (e ? e : "unknown error");
    return false;
  }
  void *sym = dlsym(handle->get(), "batchpf_get_api");
  if (!sym) {
    *why = *file + " has no batchpf_get_api entry point";
    return false;
  }
  auto get_api = reinterpret_cast<batchpf_get_api_fn>(sym);
  std::memset(api, 0, sizeof(*api));
  api->struct_size = sizeof(*api);
  const batchpf_status st = get_api(BATCHPF_API_MAJOR, api);
  if (st != BATCHPF_OK || api->api_major != BATCHPF_API_MAJOR) {
    *why = *file + " implements plugin interface " + std::to_string(api->api_major) +
           "." + std::to_string(api->api_minor) + "; ca.x needs " +
           std::to_string(BATCHPF_API_MAJOR) + ".x";
    return false;
  }
  if (!api->probe || !api->session_create || !api->session_destroy ||
      !api->last_error || !api->get_device_info || !api->set_model ||
      !api->plan || !api->submit || !api->wait || !api->get_diagnostics) {
    *why = *file + " has an incomplete function table";
    return false;
  }
  return true;
}

}  // namespace

void HostLogger::log(int level, const std::string &msg)
{
  if (!enabled(level)) return;
  std::lock_guard<std::mutex> lock(p_mutex);
  std::printf("[gpu-batch p%d] %s%s%s\n", p_rank,
              (level <= BATCHPF_LOG_WARN) ? levelName(level) : "",
              (level <= BATCHPF_LOG_WARN) ? ": " : "", msg.c_str());
  std::fflush(stdout);
}

void HostLogger::callback(void *user, int32_t level, const char *message)
{
  if (user && message) static_cast<HostLogger *>(user)->log(level, message);
}

std::string executableDirectory()
{
  char buf[PATH_MAX + 1];
  const ssize_t n = readlink("/proc/self/exe", buf, PATH_MAX);
  if (n <= 0) return ".";
  buf[n] = '\0';
  std::string path(buf);
  const std::size_t slash = path.find_last_of('/');
  return (slash == std::string::npos) ? "." : path.substr(0, slash);
}

std::vector<std::string> pluginSearchPath(const std::string &configured,
                                          const std::string &exe_dir)
{
  if (!configured.empty()) return {configured};
  // installed layout (<prefix>/bin/ca.x), then GridPACK's build tree
  return {exe_dir + "/../lib/gridpack/plugins",
          exe_dir + "/../../lib/gridpack/plugins"};
}

struct Accelerator::Library {
  std::unique_ptr<void, DlClose> handle;
};

Accelerator::Accelerator() : p_lib(std::make_unique<Library>())
{
  std::memset(&p_api, 0, sizeof(p_api));
}

Accelerator::~Accelerator()
{
  if (p_session) p_api.session_destroy(p_session);
}

std::unique_ptr<Accelerator> Accelerator::load(const std::vector<std::string> &dirs,
                                               int device, std::string *why)
{
  std::unique_ptr<Accelerator> acc(new Accelerator);
  std::string tried;
  bool loaded = false;
  for (const std::string &dir : dirs) {
    std::string reason;
    if (loadTable(dir, &acc->p_lib->handle, &acc->p_api, &acc->p_file, &reason)) {
      acc->p_dir = dir;
      loaded = true;
      break;
    }
    tried += (tried.empty() ? "" : "; ") + reason;
  }
  if (!loaded) {
    *why = "GPU plugin not usable (" + tried + ")";
    return nullptr;
  }
  acc->p_version = acc->p_api.plugin_version ? acc->p_api.plugin_version : "?";
  acc->p_build = acc->p_api.build_info ? acc->p_api.build_info : "";
  std::memset(&acc->p_probe, 0, sizeof(acc->p_probe));
  acc->p_probe.struct_size = sizeof(acc->p_probe);
  char err[512] = {0};
  acc->p_probe_ok = acc->p_api.probe(device, &acc->p_probe, err, sizeof(err)) == BATCHPF_OK;
  acc->p_probe_msg = err;
  return acc;
}

bool Accelerator::createSession(batchpf_settings settings, std::string *why)
{
  settings.plugin_dir = p_dir.c_str();
  char err[1024] = {0};
  const batchpf_status st = p_api.session_create(&settings, &p_session, err, sizeof(err));
  if (st != BATCHPF_OK || !p_session) {
    p_session = nullptr;
    *why = std::string("accelerator unavailable: ") + err;
    return false;
  }
  return true;
}

int Accelerator::countDevices(const std::vector<std::string> &dirs, std::string *why)
{
  for (const std::string &dir : dirs) {
    std::unique_ptr<void, DlClose> handle;
    batchpf_api api;
    std::string file, reason;
    if (!loadTable(dir, &handle, &api, &file, &reason)) {
      *why = reason;
      continue;
    }
    batchpf_device_info info;
    std::memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    char err[512] = {0};
    if (api.probe(0, &info, err, sizeof(err)) != BATCHPF_OK) {
      *why = err;
      return 0;
    }
    return info.device_count;
  }
  return 0;
}

void Accelerator::check(batchpf_status st, const char *what)
{
  if (st == BATCHPF_OK) return;
  const char *msg = p_api.last_error(p_session);
  throw AcceleratorError(st, std::string(what) + ": " + (msg ? msg : "failed"));
}

batchpf_device_info Accelerator::deviceInfo()
{
  batchpf_device_info info;
  std::memset(&info, 0, sizeof(info));
  info.struct_size = sizeof(info);
  check(p_api.get_device_info(p_session, &info), "device info");
  return info;
}

void Accelerator::setModel(const batchpf_model &model)
{
  check(p_api.set_model(p_session, &model), "set model");
}

int Accelerator::plan(const batchpf_solver_params &params, int64_t expected_cases)
{
  int32_t cap = 0;
  check(p_api.plan(p_session, &params, expected_cases, &cap), "planning");
  return cap;
}

int64_t Accelerator::submit(const batchpf_batch &batch, batchpf_results &results)
{
  int64_t ticket = 0;
  check(p_api.submit(p_session, &batch, &results, &ticket), "submit batch");
  return ticket;
}

bool Accelerator::wait(int64_t ticket, int timeout_ms)
{
  const batchpf_status st = p_api.wait(p_session, ticket, timeout_ms);
  if (st == BATCHPF_ERR_TIMEOUT) return false;
  check(st, "batch");
  return true;
}

batchpf_diagnostics Accelerator::diagnostics()
{
  batchpf_diagnostics d;
  std::memset(&d, 0, sizeof(d));
  d.struct_size = sizeof(d);
  check(p_api.get_diagnostics(p_session, &d), "diagnostics");
  return d;
}

}  // namespace batchpf
}  // namespace gridpack
