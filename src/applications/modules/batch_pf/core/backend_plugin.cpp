/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   backend_plugin.cpp
 * @date   2026-10-05
 *
 * @brief Adapter that loads a backend plugin (I-11) and presents it as a
 * SolverBackend.
 *
 * The plugin is a separate shared object so a backend that needs extra
 * libraries (cuDSS) can be missing without affecting anything else. The
 * C table it returns is wrapped here: status codes become exceptions, and
 * the library handle and backend object are owned by RAII members.
 */

#include <dlfcn.h>

#include <cstring>
#include <array>
#include <gsl/span>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backend.hpp"
#include "common.hpp"
#include "planner.hpp"
#include "gridpack/batchpf/batchpf_backend.h"
#include "gridpack/batchpf/batchpf_plugin.h"

namespace gridpack {
namespace batchpf {

namespace {

struct LibraryClose {
  void operator()(void *h) const noexcept
  {
    if (h) dlclose(h);
  }
};
using LibraryHandle = std::unique_ptr<void, LibraryClose>;

/**
 * Open lib<prefix><name>.so in dir and read its function table. The cast
 * from the address dlsym returns to a function pointer is the one place
 * this conversion is needed (exception EX-CG-04, also in the host loader).
 */
bool openBackend(const std::string &dir, const std::string &name,
                 LibraryHandle *lib, batchpf_backend_api *api,
                 std::string *why)
{
  const std::string path = dir + "/libgridpack_batchpf_" + name + ".so";
  lib->reset(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL));
  if (!*lib) {
    const char *e = dlerror();
    *why = "cannot load " + path + ": " + (e ? e : "unknown error");
    return false;
  }
  void *sym = dlsym(lib->get(), "batchpf_get_backend_api");
  if (!sym) {
    *why = path + " does not export batchpf_get_backend_api";
    return false;
  }
  auto get_api = reinterpret_cast<batchpf_get_backend_api_fn>(sym);
  std::memset(api, 0, sizeof(*api));
  api->struct_size = sizeof(*api);
  const batchpf_status st = get_api(BATCHPF_BACKEND_API_MAJOR, api);
  if (st != BATCHPF_OK || api->api_major != BATCHPF_BACKEND_API_MAJOR) {
    *why = path + " implements backend interface version " +
           std::to_string(api->api_major) + ", this plugin needs " +
           std::to_string(BATCHPF_BACKEND_API_MAJOR);
    return false;
  }
  if (!api->capabilities || !api->setup || !api->refactorize || !api->solve ||
      !api->teardown || !api->last_error) {
    *why = path + " has an incomplete function table";
    return false;
  }
  return true;
}

BackendCaps toCaps(const batchpf_backend_caps &c)
{
  BackendCaps b;
  b.name = std::begin(c.name);
  b.version = std::begin(c.version);
  b.max_batch = c.max_batch;
  b.validated_batch = c.validated_batch;
  b.member_masking = c.member_masking != 0;
  b.iterative_refinement = c.iterative_refinement != 0;
  b.on_device = (c.memory_kinds & BATCHPF_MEMKIND_DEVICE) != 0;
  return b;
}

/// Releases a backend object through its own table
struct Teardown {
  const batchpf_backend_api *api = nullptr;
  void operator()(batchpf_backend *b) const noexcept
  {
    if (b && api) api->teardown(b);
  }
};

class PluginBackend : public SolverBackend {
 public:
  PluginBackend(LibraryHandle lib, const batchpf_backend_api &api,
                const BackendSetup &setup)
      : p_lib(std::move(lib)), p_api(api),
        p_capacity(setup.capacity),
        p_value_count(static_cast<std::size_t>(setup.pattern->nnz) * setup.capacity),
        p_vector_count(static_cast<std::size_t>(setup.pattern->n_rows) * setup.capacity)
  {
    batchpf_backend_caps c;
    std::memset(&c, 0, sizeof(c));
    c.struct_size = sizeof(c);
    check(p_api.capabilities(&c), "capabilities");
    p_caps = toCaps(c);
    batchpf_backend_plan plan{};
    plan.struct_size = sizeof(plan);
    plan.struct_version = 1;
    plan.n = setup.pattern->n_rows;
    plan.batch_capacity = setup.capacity;
    plan.nnz = setup.pattern->nnz;
    plan.row_ptr = setup.pattern->row_ptr.data();
    plan.col_idx = setup.pattern->col_idx.data();
    plan.row_perm = setup.lu->P.data();
    plan.col_perm = setup.lu->Q.data();
    plan.reference_values = setup.reference_values->data();
    plan.memory_kind = BATCHPF_MEMKIND_DEVICE;
    plan.device = setup.device;
    plan.stream = setup.stream;
    plan.refinement_steps = setup.refinement_steps;
    plan.threads_per_block = setup.threads_per_block;
    plan.pivot_limit = setup.pivot_limit;
    std::array<char, 512> err{};
    batchpf_backend *raw = nullptr;
    const batchpf_status st = p_api.setup(&plan, &raw, err.data(), err.size());
    if (st != BATCHPF_OK || raw == nullptr) {
      throw Error(st == BATCHPF_OK ? BATCHPF_ERR_BACKEND : st,
                  std::string(p_caps.name) + " setup failed: " + err.data());
    }
    p_backend = std::unique_ptr<batchpf_backend, Teardown>(raw, Teardown{&p_api});
  }

  BackendCaps caps() const override { return p_caps; }

  void refactorize(gsl::span<const double> values, gsl::span<const int> mask,
                   gsl::span<int> member_status) override
  {
    Expects(values.size() == p_value_count);
    Expects(mask.size() == p_capacity);
    Expects(member_status.size() == mask.size());
    check(p_api.refactorize(p_backend.get(), values.data(), mask.data(), member_status.data()),
          "refactorize");
  }

  void solve(gsl::span<const double> rhs, gsl::span<double> x,
             gsl::span<const int> mask, gsl::span<int> member_status) override
  {
    Expects(rhs.size() == p_vector_count);
    Expects(x.size() == rhs.size());
    Expects(mask.size() == p_capacity);
    Expects(member_status.size() == mask.size());
    check(p_api.solve(p_backend.get(), rhs.data(), x.data(), mask.data(), member_status.data()), "solve");
  }

 private:
  void check(batchpf_status st, const char *what) const
  {
    if (st == BATCHPF_OK) return;
    const char *msg = p_backend ? p_api.last_error(p_backend.get()) : nullptr;
    throw Error(st, p_caps.name + " " + what + ": " + (msg ? msg : "failed"));
  }

  LibraryHandle p_lib;                 // declared first: closed last
  batchpf_backend_api p_api;
  BackendCaps p_caps;
  std::unique_ptr<batchpf_backend, Teardown> p_backend;
  std::size_t p_capacity;
  std::size_t p_value_count;
  std::size_t p_vector_count;
};

}  // namespace

std::unique_ptr<SolverBackend> loadPluginBackend(const std::string &dir,
                                                 const std::string &name,
                                                 const BackendSetup &setup,
                                                 std::string *why)
{
  LibraryHandle lib;
  batchpf_backend_api api;
  if (!openBackend(dir, name, &lib, &api, why)) return nullptr;
  return std::make_unique<PluginBackend>(std::move(lib), api, setup);
}

bool probePluginBackend(const std::string &dir, const std::string &name,
                        BackendCaps *caps, std::string *why)
{
  LibraryHandle lib;
  batchpf_backend_api api;
  if (!openBackend(dir, name, &lib, &api, why)) return false;
  batchpf_backend_caps c;
  std::memset(&c, 0, sizeof(c));
  c.struct_size = sizeof(c);
  if (api.capabilities(&c) != BATCHPF_OK) {
    *why = name + " capability query failed";
    return false;
  }
  *caps = toCaps(c);
  return true;
}

}  // namespace batchpf
}  // namespace gridpack
