/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   backend_cpu.cpp
 * @date   2026-10-05
 *
 * @brief CPU reference backend: KLU on each batch member (guide 5.3.5).
 *
 * Every member gets a fresh KLU factorization with partial pivoting, the
 * same solver GridPACK's CPU path uses through PETSc. That makes this
 * backend the numerical reference for the GPU backends, which reuse one
 * fixed pivot order, and lets the whole batch pipeline run on machines
 * without a GPU (tests, CI). Members are split across a few worker tasks;
 * each task has its own KLU common block, so no data is shared between
 * them.
 */

#include <klu.h>

#include <algorithm>
#include <cmath>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "backend.hpp"

namespace gridpack {
namespace batchpf {

namespace {

class CpuReferenceBackend : public SolverBackend {
 public:
  explicit CpuReferenceBackend(const BackendSetup &setup)
      : p_n(setup.pattern->n_rows), p_B(setup.capacity),
        p_csc(toCsc(*setup.pattern)), p_numeric(setup.capacity, nullptr),
        p_commons(setup.capacity)
  {
    klu_defaults(&p_common);
    p_common.btf = 0;
    p_common.ordering = 0;
    p_symbolic = klu_analyze(p_n, p_csc.col_ptr.data(), p_csc.row_idx.data(),
                             &p_common);
    if (!p_symbolic) throw Error(BATCHPF_ERR_BACKEND, "KLU analysis failed");
    for (klu_common &c : p_commons) {
      klu_defaults(&c);
      c.btf = 0;
      c.ordering = 0;
    }
    const unsigned hw = std::thread::hardware_concurrency();
    p_tasks = std::max(1, std::min<int>(static_cast<int>(hw > 2 ? hw / 2 : 1), 8));
  }

  CpuReferenceBackend(const CpuReferenceBackend &) = delete;
  CpuReferenceBackend &operator=(const CpuReferenceBackend &) = delete;

  ~CpuReferenceBackend() override
  {
    for (int b = 0; b < p_B; b++) {
      if (p_numeric[b]) klu_free_numeric(&p_numeric[b], &p_commons[b]);
    }
    if (p_symbolic) klu_free_symbolic(&p_symbolic, &p_common);
  }

  BackendCaps caps() const override
  {
    BackendCaps c;
    c.name = "cpu_reference";
    c.version = "KLU " + std::to_string(KLU_MAIN_VERSION) + "." +
                std::to_string(KLU_SUB_VERSION) + "." +
                std::to_string(KLU_SUBSUB_VERSION);
    c.on_device = false;
    return c;
  }

  void refactorize(const double *values, const int *mask,
                   int *member_status) override
  {
    forMembers(mask, [&](int b) {
      const int64_t nnz = static_cast<int64_t>(p_csc.row_idx.size());
      std::vector<double> ax(static_cast<std::size_t>(nnz));
      for (int64_t p = 0; p < nnz; p++) {
        ax[p_csc.csr_to_csc[p]] = values[p * p_B + b];
      }
      if (p_numeric[b]) klu_free_numeric(&p_numeric[b], &p_commons[b]);
      p_numeric[b] = klu_factor(p_csc.col_ptr.data(), p_csc.row_idx.data(),
                                ax.data(), p_symbolic, &p_commons[b]);
      member_status[b] = (p_numeric[b] && p_commons[b].status == KLU_OK)
                             ? BATCHPF_MEMBER_OK : BATCHPF_MEMBER_SMALL_PIVOT;
    });
  }

  void solve(const double *rhs, double *x, const int *mask,
             int *member_status) override
  {
    forMembers(mask, [&](int b) {
      if (member_status[b] != BATCHPF_MEMBER_OK || !p_numeric[b]) return;
      std::vector<double> y(p_n);
      for (int i = 0; i < p_n; i++) y[i] = rhs[static_cast<int64_t>(i) * p_B + b];
      klu_solve(p_symbolic, p_numeric[b], p_n, 1, y.data(), &p_commons[b]);
      bool finite = true;
      for (int i = 0; i < p_n; i++) {
        finite = finite && std::isfinite(y[i]);
        x[static_cast<int64_t>(i) * p_B + b] = y[i];
      }
      if (!finite) member_status[b] = BATCHPF_MEMBER_NONFINITE;
    });
  }

 private:
  /// Run f(b) for every active member, split across a few tasks
  template <class F>
  void forMembers(const int *mask, const F &f)
  {
    std::vector<int> active;
    for (int b = 0; b < p_B; b++) {
      if (mask[b]) active.push_back(b);
    }
    if (active.empty()) return;
    const int tasks = std::min<int>(p_tasks, static_cast<int>(active.size()));
    std::vector<std::future<void>> work;
    for (int t = 1; t < tasks; t++) {
      work.push_back(std::async(std::launch::async, [&, t] {
        for (std::size_t i = t; i < active.size(); i += tasks) f(active[i]);
      }));
    }
    for (std::size_t i = 0; i < active.size(); i += tasks) f(active[i]);
    for (std::future<void> &w : work) w.get();
  }

  int p_n;
  int p_B;
  int p_tasks = 1;
  CscPattern p_csc;
  klu_common p_common;
  klu_symbolic *p_symbolic = nullptr;
  std::vector<klu_numeric *> p_numeric;   // owned; freed in the destructor
  std::vector<klu_common> p_commons;      // one per member: no sharing
};

}  // namespace

std::unique_ptr<SolverBackend> makeCpuReferenceBackend(const BackendSetup &setup)
{
  return std::make_unique<CpuReferenceBackend>(setup);
}

}  // namespace batchpf
}  // namespace gridpack
