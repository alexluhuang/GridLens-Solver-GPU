/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   planner.cpp
 * @date   2026-10-05
 *
 * @brief One-time planning of the batched factorization (block B6). See
 * planner.hpp for what is computed and why.
 */

#include "planner.hpp"

#include <klu.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "common.hpp"

namespace gridpack {
namespace batchpf {

JacobianPattern buildPattern(int n_bus, const std::vector<int> &row_start,
                             const std::vector<int> &edge_col)
{
  JacobianPattern p;
  const int n_edge = static_cast<int>(edge_col.size());
  p.n_bus = n_bus;
  p.n_rows = 2 * n_bus;
  p.row_ptr.assign(static_cast<std::size_t>(p.n_rows) + 1, 0);
  p.diag_pos.assign(4 * static_cast<std::size_t>(n_bus), -1);
  p.edge_pos.assign(4 * static_cast<std::size_t>(n_edge), -1);
  p.nnz = 4 * (static_cast<int64_t>(n_bus) + n_edge);
  p.col_idx.assign(static_cast<std::size_t>(p.nnz), -1);

  // Each bus row lists its own block and one block per neighbor, by
  // increasing bus number; the P and Q rows share the same columns.
  std::vector<std::pair<int, int>> cols;   // (column bus, edge or -1)
  for (int k = 0; k < n_bus; k++) {
    cols.clear();
    cols.emplace_back(k, -1);
    for (int e = row_start[k]; e < row_start[k + 1]; e++) {
      cols.emplace_back(edge_col[e], e);
    }
    std::sort(cols.begin(), cols.end());
    const int width = 2 * static_cast<int>(cols.size());
    const int base0 = p.row_ptr[2 * k];
    const int base1 = base0 + width;
    p.row_ptr[2 * k + 1] = base1;
    p.row_ptr[2 * k + 2] = base1 + width;
    for (int idx = 0; idx < static_cast<int>(cols.size()); idx++) {
      const int m = cols[idx].first;
      const int e = cols[idx].second;
      p.col_idx[base0 + 2 * idx] = 2 * m;
      p.col_idx[base0 + 2 * idx + 1] = 2 * m + 1;
      p.col_idx[base1 + 2 * idx] = 2 * m;
      p.col_idx[base1 + 2 * idx + 1] = 2 * m + 1;
      int *pos = (e < 0) ? &p.diag_pos[4 * static_cast<std::size_t>(k)]
                         : &p.edge_pos[4 * static_cast<std::size_t>(e)];
      pos[0] = base0 + 2 * idx;       // (P, theta)
      pos[1] = base1 + 2 * idx;       // (Q, theta)
      pos[2] = base0 + 2 * idx + 1;   // (P, V)
      pos[3] = base1 + 2 * idx + 1;   // (Q, V)
    }
  }
  return p;
}

int64_t minimalNnz(int n_bus, const std::vector<int> &row_start,
                   const std::vector<int> &edge_col,
                   const std::vector<int> &bus_type)
{
  auto dim = [&](int k) {
    const int t = bus_type[k];
    if (t == BATCHPF_BUS_REF || t == BATCHPF_BUS_ISOLATED) return 0;
    return (t == BATCHPF_BUS_PV) ? 1 : 2;
  };
  int64_t nnz = 0;
  for (int k = 0; k < n_bus; k++) {
    const int64_t dk = dim(k);
    nnz += dk * dk;
    for (int e = row_start[k]; e < row_start[k + 1]; e++) {
      nnz += dk * dim(edge_col[e]);
    }
  }
  return nnz;
}

CscPattern toCsc(const JacobianPattern &pattern)
{
  CscPattern c;
  const int n = pattern.n_rows;
  c.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
  c.row_idx.assign(static_cast<std::size_t>(pattern.nnz), 0);
  c.csr_to_csc.assign(static_cast<std::size_t>(pattern.nnz), 0);
  for (int64_t p = 0; p < pattern.nnz; p++) c.col_ptr[pattern.col_idx[p] + 1]++;
  for (int j = 0; j < n; j++) c.col_ptr[j + 1] += c.col_ptr[j];
  std::vector<int> next(c.col_ptr.begin(), c.col_ptr.end() - 1);
  for (int i = 0; i < n; i++) {
    for (int p = pattern.row_ptr[i]; p < pattern.row_ptr[i + 1]; p++) {
      const int q = next[pattern.col_idx[p]]++;
      c.row_idx[q] = i;
      c.csr_to_csc[p] = q;
    }
  }
  return c;
}

namespace {

/// Owner of KLU's symbolic and numeric objects (they need the common block
/// to be freed)
struct KluObjects {
  klu_common common;
  klu_symbolic *symbolic = nullptr;
  klu_numeric *numeric = nullptr;
  KluObjects() { klu_defaults(&common); }
  KluObjects(const KluObjects &) = delete;
  KluObjects &operator=(const KluObjects &) = delete;
  ~KluObjects()
  {
    if (numeric) klu_free_numeric(&numeric, &common);
    if (symbolic) klu_free_symbolic(&symbolic, &common);
  }
};

/// Assign each index the level 1 + max(level of its dependencies) and
/// bucket indices by level
void bucketLevels(const std::vector<int> &level, std::vector<int> *ptr,
                  std::vector<int> *items)
{
  const int nlev = level.empty()
      ? 0 : 1 + *std::max_element(level.begin(), level.end());
  ptr->assign(static_cast<std::size_t>(nlev) + 1, 0);
  for (int l : level) (*ptr)[l + 1]++;
  for (int l = 0; l < nlev; l++) (*ptr)[l + 1] += (*ptr)[l];
  items->assign(level.size(), 0);
  std::vector<int> next(ptr->begin(), ptr->end() - 1);
  for (int i = 0; i < static_cast<int>(level.size()); i++) {
    (*items)[next[level[i]]++] = i;
  }
}

}  // namespace

LuPlan planLu(const JacobianPattern &pattern,
              const std::vector<double> &values, int ordering,
              double pivot_tolerance)
{
  const int n = pattern.n_rows;
  CscPattern csc = toCsc(pattern);
  std::vector<double> ax(static_cast<std::size_t>(pattern.nnz), 0.0);
  for (int64_t p = 0; p < pattern.nnz; p++) ax[csc.csr_to_csc[p]] = values[p];

  // KLU with one block (no BTF), the requested ordering, partial pivoting
  KluObjects klu;
  klu.common.btf = 0;
  klu.common.ordering = ordering;
  klu.common.tol = pivot_tolerance;
  klu.symbolic = klu_analyze(n, csc.col_ptr.data(), csc.row_idx.data(),
                             &klu.common);
  if (!klu.symbolic) {
    throw Error(BATCHPF_ERR_BACKEND, "KLU analysis of the superset Jacobian "
                "failed (status " + std::to_string(klu.common.status) + ")");
  }
  klu.numeric = klu_factor(csc.col_ptr.data(), csc.row_idx.data(), ax.data(),
                           klu.symbolic, &klu.common);
  if (!klu.numeric || klu.common.status != KLU_OK) {
    throw Error(BATCHPF_ERR_BACKEND, "KLU reference factorization of the "
                "base-case superset Jacobian failed (status " +
                std::to_string(klu.common.status) + "); the base case may be "
                "singular in the superset form");
  }
  const int lnz = static_cast<int>(klu.numeric->lnz);
  const int unz = static_cast<int>(klu.numeric->unz);
  std::vector<int> Lp(n + 1), Li(lnz), Up(n + 1), Ui(unz);
  std::vector<double> Lx(lnz), Ux(unz);
  LuPlan plan;
  plan.n = n;
  plan.P.assign(n, 0);
  plan.Q.assign(n, 0);
  if (!klu_extract(klu.numeric, klu.symbolic, Lp.data(), Li.data(), Lx.data(),
                   Up.data(), Ui.data(), Ux.data(), nullptr, nullptr, nullptr,
                   plan.P.data(), plan.Q.data(), nullptr, nullptr,
                   &klu.common)) {
    throw Error(BATCHPF_ERR_BACKEND, "klu_extract failed");
  }
  plan.Pinv.assign(n, 0);
  plan.Qinv.assign(n, 0);
  for (int i = 0; i < n; i++) {
    plan.Pinv[plan.P[i]] = i;
    plan.Qinv[plan.Q[i]] = i;
  }

  // Combined column storage: U(:,j) (rows <= j) then strict L(:,j), each
  // sorted by row, so a whole column is sorted by row.
  plan.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
  plan.diag.assign(n, -1);
  std::vector<int> rows;
  for (int j = 0; j < n; j++) {
    rows.clear();
    for (int p = Up[j]; p < Up[j + 1]; p++) rows.push_back(Ui[p]);
    for (int p = Lp[j]; p < Lp[j + 1]; p++) {
      if (Li[p] != j) rows.push_back(Li[p]);
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    plan.col_ptr[j + 1] = plan.col_ptr[j] + static_cast<int>(rows.size());
    plan.row_idx.insert(plan.row_idx.end(), rows.begin(), rows.end());
  }
  plan.nnz = plan.col_ptr[n];
  for (int j = 0; j < n; j++) {
    const auto b = plan.row_idx.begin() + plan.col_ptr[j];
    const auto e = plan.row_idx.begin() + plan.col_ptr[j + 1];
    const auto it = std::lower_bound(b, e, j);
    if (it == e || *it != j) {
      throw Error(BATCHPF_ERR_INTERNAL, "factor column without a diagonal");
    }
    plan.diag[j] = static_cast<int>(it - plan.row_idx.begin());
  }
  auto position = [&](int i, int j) {
    const auto b = plan.row_idx.begin() + plan.col_ptr[j];
    const auto e = plan.row_idx.begin() + plan.col_ptr[j + 1];
    const auto it = std::lower_bound(b, e, i);
    if (it == e || *it != i) return -1;
    return static_cast<int>(it - plan.row_idx.begin());
  };

  // Where every Jacobian entry goes, and the per-column scale for pivot
  // health checks
  plan.a_to_lu.assign(static_cast<std::size_t>(pattern.nnz), -1);
  plan.col_scale.assign(n, 0.0);
  for (int r = 0; r < n; r++) {
    for (int p = pattern.row_ptr[r]; p < pattern.row_ptr[r + 1]; p++) {
      const int i = plan.Pinv[r];
      const int j = plan.Qinv[pattern.col_idx[p]];
      const int pos = position(i, j);
      if (pos < 0) {
        throw Error(BATCHPF_ERR_INTERNAL, "Jacobian entry missing from factor");
      }
      plan.a_to_lu[p] = pos;
      plan.col_scale[j] = std::max(plan.col_scale[j], std::fabs(values[p]));
    }
  }

  // Left-looking update segments and factorization levels
  plan.seg_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
  std::vector<int> rowpos(n, -1);
  std::vector<int> level(n, 0);
  for (int j = 0; j < n; j++) {
    for (int p = plan.col_ptr[j]; p < plan.col_ptr[j + 1]; p++) {
      rowpos[plan.row_idx[p]] = p;
    }
    for (int p = plan.col_ptr[j]; p < plan.diag[j]; p++) {
      const int k = plan.row_idx[p];
      const int lbeg = plan.diag[k] + 1;
      const int len = plan.col_ptr[k + 1] - lbeg;
      plan.seg_upos.push_back(p);
      plan.seg_lbeg.push_back(lbeg);
      plan.seg_len.push_back(len);
      plan.seg_dst.push_back(static_cast<int>(plan.dst.size()));
      for (int t = 0; t < len; t++) {
        const int target = rowpos[plan.row_idx[lbeg + t]];
        if (target < 0) {
          throw Error(BATCHPF_ERR_INTERNAL, "fill entry missing from factor");
        }
        plan.dst.push_back(target);
      }
      plan.flops += len;
      level[j] = std::max(level[j], level[k] + 1);
    }
    plan.seg_ptr[j + 1] = static_cast<int>(plan.seg_upos.size());
    for (int p = plan.col_ptr[j]; p < plan.col_ptr[j + 1]; p++) {
      rowpos[plan.row_idx[p]] = -1;
    }
  }
  bucketLevels(level, &plan.lev_ptr, &plan.lev_cols);

  // Row lists of the strict triangles for the solves
  plan.lrow_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
  plan.urow_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
  for (int j = 0; j < n; j++) {
    for (int p = plan.col_ptr[j]; p < plan.col_ptr[j + 1]; p++) {
      const int i = plan.row_idx[p];
      if (i > j) plan.lrow_ptr[i + 1]++;
      if (i < j) plan.urow_ptr[i + 1]++;
    }
  }
  for (int i = 0; i < n; i++) {
    plan.lrow_ptr[i + 1] += plan.lrow_ptr[i];
    plan.urow_ptr[i + 1] += plan.urow_ptr[i];
  }
  plan.lrow_col.assign(plan.lrow_ptr[n], 0);
  plan.lrow_pos.assign(plan.lrow_ptr[n], 0);
  plan.urow_col.assign(plan.urow_ptr[n], 0);
  plan.urow_pos.assign(plan.urow_ptr[n], 0);
  std::vector<int> lnext(plan.lrow_ptr.begin(), plan.lrow_ptr.end() - 1);
  std::vector<int> unext(plan.urow_ptr.begin(), plan.urow_ptr.end() - 1);
  for (int j = 0; j < n; j++) {
    for (int p = plan.col_ptr[j]; p < plan.col_ptr[j + 1]; p++) {
      const int i = plan.row_idx[p];
      if (i > j) {
        plan.lrow_col[lnext[i]] = j;
        plan.lrow_pos[lnext[i]++] = p;
      } else if (i < j) {
        plan.urow_col[unext[i]] = j;
        plan.urow_pos[unext[i]++] = p;
      }
    }
  }
  std::vector<int> llev(n, 0), ulev(n, 0);
  for (int i = 0; i < n; i++) {
    for (int p = plan.lrow_ptr[i]; p < plan.lrow_ptr[i + 1]; p++) {
      llev[i] = std::max(llev[i], llev[plan.lrow_col[p]] + 1);
    }
  }
  for (int i = n - 1; i >= 0; i--) {
    for (int p = plan.urow_ptr[i]; p < plan.urow_ptr[i + 1]; p++) {
      ulev[i] = std::max(ulev[i], ulev[plan.urow_col[p]] + 1);
    }
  }
  bucketLevels(llev, &plan.llev_ptr, &plan.llev_rows);
  bucketLevels(ulev, &plan.ulev_ptr, &plan.ulev_rows);
  return plan;
}

}  // namespace batchpf
}  // namespace gridpack
