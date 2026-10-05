/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   planner.hpp
 * @date   2026-10-05
 *
 * @brief One-time planning of the batched sparse factorization (block B6).
 *
 * Every contingency case uses the same Jacobian sparsity pattern (the
 * superset pattern), so the expensive structural work is done once per
 * study, as in the hybrid method of D'Orto et al. (guide appendix A):
 *
 *  1. Build the superset pattern: a 2x2 block for every bus and for every
 *     bus pair joined by a branch (guide section 8.3.3).
 *  2. Run KLU with AMD ordering and partial pivoting on a reference
 *     Jacobian (the base case). This fixes the row and column order and the
 *     structure of the L and U factors (ADR-02).
 *  3. From the L and U structure, derive what the batched GPU kernels
 *     need: where each Jacobian entry lands in the factor storage, the list
 *     of updates each column receives, and dependency levels (columns or
 *     rows in one level can be processed at the same time; guide 8.4.2).
 *
 * Unknowns and equations are ordered per bus: column 2m is the angle of bus
 * m, column 2m+1 its voltage magnitude; row 2k is the P equation of bus k,
 * row 2k+1 its Q equation. A 2x2 block is stored in GridPACK's column-major
 * order: (P,theta), (Q,theta), (P,V), (Q,V).
 */

#ifndef GRIDPACK_BATCHPF_CORE_PLANNER_HPP
#define GRIDPACK_BATCHPF_CORE_PLANNER_HPP

#include <cstdint>
#include <vector>

namespace gridpack {
namespace batchpf {

/// Superset Jacobian pattern (CSR) and where each block lives in it
struct JacobianPattern {
  int n_bus = 0;
  int n_rows = 0;
  int64_t nnz = 0;
  std::vector<int> row_ptr;   // n_rows + 1
  std::vector<int> col_idx;   // nnz, sorted within each row
  std::vector<int> diag_pos;  // 4 per bus: positions of its diagonal block
  std::vector<int> edge_pos;  // 4 per edge: positions of its off-diagonal block
};

/**
 * Build the superset pattern from the admittance pattern
 * @param n_bus number of buses
 * @param row_start CSR row starts of the edge list (n_bus + 1)
 * @param edge_col column bus of each edge
 */
JacobianPattern buildPattern(int n_bus, const std::vector<int> &row_start,
                             const std::vector<int> &edge_col);

/**
 * Number of entries GridPACK's standard layout would have for the given
 * bus roles (no rows for the reference and isolated buses, one row for PV
 * buses). Used to report the cost of the superset form (guide 8.14).
 */
int64_t minimalNnz(int n_bus, const std::vector<int> &row_start,
                   const std::vector<int> &edge_col,
                   const std::vector<int> &bus_type);

/**
 * Fixed-order LU plan. With A' = A(P, Q), i.e. A'(i,j) = A(P[i], Q[j]), the
 * factors satisfy A' = L U with unit lower L. L and U are stored together
 * per column (CSC): rows <= j belong to U, rows > j to L.
 */
struct LuPlan {
  int n = 0;
  std::vector<int> P, Q;          // row and column order
  std::vector<int> Pinv, Qinv;
  int64_t nnz = 0;                // entries of L (strict) plus U
  std::vector<int> col_ptr;       // n + 1
  std::vector<int> row_idx;       // nnz
  std::vector<int> diag;          // n: position of U(j,j)
  std::vector<int> a_to_lu;       // per Jacobian entry: its factor position
  // Left-looking factorization tasks. Column j receives one update segment
  // for each k < j with U(k,j) != 0, in increasing k: subtract
  // L(:,k) * U(k,j) from the rows of column j listed in dst.
  std::vector<int> seg_ptr;       // n + 1: segments of column j
  std::vector<int> seg_upos;      // position of U(k,j)
  std::vector<int> seg_lbeg;      // first position of L(:,k) (strict)
  std::vector<int> seg_dst;       // first index in dst
  std::vector<int> seg_len;       // number of L(:,k) entries
  std::vector<int> dst;           // target positions in column j
  std::vector<int> lev_ptr;       // factorization levels
  std::vector<int> lev_cols;
  // Row-oriented triangular solves (strict parts, CSR)
  std::vector<int> lrow_ptr, lrow_col, lrow_pos;
  std::vector<int> urow_ptr, urow_col, urow_pos;
  std::vector<int> llev_ptr, llev_rows;   // forward solve levels
  std::vector<int> ulev_ptr, ulev_rows;   // backward solve levels
  std::vector<double> col_scale;  // max |A'(:,j)| of the reference matrix
  int64_t flops = 0;              // multiply-adds per factorization
};

/**
 * Run KLU (no block triangular preordering, so there is a single block) on
 * the reference Jacobian and derive the fixed-order LU plan.
 * @param pattern superset pattern
 * @param values reference Jacobian values in pattern order
 * @param ordering 0 = AMD, 1 = COLAMD (KLU's own codes)
 * @param pivot_tolerance KLU partial pivoting tolerance
 * @throws Error if KLU fails (e.g. the reference Jacobian is singular)
 */
LuPlan planLu(const JacobianPattern &pattern,
              const std::vector<double> &values, int ordering,
              double pivot_tolerance);

/**
 * Column-compressed copy of the pattern for libraries that want CSC
 * (KLU). csr_to_csc[p] gives the CSC position of CSR entry p.
 */
struct CscPattern {
  std::vector<int> col_ptr;
  std::vector<int> row_idx;
  std::vector<int> csr_to_csc;
};
CscPattern toCsc(const JacobianPattern &pattern);

}  // namespace batchpf
}  // namespace gridpack

#endif
