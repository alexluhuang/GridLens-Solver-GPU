/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   model.cpp
 * @date   2026-10-05
 *
 * @brief Copy and check the superset model handed over by ca.x (I-4).
 *
 * The plugin keeps its own copy, so ca.x may free its arrays as soon as
 * set_model() returns. Indices are checked once here so the kernels can
 * trust them.
 */

#include <vector>

#include "engine.hpp"

namespace gridpack {
namespace batchpf {

ModelHost copyModel(const batchpf_model &m)
{
  if (m.struct_size < sizeof(batchpf_model) || m.n_bus <= 0 || m.n_edge < 0) {
    throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "invalid model record");
  }
  ModelHost h;
  h.n_bus = m.n_bus;
  h.n_edge = m.n_edge;
  h.sbase = m.sbase;
  const std::size_t n = static_cast<std::size_t>(m.n_bus);
  const std::size_t e = static_cast<std::size_t>(m.n_edge);
  auto copyD = [](const double *p, std::size_t len) {
    if (len == 0) return std::vector<double>{};
    if (p == nullptr) throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "null model array");
    const gsl::span<const double> values(p, gsl::narrow<gsl::index>(len));
    return std::vector<double>(values.begin(), values.end());
  };
  auto copyI = [](const int32_t *p, std::size_t len) {
    if (len == 0) return std::vector<int>{};
    if (p == nullptr) throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "null model array");
    const gsl::span<const int32_t> values(p, gsl::narrow<gsl::index>(len));
    return std::vector<int>(values.begin(), values.end());
  };
  h.bus_type = copyI(m.bus_type, n);
  h.g = copyD(m.g_diag, n);
  h.b = copyD(m.b_diag, n);
  h.p0 = copyD(m.p0, n);
  h.q0 = copyD(m.q0, n);
  h.v_init = copyD(m.v_init, n);
  h.theta_init = copyD(m.theta_init, n);
  h.v_base = copyD(m.v_base, n);
  h.theta_base = copyD(m.theta_base, n);
  h.ql = copyD(m.ql, n);
  h.ip = copyD(m.ip, n);
  h.iq = copyD(m.iq, n);
  h.yp = copyD(m.yp, n);
  h.yq = copyD(m.yq, n);
  h.qmax = copyD(m.qmax, n);
  h.qmin = copyD(m.qmin, n);
  h.row_start = copyI(m.row_start, n + 1);
  h.edge_col = copyI(m.edge_col, e);
  h.edge_mate = copyI(m.edge_mate, e);
  h.eg = copyD(m.edge_g, e);
  h.eb = copyD(m.edge_b, e);
  if (h.row_start[0] != 0 || h.row_start[n] != m.n_edge) {
    throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "edge row starts do not match");
  }
  h.edge_row.assign(e, 0);
  for (std::size_t k = 0; k < n; k++) {
    if (h.row_start[k] < 0 || h.row_start[k + 1] < h.row_start[k] ||
        h.row_start[k + 1] > m.n_edge) {
      throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "edge row starts are not ordered within bounds");
    }
    for (int x = h.row_start[k]; x < h.row_start[k + 1]; x++) {
      if (h.edge_col[x] < 0 || h.edge_col[x] >= m.n_bus) {
        throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "edge column out of range");
      }
      h.edge_row[x] = static_cast<int>(k);
    }
  }
  for (std::size_t x = 0; x < e; x++) {
    const int mate = h.edge_mate[x];
    if (mate < 0 || mate >= m.n_edge || h.edge_mate[mate] != static_cast<int>(x) ||
        h.edge_col[x] != h.edge_row[mate] || h.edge_col[mate] != h.edge_row[x]) {
      throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "reverse edge does not match its endpoints");
    }
  }
  for (std::size_t k = 0; k < n; k++) {
    if (h.bus_type[k] == BATCHPF_BUS_REF) h.base_slack = static_cast<int>(k);
  }
  if (h.base_slack < 0) {
    throw Error(BATCHPF_ERR_INVALID_ARGUMENT, "model has no reference bus");
  }
  return h;
}

}  // namespace batchpf
}  // namespace gridpack
