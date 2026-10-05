/*
 * Copyright (c) 2013 Battelle Memorial Institute
 * Licensed under modified BSD License. See LICENSE in the top level directory.
 */
#include "../core/engine.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

namespace {

void rejected(const batchpf_model &model)
{
  try {
    gridpack::batchpf::copyModel(model);
  } catch (const gridpack::batchpf::Error &e) {
    if (e.code() == BATCHPF_ERR_INVALID_ARGUMENT) return;
    throw;
  }
  throw std::runtime_error("invalid model was accepted");
}

}  // namespace

int main()
{
  try {
    std::array<double, 2> values{1.0, 1.0};
    std::array<int32_t, 2> types{BATCHPF_BUS_REF, BATCHPF_BUS_PQ};
    std::array<int32_t, 3> starts{0, 1, 2};
    std::array<int32_t, 2> columns{1, 0}, mates{1, 0};
    batchpf_model model{};
    model.struct_size = sizeof(model);
    model.struct_version = 1;
    model.n_bus = 2;
    model.n_edge = 2;
    model.sbase = 100;
    model.bus_type = types.data();
    model.g_diag = model.b_diag = model.p0 = model.q0 = values.data();
    model.v_init = model.theta_init = model.v_base = model.theta_base = values.data();
    model.ql = model.ip = model.iq = model.yp = model.yq = values.data();
    model.qmax = model.qmin = model.edge_g = model.edge_b = values.data();
    model.row_start = starts.data();
    model.edge_col = columns.data();
    model.edge_mate = mates.data();
    gridpack::batchpf::copyModel(model);
    starts[1] = -1;
    rejected(model);
    starts[1] = 3;
    rejected(model);
    starts[1] = 1;
    mates[0] = 2;
    rejected(model);
    mates[0] = 0;
    rejected(model);
    mates[0] = 1;
    columns[0] = 2;
    rejected(model);
    columns[0] = 1;
    starts = {0, 0, 0};
    model.n_edge = 0;
    model.edge_col = model.edge_mate = nullptr;
    model.edge_g = model.edge_b = nullptr;
    gridpack::batchpf::copyModel(model);
    std::cout << "No errors detected\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "failure detected: " << e.what() << '\n';
    return 1;
  }
}
