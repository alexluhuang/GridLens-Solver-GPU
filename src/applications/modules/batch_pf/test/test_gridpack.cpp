/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   test_gridpack.cpp
 * @date   2026-10-05
 *
 * @brief batchpf.parity.*: the batch path against GridPACK's own components.
 *
 * Usage: batchpf_test_gridpack input.xml
 * (the input names a RAW file and sets Powerflow/jacobianFormulation=large)
 *
 *  1. Kernel parity (guide 8.10, 8.14): at the solved base case, the
 *     mismatch and Jacobian computed by the GPU element functions from the
 *     exported superset model equal the values GridPACK's power flow
 *     components return in their large layout (extension E6), entry by
 *     entry. Where GridPACK has no block (a branch next to the reference
 *     bus), the superset value must be zero.
 *     The reactive-limit check then converts the same buses as GridPACK's
 *     chkQlim() at that state, and neither check converts a disconnected
 *     bus, whatever injection that bus last computed.
 *  2. Classification: every N-1 case classified by the fast path gives the
 *     same path and the same values as GridPACK's full contingency routine.
 */

#include <mpi.h>
#include <ga.h>
#include <macdecls.h>

#include <algorithm>
#include <cmath>
#include <array>
#include <iostream>
#include <iomanip>
#include <boost/make_shared.hpp>
#include <map>
#include <string>
#include <vector>

#include "../core/pf_kernels.cuh"
#include "../core/planner.hpp"
#include "../host/classifier.hpp"
#include "gridpack/applications/modules/powerflow/pf_app_module.hpp"
#include "gridpack/configuration/configuration.hpp"
#include "gridpack/math/math.hpp"

using namespace gridpack::batchpf;
using gridpack::powerflow::PFBranch;
using gridpack::powerflow::PFBus;

namespace {

struct Checks {
  int failures = 0;
  void operator()(bool ok, const std::string &what) {
    if (!ok) {
      std::cout << "FAILED: " << what << "\n";
      failures++;
    }
  }
};

bool close(double a, double b, double tol)
{
  return std::fabs(a - b) <= tol * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

/// Kernel parity at the current (solved) state
void kernelParity(gridpack::powerflow::PFAppModule &app,
                   boost::shared_ptr<gridpack::powerflow::PFNetwork> net, Checks &check)
{
  gridpack::powerflow::SupersetModel model;
  app.exportSupersetModel(&model);
  const int n = static_cast<int>(model.buses.size());
  const JacobianPattern pat = buildPattern(n, model.row_start, model.edge_col);
  // Superset values from the element functions, one member, on the CPU
  std::vector<int> type(n), one(1, 1), conv(n, 0), edge_row(model.edge_col.size());
  std::vector<double> g(n), b(n), p0(n), q0(n), qmax(n), qmin(n), v(n), th(n), thw(n);
  std::vector<double> ql(n), ip(n), iq(n), yp(n), yq(n), pinj(n), qinj(n), qreq(n);
  std::vector<double> dg_q(n), dc_q(n);
  for (int k = 0; k < n; k++) {
    const auto &s = model.buses[k];
    type[k] = s.type;
    g[k] = s.g_diag;
    b[k] = s.b_diag;
    p0[k] = s.p0;
    q0[k] = s.q0;
    qmax[k] = s.qmax;
    qmin[k] = s.qmin;
    v[k] = s.v_solved;
    th[k] = s.theta_solved;
    thw[k] = s.theta_solved;   // already GridPACK's wrapped (exchanged) angle
    ql[k] = s.ql;
    dg_q[k] = s.dg_q;
    dc_q[k] = s.dc_q;
    ip[k] = s.ip;
    iq[k] = s.iq;
    yp[k] = s.yp;
    yq[k] = s.yq;
    for (int e = model.row_start[k]; e < model.row_start[k + 1]; e++) edge_row[e] = k;
  }
  std::vector<double> eg = model.edge_g, eb = model.edge_b;
  std::vector<double> F(2 * n), X(2 * n), J(static_cast<std::size_t>(pat.nnz), 0.0);
  std::vector<unsigned long long> mp(1, 0), mq(1, 0);
  std::vector<int> ap(1, 1 << 30), aq(1, 1 << 30), qv(1, 0);
  ModelView m;
  m.n_bus = n;
  m.n_edge = static_cast<int>(eg.size());
  m.sbase = model.sbase;
  m.row_start = model.row_start.data();
  m.edge_col = model.edge_col.data();
  m.ql = ql.data();
  m.dg_q = dg_q.data();
  m.dc_q = dc_q.data();
  m.ip = ip.data();
  m.iq = iq.data();
  m.yp = yp.data();
  m.yq = yq.data();
  m.diag_pos = pat.diag_pos.data();
  m.edge_pos = pat.edge_pos.data();
  BatchView w;
  w.B = 1;
  w.type = type.data();
  w.g = g.data();
  w.b = b.data();
  w.p0 = p0.data();
  w.q0 = q0.data();
  w.qmax = qmax.data();
  w.qmin = qmin.data();
  w.eg = eg.data();
  w.eb = eb.data();
  w.v = v.data();
  w.theta = th.data();
  w.thw = thw.data();
  w.pinj = pinj.data();
  w.qinj = qinj.data();
  w.F = F.data();
  w.X = X.data();
  w.J = J.data();
  w.conv = conv.data();
  w.qreq = qreq.data();
  w.m_eval = one.data();
  w.m_maxp = mp.data();
  w.m_maxq = mq.data();
  w.m_argp = ap.data();
  w.m_argq = aq.data();
  w.m_qviol = qv.data();
  const Mismatch mis{m, w};
  const JacobianDiag jd{m, w};
  const JacobianEdge je{m, w, edge_row.data()};
  for (int k = 0; k < n; k++) mis(k);
  for (int k = 0; k < n; k++) jd(k);
  for (int e = 0; e < m.n_edge; e++) je(e);

  // GridPACK's own values in the large layout
  double worst_f = 0.0, worst_d = 0.0, worst_o = 0.0;
  for (int k = 0; k < n; k++) {
    PFBus *bus = dynamic_cast<PFBus *>(net->getBus(k).get());
    if (bus->isIsolated()) continue;
    std::array<double, 4> r{};
    const int nr = bus->rhsValues(r.data());
    if (nr == 2 && !bus->getReferenceBus()) {
      worst_f = std::max(worst_f, std::fabs(r.at(0) - F[2 * k]));
      worst_f = std::max(worst_f, std::fabs(r.at(1) - F[2 * k + 1]));
    }
    std::array<double, 4> jv{};
    const int nj = bus->diagonalJacobianValues(jv.data());
    check(nj == 4, "large layout gives 2x2 diagonal blocks");
    for (int q = 0; q < 4; q++) {
      const double mine = J[pat.diag_pos[4 * k + q]];
      if (!close(jv.at(q), mine, 1e-10)) {
        worst_d = std::max(worst_d, std::fabs(jv.at(q) - mine));
      }
    }
  }
  // Branch blocks summed per bus pair, compared with the edge blocks
  std::map<int, std::vector<double>> sum;
  for (int i = 0; i < net->numBranches(); i++) {
    const int e = model.branch_edge[i];
    if (e < 0) continue;
    PFBranch *br = dynamic_cast<PFBranch *>(net->getBranch(i).get());
    std::array<double, 4> f{}, r{};
    const int nf = br->forwardJacobianValues(f.data());
    const int nrv = br->reverseJacobianValues(r.data());
    std::vector<double> &sf = sum[e], &sr = sum[model.edge_mate[e]];
    sf.resize(4, 0.0);
    sr.resize(4, 0.0);
    for (int q = 0; q < 4; q++) {
      if (nf == 4) sf[q] += f.at(q);
      if (nrv == 4) sr[q] += r.at(q);
    }
  }
  for (const auto &x : sum) {
    for (int q = 0; q < 4; q++) {
      const double mine = J[pat.edge_pos[4 * x.first + q]];
      if (!close(x.second[q], mine, 1e-10)) {
        worst_o = std::max(worst_o, std::fabs(x.second[q] - mine));
      }
    }
  }
  std::cout << std::scientific << std::setprecision(2)
            << "kernel parity: mismatch max |diff| " << worst_f
            << ", diagonal blocks " << worst_d << ", branch blocks " << worst_o << "\n";
  check(worst_f < 1e-9, "mismatch equals GridPACK's rhsValues()");
  check(worst_d == 0.0, "diagonal blocks equal GridPACK's large layout");
  check(worst_o == 0.0, "branch blocks equal GridPACK's large layout");

  // Reactive-limit check (B8.7) at the same state: the GPU check converts
  // exactly the buses GridPACK's chkQlim() converts. Buses that are PV
  // without generators (remote voltage regulation) must never convert.
  std::vector<int> qcheck(1, 1);
  w.m_qcheck = qcheck.data();
  w.qlim_deadband = app.getSolverParameters().qlim_deadband;
  const QlimCheck qc{m, w};
  for (int k = 0; k < n; k++) qc(k);
  std::vector<char> was_pv(n);
  int no_gen_pv = 0;
  for (int k = 0; k < n; k++) {
    PFBus *bus = dynamic_cast<PFBus *>(net->getBus(k).get());
    was_pv[k] = bus->isPV();
    if (model.buses[k].type == BATCHPF_BUS_PV && bus->getNumGenerators() == 0) no_gen_pv++;
  }
  app.checkQlimViolations();
  int converted = 0, differ = 0;
  for (int k = 0; k < n; k++) {
    PFBus *bus = dynamic_cast<PFBus *>(net->getBus(k).get());
    if (bus->getReferenceBus() || bus->isIsolated()) continue;
    const bool by_gridpack = was_pv[k] && !bus->isPV();
    if (by_gridpack) converted++;
    if (by_gridpack != (conv[k] != 0)) differ++;
  }
  app.clearQlimViolations();
  PFBus::clearQlimWarnings();
  std::cout << "Q-limit check: GridPACK converts " << converted
            << " buses, the GPU check differs at " << differ << "; " << no_gen_pv
            << " PV buses have no generators\n";
  check(qv[0] == converted, "GPU check counts the same conversions");
  check(differ == 0, "GPU check converts the buses GridPACK converts");
}

/**
 * A disconnected bus is outside the solved network (guide 8.6.2), so the
 * reactive-limit check must ignore the injection it last computed. A PV bus
 * is moved far from its solution, as a diverging case leaves it, so that
 * its computed reactive output exceeds its limits. GridPACK converts it
 * while it is connected; once it is isolated it must stay PV, as in the GPU
 * check, which only looks at PV rows.
 */
void isolatedQlimCheck(gridpack::powerflow::PFAppModule &app,
                       boost::shared_ptr<gridpack::powerflow::PFNetwork> net, Checks &check)
{
  const double deadband = app.getSolverParameters().qlim_deadband;
  // Raising one bus's voltage by half drives its computed reactive output
  // far past any generator limit, as a diverging case can leave it
  constexpr double far_from_solution = 1.5;
  std::array<double, 4> r{};
  int tested = -1;
  bool converts_isolated = false, stays_pv = true;
  for (int k = 0; k < net->numBuses() && tested < 0; k++) {
    PFBus *bus = dynamic_cast<PFBus *>(net->getBus(k).get());
    if (!bus->isPV() || bus->getReferenceBus() || bus->isIsolated() ||
        bus->getNumGenerators() == 0) {
      continue;
    }
    const double v = bus->getVoltage(), theta = bus->getPhase();
    bus->setVoltageState(far_from_solution * v, theta);
    bus->rhsValues(r.data());
    const bool converts_connected = bus->chkQlim(deadband);
    bus->clearQlim();
    if (converts_connected) {
      tested = bus->getOriginalIndex();
      bus->setIsolated(true);
      converts_isolated = bus->chkQlim(deadband);
      stays_pv = bus->isPV();
      bus->setIsolated(false);
      bus->clearQlim();
    }
    bus->setVoltageState(v, theta);
    bus->rhsValues(r.data());
  }
  PFBus::clearQlimWarnings();
  std::cout << "isolated-bus check: bus " << tested << " exceeds its limits while connected; "
            << "isolated, it is " << (converts_isolated || !stays_pv ? "" : "not ")
            << "converted\n";
  check(tested >= 0, "some PV bus exceeds its reactive limits in the test state");
  check(!converts_isolated && stays_pv, "a disconnected bus is not converted on a stale injection");
}

/// Fast-path classification against GridPACK's full routine, all N-1
void classifierParity(gridpack::powerflow::PFAppModule &app,
                       boost::shared_ptr<gridpack::powerflow::PFNetwork> net, Checks &check)
{
  gridpack::powerflow::SupersetModel model;
  app.exportSupersetModel(&model);
  Classifier cls(app, net, model, false);
  std::vector<gridpack::powerflow::Contingency> cases;
  for (int i = 0; i < net->numBranches(); i++) {
    PFBranch *br = dynamic_cast<PFBranch *>(net->getBranch(i).get());
    for (const std::string &tag : br->getLineTags()) {
      gridpack::powerflow::Contingency c;
      c.p_type = gridpack::powerflow::Branch;
      c.p_name = "branch";
      c.p_from.push_back(br->getBus1OriginalIndex());
      c.p_to.push_back(br->getBus2OriginalIndex());
      c.p_ckt.push_back(tag);
      c.p_saveLineStatus.push_back(true);
      cases.push_back(c);
    }
  }
  for (int k = 0; k < net->numBuses(); k++) {
    PFBus *bus = dynamic_cast<PFBus *>(net->getBus(k).get());
    for (const std::string &gid : bus->getGenerators()) {
      gridpack::powerflow::Contingency c;
      c.p_type = gridpack::powerflow::Generator;
      c.p_name = "generator";
      c.p_busid.push_back(bus->getOriginalIndex());
      c.p_genid.push_back(gid);
      c.p_saveGenStatus.push_back(true);
      cases.push_back(c);
    }
  }
  int fast = 0, mismatched = 0;
  for (std::size_t e = 0; e < cases.size(); e++) {
    const CaseClass a = cls.classify(CaseIndex{static_cast<int>(e)}, cases[e]);
    if (!a.fast) continue;
    fast++;
    const CaseClass b = cls.classifyFull(CaseIndex{static_cast<int>(e)}, cases[e]);
    bool same = a.path == b.path && (a.path == CasePath::Gpu || a.reason == b.reason);
    std::map<int, batchpf_bus_update> fb;
    for (const auto &u : b.bus_updates) fb[u.bus] = u;
    for (const auto &u : a.bus_updates) {
      const auto it = fb.find(u.bus);
      if (it == fb.end()) {
        same = false;
        continue;
      }
      same = same && it->second.type == u.type && it->second.g_diag == u.g_diag &&
             it->second.b_diag == u.b_diag && it->second.p0 == u.p0 &&
             it->second.q0 == u.q0 && it->second.qmax == u.qmax;
    }
    std::map<int, batchpf_edge_update> fe;
    for (const auto &u : b.edge_updates) fe[u.edge] = u;
    for (const auto &u : a.edge_updates) {
      const auto it = fe.find(u.edge);
      same = same && it != fe.end() && it->second.g == u.g && it->second.b == u.b;
    }
    if (!same) {
      mismatched++;
      if (mismatched < 5) std::cout << "classification differs for case " << e << "\n";
    }
  }
  std::cout << "classifier parity: " << cases.size() << " cases, " << fast
            << " by the fast path, " << mismatched << " differ from GridPACK's full routine\n";
  check(mismatched == 0, "fast path equals GridPACK's contingency routine");
}

}  // namespace

int main(int argc, char **argv)
{
  Checks check;
  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(C_DBL, 200000, 200000);
  gridpack::math::Initialize(&argc, &argv);
  {
    gridpack::parallel::Communicator world;
    gridpack::utility::Configuration *config =
        gridpack::utility::Configuration::configuration();
    config->open(argc > 1 ? argv[1] : "input.xml", world);
    auto net = boost::make_shared<gridpack::powerflow::PFNetwork>(world);
    gridpack::powerflow::PFAppModule app;
    app.suppressOutput(true);
    app.readNetwork(net, config);
    app.initialize();
    check(app.getJacobianFormulation() == gridpack::powerflow::JACOBIAN_LARGE,
          "input selects the large Jacobian layout");
    const bool ok = app.solve();
    check(ok, "base case converges");
    kernelParity(app, net, check);
    isolatedQlimCheck(app, net, check);
    classifierParity(app, net, check);
  }
  gridpack::math::Finalize();
  GA_Terminate();
  if (check.failures == 0) {
    std::cout << "No errors detected\n";
  } else {
    std::cout << check.failures << " failure detected\n";
  }
  MPI_Finalize();
  return check.failures == 0 ? 0 : 1;
}
