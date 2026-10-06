/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   test_kernels.cu
 * @date   2026-10-05
 *
 * @brief batchpf.unit.kernels: element functions and batched linear algebra.
 *
 *  1. The mismatch and Jacobian functors give the same values on the GPU
 *     and on the CPU (CUDA Best Practices Guide 7.1.2). Results are compared
 *     within a tolerance, never bitwise, because the GPU may fuse
 *     multiply-adds and its sine and cosine round differently (7.3).
 *  2. The Jacobian matches finite differences of the mismatch (guide 8.1.5).
 *  3. The Algorithm 2 backend (fixed pivot order) solves every member to
 *     the accuracy of KLU with full pivoting (the CPU reference backend),
 *     including members whose bus roles differ from the planning matrix.
 *
 * The network is synthetic: a ring with random chords, every bus role
 * (reference, PV, PQ, isolated) and constant current and impedance loads.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../core/backend.hpp"
#include "../core/executor.cuh"
#include "../core/pf_kernels.cuh"
#include "../core/planner.hpp"

using namespace gridpack::batchpf;

namespace {

int failures = 0;

void check(bool ok, const std::string &what)
{
  if (!ok) {
    std::printf("FAILED: %s\n", what.c_str());
    failures++;
  }
}

/// Synthetic network in the model's CSR edge form
struct Net {
  int n = 0;
  std::vector<int> row_start, edge_col, edge_row, type;
  std::vector<double> eg, eb, g, b, p0, q0, ql, ip, iq, yp, yq, qmax, qmin;
};

Net makeNet(int n, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<std::vector<std::pair<int, std::pair<double, double>>>> adj(n);
  auto add = [&](int a, int c) {
    if (a == c) return;
    for (auto &x : adj[a]) {
      if (x.first == c) return;
    }
    const double r = 0.002 + 0.02 * u(rng), x = 0.02 + 0.2 * u(rng);
    const double den = r * r + x * x;
    const double gy = -r / den, by = x / den;   // off-diagonal = -1/z
    adj[a].push_back({c, {gy, by}});
    adj[c].push_back({a, {gy, by}});
  };
  for (int k = 0; k < n; k++) add(k, (k + 1) % n);
  for (int k = 0; k < n / 3; k++) {
    add(static_cast<int>(u(rng) * n) % n, static_cast<int>(u(rng) * n) % n);
  }
  Net net;
  net.n = n;
  net.row_start.push_back(0);
  for (int k = 0; k < n; k++) {
    std::sort(adj[k].begin(), adj[k].end());
    double gs = 0.0, bs = 0.0;
    for (auto &x : adj[k]) {
      net.edge_col.push_back(x.first);
      net.edge_row.push_back(k);
      net.eg.push_back(x.second.first);
      net.eb.push_back(x.second.second);
      gs -= x.second.first;
      bs -= x.second.second;
    }
    net.row_start.push_back(static_cast<int>(net.edge_col.size()));
    net.g.push_back(gs);
    net.b.push_back(bs + 0.05 * u(rng));   // line charging and shunts
    int t = BATCHPF_BUS_PQ;
    if (k == 0) t = BATCHPF_BUS_REF;
    else if (k % 7 == 3) t = BATCHPF_BUS_PV;
    else if (k == n / 2 || k == n / 2 + 5) t = BATCHPF_BUS_ISOLATED;
    net.type.push_back(t);
    net.p0.push_back(0.5 * (u(rng) - 0.6));
    net.q0.push_back(0.3 * (u(rng) - 0.6));
    net.ql.push_back(20.0 * u(rng));
    net.ip.push_back(5.0 * u(rng));
    net.iq.push_back(3.0 * u(rng));
    net.yp.push_back(4.0 * u(rng));
    net.yq.push_back(2.0 * u(rng));
    net.qmax.push_back(50.0);
    net.qmin.push_back(-20.0);
  }
  return net;
}

/// Buffers for B members in one memory kind
struct Batch {
  int B;
  Buffer<int> type, conv, ones, row_start, edge_col, edge_row, diag_pos, edge_pos;
  Buffer<double> g, b, p0, q0, qmax, qmin, eg, eb, v, th, thw, pinj, qinj, F, X, J, qreq;
  Buffer<double> ql, ip, iq, yp, yq;
  Buffer<unsigned long long> maxp, maxq;
  Buffer<int> argp, argq, qv;
  ModelView m;
  BatchView w;
};

template <class T>
void fill(Buffer<T> &dst, MemoryKind k, const std::vector<T> &src)
{
  dst.allocate(k, src.size());
  dst.upload(src.data(), src.size(), nullptr);
}

/// Replicate a per-item vector over B members, interleaved
template <class T>
std::vector<T> rep(const std::vector<T> &x, int B)
{
  std::vector<T> out(x.size() * B);
  for (std::size_t i = 0; i < x.size(); i++) {
    for (int b = 0; b < B; b++) out[i * B + b] = x[i];
  }
  return out;
}

void setup(Batch &d, MemoryKind k, const Net &net, const JacobianPattern &pat, int B,
           const std::vector<double> &v, const std::vector<double> &th,
           const std::vector<int> &types)
{
  d.B = B;
  fill(d.row_start, k, net.row_start);
  fill(d.edge_col, k, net.edge_col);
  fill(d.edge_row, k, net.edge_row);
  fill(d.diag_pos, k, pat.diag_pos);
  fill(d.edge_pos, k, pat.edge_pos);
  fill(d.ql, k, net.ql);
  fill(d.ip, k, net.ip);
  fill(d.iq, k, net.iq);
  fill(d.yp, k, net.yp);
  fill(d.yq, k, net.yq);
  fill(d.type, k, types);
  fill(d.g, k, rep(net.g, B));
  fill(d.b, k, rep(net.b, B));
  fill(d.p0, k, rep(net.p0, B));
  fill(d.q0, k, rep(net.q0, B));
  fill(d.qmax, k, rep(net.qmax, B));
  fill(d.qmin, k, rep(net.qmin, B));
  fill(d.eg, k, rep(net.eg, B));
  fill(d.eb, k, rep(net.eb, B));
  fill(d.v, k, v);
  fill(d.th, k, th);
  std::vector<double> thw(th.size());
  for (std::size_t i = 0; i < th.size(); i++) thw[i] = wrapAngle(th[i]);
  fill(d.thw, k, thw);
  const std::size_t nB = static_cast<std::size_t>(net.n) * B;
  for (Buffer<double> *x : {&d.pinj, &d.qinj, &d.qreq}) x->allocate(k, nB);
  d.F.allocate(k, 2 * nB);
  d.X.allocate(k, 2 * nB);
  d.J.allocate(k, static_cast<std::size_t>(pat.nnz) * B);
  d.conv.allocate(k, nB);
  fill(d.ones, k, std::vector<int>(B, 1));
  fill(d.maxp, k, std::vector<unsigned long long>(B, 0));
  fill(d.maxq, k, std::vector<unsigned long long>(B, 0));
  fill(d.argp, k, std::vector<int>(B, 1 << 30));
  fill(d.argq, k, std::vector<int>(B, 1 << 30));
  fill(d.qv, k, std::vector<int>(B, 0));
  ModelView &m = d.m;
  m.n_bus = net.n;
  m.n_edge = static_cast<int>(net.edge_col.size());
  m.sbase = 100.0;
  m.row_start = d.row_start.data();
  m.edge_col = d.edge_col.data();
  m.ql = d.ql.data();
  m.ip = d.ip.data();
  m.iq = d.iq.data();
  m.yp = d.yp.data();
  m.yq = d.yq.data();
  m.diag_pos = d.diag_pos.data();
  m.edge_pos = d.edge_pos.data();
  BatchView &w = d.w;
  w.B = B;
  w.type = d.type.data();
  w.g = d.g.data();
  w.b = d.b.data();
  w.p0 = d.p0.data();
  w.q0 = d.q0.data();
  w.qmax = d.qmax.data();
  w.qmin = d.qmin.data();
  w.eg = d.eg.data();
  w.eb = d.eb.data();
  w.v = d.v.data();
  w.theta = d.th.data();
  w.thw = d.thw.data();
  w.pinj = d.pinj.data();
  w.qinj = d.qinj.data();
  w.F = d.F.data();
  w.X = d.X.data();
  w.J = d.J.data();
  w.conv = d.conv.data();
  w.qreq = d.qreq.data();
  w.m_apply = d.ones.data();
  w.m_qcheck = d.ones.data();
  w.m_eval = d.ones.data();
  w.m_maxp = d.maxp.data();
  w.m_maxq = d.maxq.data();
  w.m_argp = d.argp.data();
  w.m_argq = d.argq.data();
  w.m_qviol = d.qv.data();
}

void evaluate(Batch &d, const Executor &ex, const Net &net)
{
  const int64_t nB = static_cast<int64_t>(net.n) * d.B;
  const int64_t eB = static_cast<int64_t>(net.edge_col.size()) * d.B;
  ex.run(nB, Mismatch{d.m, d.w}, "Mismatch");
  ex.run(nB, JacobianDiag{d.m, d.w}, "JacobianDiag");
  ex.run(eB, JacobianEdge{d.m, d.w, d.edge_row.data()}, "JacobianEdge");
  if (ex.onDevice()) cudaCheck(cudaDeviceSynchronize(), "sync");
}

template <class T>
std::vector<T> host(const Buffer<T> &x)
{
  std::vector<T> out(x.size());
  x.download(out.data(), x.size(), nullptr);
  cudaDeviceSynchronize();
  return out;
}

double maxRelDiff(const std::vector<double> &a, const std::vector<double> &b)
{
  double scale = 1e-30, d = 0.0;
  for (std::size_t i = 0; i < a.size(); i++) {
    scale = std::max(scale, std::fabs(a[i]));
    d = std::max(d, std::fabs(a[i] - b[i]));
  }
  return d / scale;
}

}  // namespace

int main()
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    std::printf("no GPU: GPU parts of this test skipped\nNo errors detected\n");
    return 0;
  }
  const int n = 300, B = 64;
  const Net net = makeNet(n, 12345);
  const JacobianPattern pat = buildPattern(n, net.row_start, net.edge_col);
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  // Random states per member; members 1.. also flip a few bus roles to
  // exercise the superset rows (PV to PQ, PQ to isolated)
  std::vector<double> v(static_cast<std::size_t>(n) * B), th(v.size());
  std::vector<int> types(v.size());
  for (int k = 0; k < n; k++) {
    for (int b = 0; b < B; b++) {
      const std::size_t i = static_cast<std::size_t>(k) * B + b;
      v[i] = 1.0 + 0.05 * u(rng);
      th[i] = 0.3 * u(rng) + (b == 3 && k == 5 ? 7.0 : 0.0);   // one large angle
      int t = net.type[k];
      if (b > 0 && t == BATCHPF_BUS_PV && (k + b) % 5 == 0) t = BATCHPF_BUS_PQ;
      if (b > 0 && t == BATCHPF_BUS_PQ && (k * 31 + b) % 97 == 0) t = BATCHPF_BUS_ISOLATED;
      types[i] = t;
    }
  }

  // 1. GPU and CPU evaluate the same functors to the same values
  Batch hb, db;
  setup(hb, MemoryKind::Host, net, pat, B, v, th, types);
  setup(db, MemoryKind::Device, net, pat, B, v, th, types);
  evaluate(hb, Executor(false, nullptr, 0), net);
  evaluate(db, Executor(true, nullptr, 0), net);
  const std::vector<double> Fh = host(hb.F), Fd = host(db.F);
  const std::vector<double> Jh = host(hb.J), Jd = host(db.J);
  const double dF = maxRelDiff(Fh, Fd), dJ = maxRelDiff(Jh, Jd);
  std::printf("GPU vs CPU: mismatch rel. diff %.2e, Jacobian rel. diff %.2e\n", dF, dJ);
  check(dF < 1e-12 && dJ < 1e-12, "GPU and CPU element functions agree");

  // 2. Jacobian against central differences of the mismatch (member 0)
  {
    const int b0 = 0;
    double worst = 0.0;
    for (int col = 2; col < 2 * n; col += 37) {
      const int bus = col / 2;
      if (types[static_cast<std::size_t>(bus) * B + b0] != BATCHPF_BUS_PQ) continue;
      const double h = 1e-6;
      std::vector<double> vp = v, tp = th, vm = v, tm = th;
      std::vector<double> &xp = (col % 2) ? vp : tp, &xm = (col % 2) ? vm : tm;
      xp[static_cast<std::size_t>(bus) * B + b0] += h;
      xm[static_cast<std::size_t>(bus) * B + b0] -= h;
      Batch p, q;
      setup(p, MemoryKind::Host, net, pat, B, vp, tp, types);
      setup(q, MemoryKind::Host, net, pat, B, vm, tm, types);
      evaluate(p, Executor(false, nullptr, 0), net);
      evaluate(q, Executor(false, nullptr, 0), net);
      const std::vector<double> Fp = host(p.F), Fm = host(q.F);
      for (int row = 0; row < 2 * n; row++) {
        const int rb = row / 2;
        if (types[static_cast<std::size_t>(rb) * B + b0] != BATCHPF_BUS_PQ) continue;
        double jv = 0.0;
        for (int pp = pat.row_ptr[row]; pp < pat.row_ptr[row + 1]; pp++) {
          if (pat.col_idx[pp] == col) jv = Jh[static_cast<std::size_t>(pp) * B + b0];
        }
        const std::size_t ri = static_cast<std::size_t>(row) * B + b0;
        const double fd = (Fp[ri] - Fm[ri]) / (2 * h);
        worst = std::max(worst, std::fabs(fd - jv) / std::max(1.0, std::fabs(jv)));
      }
    }
    std::printf("Jacobian vs finite differences: worst rel. error %.2e\n", worst);
    check(worst < 1e-6, "Jacobian matches finite differences");
  }

  // 3. Algorithm 2 against KLU with full pivoting, for every member
  {
    std::vector<double> ref(static_cast<std::size_t>(pat.nnz));
    for (int64_t p = 0; p < pat.nnz; p++) ref[p] = Jh[static_cast<std::size_t>(p) * B];
    const LuPlan lu = planLu(pat, ref, 0, 0.001);
    BackendSetup bs;
    bs.pattern = &pat;
    bs.lu = &lu;
    bs.reference_values = &ref;
    bs.capacity = B;
    bs.pivot_limit = 1e-14;
    std::unique_ptr<SolverBackend> cpu = makeCpuReferenceBackend(bs);
    std::unique_ptr<SolverBackend> gpu = makeAlg2Backend(bs);
    std::vector<double> rhs(static_cast<std::size_t>(2 * n) * B);
    for (double &r : rhs) r = u(rng);
    std::vector<int> mask(B, 1), sc(B, 0);
    std::vector<double> xc(rhs.size());
    cpu->refactorize(Jh, mask, sc);
    cpu->solve(rhs, xc, mask, sc);
    Buffer<double> drhs(MemoryKind::Device, rhs.size()), dx(MemoryKind::Device, rhs.size());
    drhs.upload(rhs.data(), rhs.size(), nullptr);
    Buffer<int> dmask(MemoryKind::Device, B), dst(MemoryKind::Device, B);
    dmask.upload(mask.data(), B, nullptr);
    dst.zero(nullptr);
    gpu->refactorize(db.J.span(), dmask.span(), dst.span());
    gpu->solve(drhs.span(), dx.span(), dmask.span(), dst.span());
    const std::vector<double> xg = host(dx);
    const std::vector<int> sg = host(dst);
    int bad = 0;
    for (int b = 0; b < B; b++) bad += (sg[b] != 0 || sc[b] != 0) ? 1 : 0;
    const double dx_rel = maxRelDiff(xc, xg);
    std::printf("Algorithm 2 vs KLU: max rel. difference %.2e, members flagged %d\n",
                dx_rel, bad);
    check(bad == 0, "no member flagged");
    check(dx_rel < 1e-9, "Algorithm 2 matches KLU");

    // Spreading a column over several warps per member must not change a
    // single bit: every factor entry gets the same operations in order
    for (int lanes : {1, 2, 8, 32}) {
      BackendSetup wide = bs;
      wide.factor_lanes = lanes;
      std::unique_ptr<SolverBackend> w = makeAlg2Backend(wide);
      dst.zero(nullptr);
      w->refactorize(db.J.span(), dmask.span(), dst.span());
      w->solve(drhs.span(), dx.span(), dmask.span(), dst.span());
      const std::vector<double> xw = host(dx);
      check(xw == xg, "Algorithm 2 with " + std::to_string(lanes) +
                          " threads per member gives bitwise the same solution");
    }

    // ROB-1: a singular member must not spoil the other solves. A masked
    // member must retain the state from its last successful solve.
    constexpr int singular = 7;
    constexpr int masked = 9;
    std::vector<double> broken = Jh;
    for (int64_t p = 0; p < pat.nnz; p++) broken[p * B + singular] = 0.0;
    mask[masked] = 0;
    db.J.upload(broken.data(), broken.size(), nullptr);
    dmask.upload(mask.data(), B, nullptr);
    dst.zero(nullptr);
    gpu->refactorize(db.J.span(), dmask.span(), dst.span());
    check(host(dst)[singular] != BATCHPF_MEMBER_OK,
          "singular member is flagged at factorization for CPU fallback");
    gpu->solve(drhs.span(), dx.span(), dmask.span(), dst.span());
    const auto isolated_status = host(dst);
    const auto isolated_x = host(dx);
    check(isolated_status[singular] == BATCHPF_MEMBER_NONFINITE,
          "non-finite takes precedence when solving the singular system");
    for (int b = 0; b < B; b++) {
      if (b == singular) continue;
      check(isolated_status[b] == BATCHPF_MEMBER_OK, "other members remain healthy");
      for (int k = 0; k < 2 * n; k++) {
        const auto i = static_cast<std::size_t>(k) * B + b;
        const double expected = b == masked ? xg[i] : xc[i];
        check(std::fabs(isolated_x[i] - expected) <= 1e-9 * std::max(1.0, std::fabs(expected)),
              "healthy solves agree with KLU and masked states are unchanged");
      }
    }
  }

  if (failures == 0) {
    std::printf("No errors detected\n");
    return 0;
  }
  std::printf("%d failure detected\n", failures);
  return 1;
}
