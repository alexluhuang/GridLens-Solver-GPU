/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   classifier.cpp
 * @date   2026-10-05
 *
 * @brief Contingency classifier and value-change builder (B4.3, B4.4). See
 * classifier.hpp for the rules.
 *
 * This file is part of the adapter layer between GridPACK and the batch
 * path: it calls GridPACK interfaces that hand out shared pointers and
 * generic components, which have to be downcast to power flow components
 * (exception EX-CG-03, confined to this layer).
 */

#include "classifier.hpp"

#include <algorithm>
#include <set>
#include <utility>

namespace gridpack {
namespace batchpf {

namespace {

using gridpack::powerflow::PFBranch;
using gridpack::powerflow::PFBus;
using gridpack::powerflow::SupersetCaseState;

PFBranch *branchAt(gridpack::powerflow::PFNetwork &net, int i)
{
  return dynamic_cast<PFBranch *>(net.getBranch(i).get());
}

PFBus *busAt(gridpack::powerflow::PFNetwork &net, int i)
{
  return dynamic_cast<PFBus *>(net.getBus(i).get());
}

/// Index of a circuit tag on a branch, -1 if absent
int tagIndex(PFBranch *br, const std::string &tag)
{
  const std::vector<std::string> tags = br->getLineTags();
  for (std::size_t k = 0; k < tags.size(); k++) {
    if (tags[k] == tag) return static_cast<int>(k);
  }
  return -1;
}

}  // namespace

const char *cpuReasonName(CpuReason r)
{
  switch (r) {
    case CpuReason::StudyControls: return "study_controls";
    case CpuReason::RemoteRegulation: return "remote_regulation";
    case CpuReason::NotFound: return "not_found_or_no_slack";
    case CpuReason::Islanded: return "islanded";
    case CpuReason::NoSlack: return "no_slack";
    case CpuReason::Other: return "other";
    default: return "";
  }
}

Classifier::Classifier(gridpack::powerflow::PFAppModule &pf_app,
                       boost::shared_ptr<gridpack::powerflow::PFNetwork> network,
                       const gridpack::powerflow::SupersetModel &model,
                       bool study_controls)
    : p_app(pf_app), p_network(std::move(network)), p_model(model),
      p_study_controls(study_controls)
{
  for (std::size_t k = 0; k < model.buses.size(); k++) {
    if (model.buses[k].type == gridpack::powerflow::SUPERSET_REF) {
      p_base_slack = static_cast<int>(k);
    }
  }
  if (study_controls || model.has_remote_regulation) {
    p_fast_note = "not used: every case goes to the CPU path";
    return;
  }
  // Apply an empty contingency with GridPACK's routines to see what the
  // unmodified network looks like to them
  gridpack::powerflow::Contingency empty;
  empty.p_type = gridpack::powerflow::Branch;
  empty.p_name = "batchpf_base_check";
  const bool ok = p_app.setContingency(empty);
  gridpack::powerflow::ContingencyEffects fx;
  p_app.getContingencyEffects(&fx);
  p_app.unSetContingency(empty);
  if (!ok || fx.island_count != 1 || fx.lone_bus || fx.slack_transferred) {
    p_fast_note = "not used: the unmodified network already has islands, lone "
                  "buses or a slack without a unit; every case takes GridPACK's "
                  "full contingency routine";
    return;
  }
  p_base_slack = fx.slack_bus;
  findBridges();
  p_fast_ok = true;
  int bridges = 0;
  for (char b : p_bridge) bridges += b ? 1 : 0;
  p_fast_note = "used; " + std::to_string(bridges) +
                " branches are bridges and take the full routine";
}

/**
 * Bridges of the graph detectIslands() uses: non-isolated buses joined by
 * branches with at least one element in service. Parallel branch objects
 * are separate edges, so a pair joined twice is never a bridge.
 */
void Classifier::findBridges()
{
  auto &net = *p_network;
  const int nb = net.numBuses();
  const int nbr = net.numBranches();
  p_bridge.assign(nbr, 0);
  std::vector<std::vector<std::pair<int, int>>> adj(nb);   // (bus, branch)
  for (int i = 0; i < nbr; i++) {
    if (!net.getActiveBranch(i)) continue;
    const std::vector<bool> st = branchAt(net, i)->getLineStatus();
    if (std::find(st.begin(), st.end(), true) == st.end()) continue;
    int a, b;
    net.getBranchEndpoints(i, &a, &b);
    if (a == b || busAt(net, a)->isIsolated() || busAt(net, b)->isIsolated()) continue;
    adj[a].emplace_back(b, i);
    adj[b].emplace_back(a, i);
  }
  // Iterative Tarjan bridge search
  std::vector<int> disc(nb, -1), low(nb, 0);
  int timer = 0;
  struct Frame { int bus; int via; std::size_t next; };
  for (int s = 0; s < nb; s++) {
    if (disc[s] >= 0 || !net.getActiveBus(s)) continue;
    std::vector<Frame> stack{{s, -1, 0}};
    disc[s] = low[s] = timer++;
    while (!stack.empty()) {
      Frame &f = stack.back();
      if (f.next < adj[f.bus].size()) {
        const std::pair<int, int> e = adj[f.bus][f.next++];
        if (e.second == f.via) continue;
        if (disc[e.first] < 0) {
          disc[e.first] = low[e.first] = timer++;
          stack.push_back({e.first, e.second, 0});
        } else {
          low[f.bus] = std::min(low[f.bus], disc[e.first]);
        }
      } else {
        const Frame done = f;
        stack.pop_back();
        if (!stack.empty()) {
          Frame &parent = stack.back();
          low[parent.bus] = std::min(low[parent.bus], low[done.bus]);
          if (low[done.bus] > disc[parent.bus]) p_bridge[done.via] = 1;
        }
      }
    }
  }
}

void Classifier::finishUpdates(const SupersetCaseState &state, CaseClass *out) const
{
  out->bus_updates.clear();
  out->edge_updates.clear();
  for (const auto &u : state.buses) {
    batchpf_bus_update b;
    b.bus = u.bus;
    b.type = u.type;
    b.g_diag = u.g_diag;
    b.b_diag = u.b_diag;
    b.p0 = u.p0;
    b.q0 = u.q0;
    b.qmax = u.qmax;
    b.qmin = u.qmin;
    out->bus_updates.push_back(b);
    if (u.remote_regulation) {
      out->path = CasePath::Cpu;
      out->reason = CpuReason::RemoteRegulation;
    }
  }
  for (const auto &u : state.edges) {
    batchpf_edge_update e;
    e.edge = u.edge;
    e.reserved = 0;
    e.g = u.g;
    e.b = u.b;
    out->edge_updates.push_back(e);
  }
}

bool Classifier::classifyFastBranch(int event, gridpack::powerflow::Contingency &c,
                                    CaseClass *out)
{
  auto &net = *p_network;
  const std::vector<int> lids = net.getLocalBranchIndices(c.p_from[0], c.p_to[0]);
  int lid = -1, idx = -1, found = 0;
  for (int i : lids) {
    const int k = tagIndex(branchAt(net, i), c.p_ckt[0]);
    if (k >= 0) {
      lid = i;
      idx = k;
      found++;
    }
  }
  if (found != 1) return false;   // missing or ambiguous: full routine
  PFBranch *br = branchAt(net, lid);
  const bool saved = br->getBranchStatus(c.p_ckt[0]);
  const std::vector<bool> st = br->getLineStatus();
  bool other_active = false;
  for (std::size_t k = 0; k < st.size(); k++) {
    if (static_cast<int>(k) != idx && st[k]) other_active = true;
  }
  if (saved && !other_active && p_bridge[lid]) return false;
  br->setBranchStatus(c.p_ckt[0], false);
  SupersetCaseState state, scratch;
  p_app.captureCaseState(p_model, std::vector<int>(), std::vector<int>(1, lid), &state);
  br->setBranchStatus(c.p_ckt[0], saved);
  p_app.captureCaseState(p_model, std::vector<int>(), std::vector<int>(1, lid), &scratch);
  out->event = event;
  out->path = CasePath::Gpu;
  out->fast = true;
  out->slack_bus = p_base_slack;
  finishUpdates(state, out);
  return true;
}

bool Classifier::classifyFastGenerator(int event, gridpack::powerflow::Contingency &c,
                                       CaseClass *out)
{
  auto &net = *p_network;
  const std::vector<int> lids = net.getLocalBusIndices(c.p_busid[0]);
  if (lids.size() != 1) return false;
  PFBus *bus = busAt(net, lids[0]);
  const std::string &tag = c.p_genid[0];
  const std::vector<std::string> gens = bus->getGenerators();
  if (std::find(gens.begin(), gens.end(), tag) == gens.end()) return false;
  const bool saved = bus->getGenStatus(tag);
  if (bus->getReferenceBus() && saved) {
    int others = 0;
    for (int j = 0; j < bus->getNumGenerators(); j++) {
      if (gens[j] != tag && bus->getGenStatusByIdx(j) == 1) others++;
    }
    if (others == 0) return false;   // slack would move: full routine
  }
  bus->setGenStatus(tag, false);
  SupersetCaseState state, scratch;
  p_app.captureCaseState(p_model, std::vector<int>(1, lids[0]), std::vector<int>(), &state);
  bus->setGenStatus(tag, saved);
  p_app.captureCaseState(p_model, std::vector<int>(1, lids[0]), std::vector<int>(), &scratch);
  out->event = event;
  out->path = CasePath::Gpu;
  out->fast = true;
  out->slack_bus = p_base_slack;
  finishUpdates(state, out);
  return true;
}

CaseClass Classifier::classifyFull(int event, gridpack::powerflow::Contingency &c)
{
  auto &net = *p_network;
  CaseClass out;
  out.event = event;
  const bool found = p_app.setContingency(c);
  gridpack::powerflow::ContingencyEffects fx;
  p_app.getContingencyEffects(&fx);
  std::vector<int> buses = fx.isolated_buses;
  std::vector<int> branches;
  if (fx.slack_bus >= 0) buses.push_back(fx.slack_bus);
  if (p_base_slack >= 0) buses.push_back(p_base_slack);
  if (c.p_type == gridpack::powerflow::Generator) {
    for (int id : c.p_busid) {
      for (int l : net.getLocalBusIndices(id)) buses.push_back(l);
    }
  } else {
    for (std::size_t k = 0; k < c.p_from.size(); k++) {
      for (int l : net.getLocalBranchIndices(c.p_from[k], c.p_to[k])) {
        if (tagIndex(branchAt(net, l), c.p_ckt[k]) >= 0) branches.push_back(l);
      }
    }
  }
  std::sort(branches.begin(), branches.end());
  branches.erase(std::unique(branches.begin(), branches.end()), branches.end());
  SupersetCaseState state, scratch;
  p_app.captureCaseState(p_model, buses, branches, &state);
  p_app.unSetContingency(c);
  p_app.captureCaseState(p_model, buses, branches, &scratch);
  out.island_count = fx.island_count;
  out.lone_bus = fx.lone_bus;
  out.slack_transferred = fx.slack_transferred;
  out.slack_bus = fx.slack_bus;
  if (!found) {
    out.path = CasePath::Cpu;
    out.reason = CpuReason::NotFound;
    return out;
  }
  if (fx.island_count > 1) {
    out.path = CasePath::Cpu;
    out.reason = CpuReason::Islanded;
    return out;
  }
  out.path = CasePath::Gpu;
  finishUpdates(state, &out);
  return out;
}

CaseClass Classifier::classify(int event, gridpack::powerflow::Contingency &c)
{
  CaseClass out;
  out.event = event;
  if (p_study_controls) {
    out.reason = CpuReason::StudyControls;
    return out;
  }
  if (p_model.has_remote_regulation) {
    out.reason = CpuReason::RemoteRegulation;
    return out;
  }
  if (p_fast_ok) {
    if (c.p_type == gridpack::powerflow::Branch && c.p_from.size() == 1 &&
        c.p_to.size() == 1 && c.p_ckt.size() == 1) {
      if (classifyFastBranch(event, c, &out)) return out;
    } else if (c.p_type == gridpack::powerflow::Generator && c.p_busid.size() == 1 &&
               c.p_genid.size() == 1) {
      if (classifyFastGenerator(event, c, &out)) return out;
    }
  }
  return classifyFull(event, c);
}

}  // namespace batchpf
}  // namespace gridpack
