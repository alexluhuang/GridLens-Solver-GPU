/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   pf_factory.cpp
 * @author Bruce Palmer
 * @date   2014-01-28 11:31:23 d3g096
 *
 * @updated Yousu Chen
 * - Added setInitStartMode for power flow initialization (warm/flat start)
 * @date  2026-02-02
 *
 * @updated Yousu Chen
 * - Added checkSwitchedShuntViolations() and clearSwitchedShunts()
 * @date  2026-02-24
 *
 * @updated Yousu Chen
 * - Added checkLTCViolations() and clearLTCControls()
 * - Added computeAreaExport() for area interchange control
 * @date  2026-03-28
 *
 * @updated Yousu Chen
 * - Added setupIREGPointers() for IREG PV bus swap with MPI Allgatherv
 *   for multi-process consistency (active + ghost bus copies)
 * - Slack bus remote IREG auto-correction with warning
 * - Skip IREG outer loop for buses handled by PV swap
 * @date  2026-04-04
 *
 * @brief
 *
 *
 */
// -------------------------------------------------------------

#include <vector>
#include <queue>
#include <map>
#include <algorithm>
#include <limits>
#include "boost/smart_ptr/shared_ptr.hpp"
#include "gridpack/parser/dictionary.hpp"
#include "gridpack/parallel/global_vector.hpp"
#include "pf_factory_module.hpp"


namespace gridpack {
namespace powerflow {

// Powerflow factory class implementations

/**
 * Basic constructor
 * @param network: network associated with factory
 */
PFFactoryModule::PFFactoryModule(PFFactoryModule::NetworkPtr network)
  : gridpack::factory::BaseFactory<PFNetwork>(network)
{
  p_network = network;
  p_rateB = false;
  p_contingencyRating = "A";
  p_islandCount = 0;
  p_hasLoneBus = false;
  p_qlim_deadband = 0.1;
  p_originalSlackBusIdx = -1;
  p_currentSlackBusIdx = -1;
  p_slackTransferred = false;
}

/**
 * Basic destructor
 */
gridpack::powerflow::PFFactoryModule::~PFFactoryModule()
{
}

/**
 * Create the admittance (Y-Bus) matrix
 */
void gridpack::powerflow::PFFactoryModule::setYBus(void)
{
  int numBus = p_network->numBuses();
  int numBranch = p_network->numBranches();
  int i;

  // Invoke setYBus method on all branch objects
  for (i=0; i<numBranch; i++) {
    dynamic_cast<PFBranch*>(p_network->getBranch(i).get())->setYBus();
  }

  // Invoke setYBus method on all bus objects
  for (i=0; i<numBus; i++) {
    dynamic_cast<PFBus*>(p_network->getBus(i).get())->setYBus();
  }

}

/**
  * Make SBus vector 
  */
void gridpack::powerflow::PFFactoryModule::setSBus(void)
{
  int numBus = p_network->numBuses();
  int i;

  // Invoke setSBus method on all bus objects
  for (i=0; i<numBus; i++) {
    dynamic_cast<PFBus*>(p_network->getBus(i).get())->setSBus();
  }
}

/**
  * Create the PQ 
  */
void gridpack::powerflow::PFFactoryModule::setPQ(void)
{
  int numBus = p_network->numBuses();
  int i;
  ComplexType values[2];

  for (i=0; i<numBus; i++) {
    dynamic_cast<PFBus*>(p_network->getBus(i).get())->vectorValues(values);
  }
}

/**
 * Update pg of specified bus element based on their genID
 * @param name 
 * @param busID
 * @param genID
 * @param value
 */
//void gridpack::powerflow::PFFactoryModule::updatePg(std::string &name, int busID, std::string genID, double value)
void gridpack::powerflow::PFFactoryModule::updatePg(int busID, std::string genID, double value)
{
  int numBus = p_network->numBuses();
  int i;
  int genIndex=0;
  for (i=0; i<numBus; i++) {
//    dynamic_cast<PFBus*>(p_network->getBus(i).get())->setParam(name,busID, genID, value);
    dynamic_cast<PFBus*>(p_network->getBus(i).get())->setParam(GENERATOR_PG,
        busID, genID, value);
  }
}

void gridpack::powerflow::PFFactoryModule::updateQg(int busID, std::string genID, double value)
{
  int numBus = p_network->numBuses();
  int i;
  int genIndex=0;
  for (i=0; i<numBus; i++) {
//    dynamic_cast<PFBus*>(p_network->getBus(i).get())->setParam(name,busID, genID, value);
    dynamic_cast<PFBus*>(p_network->getBus(i).get())->setParam(GENERATOR_QG,
        busID, genID, value);
  }
}

/**
 * Check for lone buses in the system. Do this by looking for buses that
 * have no branches attached to them or for whom all the branches attached
 * to the bus have all transmission elements with status false (the element
 * is off). Set status of bus to isolated so that it does not contribute to
 * powerflow matrix
 * @param stream optional stream pointer that can be used to print out IDs
 * of isolated buses
 * @return false if there is an isolated bus in the network
 */
bool gridpack::powerflow::PFFactoryModule::checkLoneBus(std::ofstream *stream)
{
  int numBus = p_network->numBuses();
  int i, j, k;
  bool bus_ok = true;
  char buf[128];
  p_saveIsolatedStatus.clear();
  p_loneBusIndices.clear();
  for (i=0; i<numBus; i++) {
    if (!p_network->getActiveBus(i)) continue;
    gridpack::powerflow::PFBus *bus =
      dynamic_cast<gridpack::powerflow::PFBus*>
      (p_network->getBus(i).get());
    // Skip already-isolated buses (e.g. PSS/E type-4) so they are not
    // re-flagged as lone on every call.
    if (bus->isIsolated()) continue;
    std::vector<boost::shared_ptr<gridpack::component::BaseComponent> > branches;
    bus->getNeighborBranches(branches);
    int size = branches.size();
    bool ok = true;
    if (size == 0) {
      ok = false;
    }
    if (ok) {
      ok = false;
      for (j=0; j<size; j++) {
        bool branch_ok = false;
        std::vector<bool> status =
          dynamic_cast<gridpack::powerflow::PFBranch*>
          (branches[j].get())->getLineStatus();
        int nlines = status.size();
        for (k=0; k<nlines; k++) {
          if (status[k]) branch_ok = true;
        }
        if (branch_ok) ok = true;
      }
    }
    if (!ok) {
      sprintf(buf,"\nLone bus %d found\n",bus->getOriginalIndex());
      p_saveIsolatedStatus.push_back(bus->isIsolated());
      p_loneBusIndices.push_back(i);
      bus->setIsolated(true);
      printf("%s",buf);
      if (stream != NULL) *stream << buf;
    }
    if (!ok) bus_ok = false;
  }
  // Check whether bus_ok is true on all processors (lone bus found if bus_ok is false)
  p_hasLoneBus = checkTrue(!bus_ok);
  return p_hasLoneBus;
}

/**
 * Set lone buses back to their original status.
 */
void gridpack::powerflow::PFFactoryModule::clearLoneBus()
{
  p_hasLoneBus = false;
  // Restore status of buses marked by the last checkLoneBus call.
  for (size_t k = 0; k < p_loneBusIndices.size(); k++) {
    int i = p_loneBusIndices[k];
    gridpack::powerflow::PFBus *bus =
      dynamic_cast<gridpack::powerflow::PFBus*>
      (p_network->getBus(i).get());
    printf("\nLone bus %d reset\n", bus->getOriginalIndex());
    bus->setIsolated(p_saveIsolatedStatus[k]);
  }
  p_loneBusIndices.clear();
  p_saveIsolatedStatus.clear();
}

/**
 * Detect islands (disconnected subnetworks) in the network using BFS.
 * Mark buses in smaller islands as isolated to prevent singular Jacobian.
 * @param stream optional stream pointer for printing island info
 * @return number of islands found (1 = connected network, >1 = islanding)
 */
int gridpack::powerflow::PFFactoryModule::detectIslands(std::ofstream *stream)
{
  int numBus = p_network->numBuses();
  int numBranch = p_network->numBranches();
  int i, j, k;
  char buf[256];

  // Clear previous island isolated status
  p_saveIslandIsolatedStatus.clear();

  // Build mapping from bus local index to original index and vice versa
  std::map<int, int> origToLocal;  // original bus ID -> local index
  std::vector<int> localToOrig;    // local index -> original bus ID
  std::vector<bool> busActive;     // whether bus is active and not already isolated

  for (i = 0; i < numBus; i++) {
    if (!p_network->getActiveBus(i)) continue;
    gridpack::powerflow::PFBus *bus =
      dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(i).get());
    if (bus->isIsolated()) continue;  // Skip already isolated buses

    int origIdx = bus->getOriginalIndex();
    origToLocal[origIdx] = localToOrig.size();
    localToOrig.push_back(i);  // Store local network index
    busActive.push_back(true);
  }

  int activeBusCount = localToOrig.size();
  if (activeBusCount == 0) {
    p_islandCount = 0;
    return 0;
  }

  // Build adjacency list based on active branches
  std::vector<std::vector<int> > adj(activeBusCount);

  for (i = 0; i < numBranch; i++) {
    if (!p_network->getActiveBranch(i)) continue;

    gridpack::powerflow::PFBranch *branch =
      dynamic_cast<gridpack::powerflow::PFBranch*>(p_network->getBranch(i).get());

    // Check if branch has any active lines
    std::vector<bool> status = branch->getLineStatus();
    bool branchActive = false;
    for (k = 0; k < status.size(); k++) {
      if (status[k]) {
        branchActive = true;
        break;
      }
    }
    if (!branchActive) continue;

    // Get the two buses connected by this branch
    int bus1Orig = branch->getBus1OriginalIndex();
    int bus2Orig = branch->getBus2OriginalIndex();

    // Check if both buses are in our active set
    std::map<int, int>::iterator it1 = origToLocal.find(bus1Orig);
    std::map<int, int>::iterator it2 = origToLocal.find(bus2Orig);

    if (it1 != origToLocal.end() && it2 != origToLocal.end()) {
      int idx1 = it1->second;
      int idx2 = it2->second;
      adj[idx1].push_back(idx2);
      adj[idx2].push_back(idx1);
    }
  }

  // BFS to find connected components (islands)
  std::vector<int> islandId(activeBusCount, -1);
  std::vector<std::vector<int> > islands;  // Each island contains list of local indices
  int currentIsland = 0;

  for (i = 0; i < activeBusCount; i++) {
    if (islandId[i] >= 0) continue;  // Already assigned to an island

    // BFS from bus i
    std::vector<int> currentIslandBuses;
    std::queue<int> q;
    q.push(i);
    islandId[i] = currentIsland;

    while (!q.empty()) {
      int curr = q.front();
      q.pop();
      currentIslandBuses.push_back(curr);

      for (j = 0; j < adj[curr].size(); j++) {
        int neighbor = adj[curr][j];
        if (islandId[neighbor] < 0) {
          islandId[neighbor] = currentIsland;
          q.push(neighbor);
        }
      }
    }

    islands.push_back(currentIslandBuses);
    currentIsland++;
  }

  p_islandCount = islands.size();

  // If only one island, network is connected
  if (p_islandCount <= 1) {
    return p_islandCount;
  }

  // Find the largest island (main network)
  int largestIsland = 0;
  int largestSize = islands[0].size();
  for (i = 1; i < islands.size(); i++) {
    if (islands[i].size() > largestSize) {
      largestSize = islands[i].size();
      largestIsland = i;
    }
  }

  // Mark buses in smaller islands as isolated
  p_islandIsolatedBusIndices.clear();
  for (i = 0; i < islands.size(); i++) {
    if (i == largestIsland) continue;  // Keep main island

    sprintf(buf, "\nIsland %d detected with %d buses (marking as isolated):\n",
            i + 1, (int)islands[i].size());
    printf("%s", buf);
    if (stream != NULL) *stream << buf;

    for (j = 0; j < islands[i].size(); j++) {
      int localIdx = localToOrig[islands[i][j]];  // Get network local index
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(localIdx).get());

      sprintf(buf, "  Bus %d\n", bus->getOriginalIndex());
      printf("%s", buf);
      if (stream != NULL) *stream << buf;

      // Save current isolated status and local index, then mark as isolated
      p_saveIslandIsolatedStatus.push_back(bus->isIsolated());
      p_islandIsolatedBusIndices.push_back(localIdx);
      bus->setIsolated(true);
    }
  }

  sprintf(buf, "\nNetwork split into %d islands. Main island has %d buses.\n",
          p_islandCount, largestSize);
  printf("%s", buf);
  if (stream != NULL) *stream << buf;

  return p_islandCount;
}

/**
 * Get the number of islands detected in the last call to detectIslands
 * @return number of islands (0 if detectIslands not called)
 */
int gridpack::powerflow::PFFactoryModule::getIslandCount() const
{
  return p_islandCount;
}

/**
 * Check if any lone buses were found in the last call to checkLoneBus
 * @return true if at least one lone bus was found
 */
bool gridpack::powerflow::PFFactoryModule::hasLoneBus() const
{
  return p_hasLoneBus;
}

/**
 * Check if the reference (slack) bus has an online generator.
 * If not, transfer the slack function to the bus with the largest
 * online generator capacity.
 * @return true if a valid slack bus exists (or was transferred),
 *         false if no generator with real power capacity is available
 */
bool gridpack::powerflow::PFFactoryModule::checkAndTransferSlack()
{
  int numBus = p_network->numBuses();
  int i;

  // Find the current slack bus and check if it has an online generator
  int slackBusIdx = -1;
  bool slackHasOnlineGen = false;

  for (i = 0; i < numBus; i++) {
    if (!p_network->getActiveBus(i)) continue;
    gridpack::powerflow::PFBus *bus =
      dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(i).get());
    if (bus->getReferenceBus()) {
      slackBusIdx = i;
      p_originalSlackBusIdx = i;  // Save original slack bus
      slackHasOnlineGen = bus->hasOnlineGenerator();
      break;
    }
  }

  if (slackBusIdx < 0) {
    // No slack bus found - this shouldn't happen
    printf("ERROR: No reference bus found in network\n");
    return false;
  }

  // If slack bus has an online generator, we're good
  if (slackHasOnlineGen) {
    p_slackTransferred = false;
    p_currentSlackBusIdx = slackBusIdx;
    return true;
  }

  // Slack bus generator is offline - find the best candidate for new slack
  // Look for the bus with the largest online generator capacity
  int bestCandidateIdx = -1;
  double maxCapacity = 0.0;

  for (i = 0; i < numBus; i++) {
    if (!p_network->getActiveBus(i)) continue;
    gridpack::powerflow::PFBus *bus =
      dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(i).get());

    // Skip isolated buses
    if (bus->isIsolated()) continue;

    // Check if this bus has online generators with real power capacity
    double capacity = bus->getOnlineGenCapacity();
    if (capacity > maxCapacity) {
      maxCapacity = capacity;
      bestCandidateIdx = i;
    }
  }

  // If no bus with generation capacity found, system cannot be solved
  if (bestCandidateIdx < 0 || maxCapacity <= 0.0) {
    printf("WARNING: No generator with real power capacity available after contingency\n");
    printf("  Original slack bus has no online generator and no transfer candidate found\n");
    return false;
  }

  // Transfer slack to the new bus
  gridpack::powerflow::PFBus *oldSlack =
    dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(slackBusIdx).get());
  gridpack::powerflow::PFBus *newSlack =
    dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(bestCandidateIdx).get());

  oldSlack->setReferenceBus(false);
  newSlack->setReferenceBus(true);

  p_slackTransferred = true;
  p_currentSlackBusIdx = bestCandidateIdx;

  printf("Slack bus transferred from bus %d to bus %d (capacity: %.1f MW)\n",
         oldSlack->getOriginalIndex(), newSlack->getOriginalIndex(), maxCapacity);

  return true;
}

/**
 * Restore the original slack bus after a contingency.
 */
void gridpack::powerflow::PFFactoryModule::restoreSlack()
{
  if (!p_slackTransferred) return;

  // Restore the original slack bus
  if (p_originalSlackBusIdx >= 0 && p_currentSlackBusIdx >= 0 &&
      p_originalSlackBusIdx != p_currentSlackBusIdx) {
    gridpack::powerflow::PFBus *oldSlack =
      dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(p_currentSlackBusIdx).get());
    gridpack::powerflow::PFBus *origSlack =
      dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(p_originalSlackBusIdx).get());

    oldSlack->setReferenceBus(false);
    origSlack->setReferenceBus(true);

    printf("Slack bus restored to bus %d\n", origSlack->getOriginalIndex());
  }

  p_slackTransferred = false;
  p_currentSlackBusIdx = p_originalSlackBusIdx;
}

/**
 * Check if slack bus generator output exceeds its capacity (Pmax).
 * Should be called after power flow solve.
 * @return true if within limits, false if Pgen > Pmax
 */
bool gridpack::powerflow::PFFactoryModule::checkSlackCapacity()
{
  int numBus = p_network->numBuses();

  // Find the current slack bus
  for (int i = 0; i < numBus; i++) {
    if (!p_network->getActiveBus(i)) continue;
    gridpack::powerflow::PFBus *bus =
      dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(i).get());
    if (bus->getReferenceBus()) {
      bool withinLimits = bus->checkGenCapacity();
      if (!withinLimits) {
        double pgen = bus->getTotalGenOutput();
        double pmax = bus->getOnlineGenCapacity();
        printf("WARNING: Slack bus %d generator output (%.1f MW) exceeds capacity (%.1f MW)\n",
               bus->getOriginalIndex(), pgen, pmax);
      }
      return withinLimits;
    }
  }
  // No slack bus found
  return false;
}

/**
 * Clear island detection state and restore isolated status of buses
 * that were marked as isolated due to islanding
 */
void gridpack::powerflow::PFFactoryModule::clearIslands()
{
  if (p_islandIsolatedBusIndices.size() == 0) {
    p_islandCount = 0;
    p_saveIslandIsolatedStatus.clear();
    return;
  }

  // Restore isolated status of buses that were marked during island detection
  for (int i = 0; i < p_islandIsolatedBusIndices.size(); i++) {
    int localIdx = p_islandIsolatedBusIndices[i];
    gridpack::powerflow::PFBus *bus =
      dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(localIdx).get());
    bus->setIsolated(p_saveIslandIsolatedStatus[i]);
  }

  p_saveIslandIsolatedStatus.clear();
  p_islandIsolatedBusIndices.clear();
  p_islandCount = 0;
}

/**
 * Set voltage limits on all buses
 * @param Vmin lower bound on voltages
 * @param Vmax upper bound on voltages
 */
void gridpack::powerflow::PFFactoryModule::setVoltageLimits(double Vmin,
    double Vmax)
{
  int numBus = p_network->numBuses();
  int i;
  bool bus_ok = true;
  for (i=0; i<numBus; i++) {
    gridpack::powerflow::PFBus *bus =
      dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(i).get());
    bus->setVoltageLimits(Vmin,Vmax);
  }
}


/**
 * Check to see if there are any voltage violations in the network
 * @param minV maximum voltage limit
 * @param maxV maximum voltage limit
 * @return true if no violations found
 */
bool gridpack::powerflow::PFFactoryModule::checkVoltageViolations()
{
  int numBus = p_network->numBuses();
  int i;
  bool bus_ok = true;
  char buf[128];
  for (i=0; i<numBus; i++) {
    if (p_network->getActiveBus(i)) {
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>
        (p_network->getBus(i).get());
      if (!bus->getIgnore()) {
        if (!bus->checkVoltageViolation()) bus_ok = false;
      }
    }
  }
  return checkTrue(bus_ok);
}

/**
 * Check to see if there are any voltage violations in the network
 * @param area only check for voltage violations in this area
 * @return true if no violations found
 */
bool gridpack::powerflow::PFFactoryModule::checkVoltageViolations(
    int area)
{
  int numBus = p_network->numBuses();
  int i;
  bool bus_ok = true;
  for (i=0; i<numBus; i++) {
    if (p_network->getActiveBus(i)) {
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>
        (p_network->getBus(i).get());
      if (!bus->getIgnore() && bus->getArea() == area) {
        if (!bus->checkVoltageViolation()) {
          bus_ok = false;
          gridpack::powerflow::PFFactoryModule::Violation violation;
          int idx = bus->getOriginalIndex();
          violation.bus_violation = true;
          violation.line_violation = false;
          violation.bus1 = idx;
          violation.bus2 = -1;
          p_violations.push_back(violation);
        }
      }
    }
  }
  return checkTrue(bus_ok);
}

/**
 * Set "ignore" parameter on all buses with violations so that subsequent
 * checks are not counted as violations
 */
void gridpack::powerflow::PFFactoryModule::ignoreVoltageViolations()
{
  int numBus = p_network->numBuses();
  int i;
  for (i=0; i<numBus; i++) {
    if (p_network->getActiveBus(i)) {
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>
        (p_network->getBus(i).get());
      // Set ignore on buses WITH violations (checkVoltageViolation returns false when violated)
      if (!bus->checkVoltageViolation()) bus->setIgnore(true);
    }
  }
}


/**
 * Clear "ignore" parameter on all buses
 */
void gridpack::powerflow::PFFactoryModule::clearVoltageViolations()
{
  int numBus = p_network->numBuses();
  int i;
  for (i=0; i<numBus; i++) {
    if (p_network->getActiveBus(i)) {
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>
        (p_network->getBus(i).get());
      bus->setIgnore(false);
    }
  }
}

/**
 * Check to see if there are any line overload violations in the
 * network
 * @return true if no violations found
 */
bool gridpack::powerflow::PFFactoryModule::checkLineOverloadViolations()
{
  int numBranch = p_network->numBranches();
  int i;
  bool branch_ok = true;
  // p_rateB is legacy rtpr; treat it as "rating tier B" for one call.
  std::string savedRating = p_contingencyRating;
  if (p_rateB) p_contingencyRating = "B";
  for (i=0; i<numBranch; i++) {
    if (p_network->getActiveBranch(i)) {
      gridpack::powerflow::PFBranch *branch =
        dynamic_cast<gridpack::powerflow::PFBranch*>
        (p_network->getBranch(i).get());
      int nlines;
      p_network->getBranchData(i)->getValue(BRANCH_NUM_ELEMENTS,&nlines);
      std::vector<std::string> tags = branch->getLineTags();
      for (int k = 0; k<nlines; k++) {
        if (branch->getIgnore(tags[k])) continue;
        double rate = pickBranchRating(i, k);
        if (rate <= 0.0) continue;
        gridpack::ComplexType s = branch->getComplexPower(tags[k]);
        double pq = abs(s);
        if (pq > rate) {
          branch_ok = false;
          gridpack::powerflow::PFFactoryModule::Violation violation;
          violation.bus_violation = false;
          violation.line_violation = true;
          violation.bus1 = branch->getBus1OriginalIndex();
          violation.bus2 = branch->getBus2OriginalIndex();
          strncpy(violation.tag,tags[k].c_str(),2);
          violation.tag[2] = '\0';
          p_violations.push_back(violation);
        }
      }
    }
  }
  p_contingencyRating = savedRating;
  return checkTrue(branch_ok);
}

/**
 * Check to see if there are any line overload violations in the
 * network
 * @param area only check for voltage violations in this area
 * @return true if no violations found
 */
bool gridpack::powerflow::PFFactoryModule::checkLineOverloadViolations(int area)
{
  int numBranch = p_network->numBranches();
  int i;
  bool branch_ok = true;
  std::string savedRating = p_contingencyRating;
  if (p_rateB) p_contingencyRating = "B";
  for (i=0; i<numBranch; i++) {
    if (p_network->getActiveBranch(i)) {
      gridpack::powerflow::PFBranch *branch =
        dynamic_cast<gridpack::powerflow::PFBranch*>
        (p_network->getBranch(i).get());
      gridpack::powerflow::PFBus *bus1 =
        dynamic_cast<gridpack::powerflow::PFBus*>
        (branch->getBus1().get());
      gridpack::powerflow::PFBus *bus2 =
        dynamic_cast<gridpack::powerflow::PFBus*>
        (branch->getBus2().get());
      if (bus1->getArea() != area && bus2->getArea() != area) continue;
      int nlines;
      p_network->getBranchData(i)->getValue(BRANCH_NUM_ELEMENTS,&nlines);
      std::vector<std::string> tags = branch->getLineTags();
      for (int k = 0; k<nlines; k++) {
        if (branch->getIgnore(tags[k])) continue;
        double rate = pickBranchRating(i, k);
        if (rate <= 0.0) continue;
        gridpack::ComplexType s = branch->getComplexPower(tags[k]);
        double pq = abs(s);
        if (pq > rate) branch_ok = false;
      }
    }
  }
  p_contingencyRating = savedRating;
  return checkTrue(branch_ok);
}

/**
 * Check to see if there are any line overload violations on
 * specific lines.
 * @param bus1 original index of "from" bus for branch
 * @param bus2 original index of "to" bus for branch
 * @param tags line IDs for individual lines
 * @param violations false if violation detected on branch, true otherwise
 * @return true if no violations found
 */
bool gridpack::powerflow::PFFactoryModule::checkLineOverloadViolations(
    std::vector<int> &bus1, std::vector<int> &bus2,
    std::vector<std::string> &tags, std::vector<bool> &violations)
{
  bool branch_ok = true;
  int nbranch = bus1.size();
  if (nbranch != bus2.size() || nbranch != tags.size()) {
    printf("checkLineOverloadViolations: number of entries"
        " in bus1 and bus2 or bus1 and tags not equal\n");
    return false;
  }
  int i;
  violations.clear();
  std::vector<int> failure;
  failure.resize(nbranch);
  for (i=0; i<nbranch; i++) {
    failure[i] = 0;
    std::vector<int> indices = p_network->getLocalBranchIndices(bus1[i],bus2[i]);
    int j;
    for (j=0; j<indices.size(); j++) {
      gridpack::powerflow::PFBranch *branch = p_network->getBranch(indices[j]).get();
      // Loop over all lines in the branch and choose the smallest rating value
      int nlines;
      p_network->getBranchData(indices[j])->getValue(BRANCH_NUM_ELEMENTS,&nlines);
      std::vector<std::string> alltags = branch->getLineTags();
      double rate;
      for (int k = 0; k<nlines; k++) {
        if (tags[i] == alltags[k] && !branch->getIgnore(tags[k])) {
          bool foundRating=false;
          if (p_rateB) {
            if (p_network->getBranchData(indices[j])->getValue(BRANCH_RATING_B,
                  &rate,k)) {
              foundRating = true;
            } else {
              if (p_network->getBranchData(indices[j])->getValue(BRANCH_RATING_A,
                    &rate,k)) {
                foundRating = true;
              }
            }
          } else {
            if (p_network->getBranchData(indices[j])->getValue(BRANCH_RATING_A,
                  &rate,k)) {
              foundRating = true;
            }
          }
          if (p_network->getBranchData(indices[j])->getValue(BRANCH_RATING_A,
                &rate,k)) {
            if (rate > 0.0) {
              gridpack::ComplexType s = branch->getComplexPower(tags[k]);
              double pq = abs(s);
              if (pq > rate) {
                failure[i] = 1;
              }
            }
          }
        }
      }
    }
  }
  p_network->communicator().sum(&failure[0],nbranch);
  for (i=0; i<nbranch; i++) {
    if (failure[i] == 0) {
      violations.push_back(true);
    } else {
      violations.push_back(false);
      branch_ok = false;
    }
  }
  return branch_ok;
}

/**
 * Set "ignore" parameter on all lines with violations so that subsequent
 * checks are not counted as violations
 */
void gridpack::powerflow::PFFactoryModule::ignoreLineOverloadViolations()
{
  int numBranch = p_network->numBranches();
  int i;
  for (i=0; i<numBranch; i++) {
    if (p_network->getActiveBranch(i)) {
      gridpack::powerflow::PFBranch *branch =
        dynamic_cast<gridpack::powerflow::PFBranch*>
        (p_network->getBranch(i).get());
      // Loop over all lines in the branch and choose the smallest rating value
      int nlines;
      p_network->getBranchData(i)->getValue(BRANCH_NUM_ELEMENTS,&nlines);
      std::vector<std::string> tags = branch->getLineTags();
      double rateA;
      for (int k = 0; k<nlines; k++) {
        if (p_network->getBranchData(i)->getValue(BRANCH_RATING_A,&rateA,k)) {
          if (rateA > 0.0) {
            gridpack::ComplexType s = branch->getComplexPower(tags[k]);
            double pq = abs(s);
            if (pq > rateA) branch->setIgnore(tags[k],true);
          }
        }
      }
    }
  }
}

/**
 * Clear "ignore" parameter on all lines
 */
void gridpack::powerflow::PFFactoryModule::clearLineOverloadViolations()
{
  int numBranch = p_network->numBranches();
  int i;
  for (i=0; i<numBranch; i++) {
    if (p_network->getActiveBranch(i)) {
      gridpack::powerflow::PFBranch *branch =
        dynamic_cast<gridpack::powerflow::PFBranch*>
        (p_network->getBranch(i).get());
      // Loop over all lines in the branch and choose the smallest rating value
      int nlines;
      p_network->getBranchData(i)->getValue(BRANCH_NUM_ELEMENTS,&nlines);
      std::vector<std::string> tags = branch->getLineTags();
      double rateA;
      for (int k = 0; k<nlines; k++) {
        branch->setIgnore(tags[k],false);
      }
    }
  }
}

/**
 * Check to see if there are any voltage violations in the network
 * @return true if no violations found
 */
bool gridpack::powerflow::PFFactoryModule::checkQlimViolations()
{
  int numBus = p_network->numBuses();
  int i;
  bool bus_ok = true;
  for (i=0; i<numBus; i++) {
    if (p_network->getActiveBus(i)) {
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>
        (p_network->getBus(i).get());
      if (bus->chkQlim(p_qlim_deadband)) {
        bus_ok = false;
      }
    }
  }
  p_network->updateBuses();
  for (i=0; i<numBus; i++) {
    if (!p_network->getActiveBus(i)) {
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>
        (p_network->getBus(i).get());
      bus->pushIsPV();
    }
  }
  return checkTrue(bus_ok);
}

/**
 * Check to see if there are any voltage violations in the network
 * @param area only check for voltage violations in this area
 * @return true if no violations found
 */
bool gridpack::powerflow::PFFactoryModule::checkQlimViolations(int area)
{
  int numBus = p_network->numBuses();
  int i;
  bool bus_ok = true;
  for (i=0; i<numBus; i++) {
    if (p_network->getActiveBus(i)) {
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>
        (p_network->getBus(i).get());
      if (bus->getArea() == area) {
        if (!bus->chkQlim()) bus_ok = false;
      }
    }
  }
  p_network->updateBuses();
  for (i=0; i<numBus; i++) {
    if (!p_network->getActiveBus(i)) {
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>
        (p_network->getBus(i).get());
      bus->pushIsPV();
    }
  }
  return checkTrue(bus_ok);
}

/**
 * Adjust voltage setpoints for remote bus voltage regulation (IREG).
 * For each PV bus with generators that have IREG != 0 and IREG != own bus,
 * adjust the local bus voltage so that the remote bus voltage matches VS.
 * @param tol tolerance for remote voltage error (default 1e-4 pu)
 * @return true if all remote regulations are satisfied within tolerance
 */
bool gridpack::powerflow::PFFactoryModule::adjustRemoteRegulation(double tol)
{
  int numBus = p_network->numBuses();
  bool all_ok = true;

  for (int i = 0; i < numBus; i++) {
    if (!p_network->getActiveBus(i)) continue;
    gridpack::powerflow::PFBus *bus =
      dynamic_cast<gridpack::powerflow::PFBus*>(
        p_network->getBus(i).get());
    if (!bus->isPV()) continue;

    // Skip IREG PV buses — voltage regulation handled in augmented Jacobian
    if (bus->isIREG_PV()) continue;

    int orig_idx = bus->getOriginalIndex();
    int ngen = bus->getNumGenerators();

    // If any online generator has local regulation (IREG=0), local voltage
    // control takes precedence.  Do not override the bus terminal voltage via
    // the remote-regulation outer loop; the remote-reg generator's Q
    // contribution is handled implicitly by the NR Jacobian.
    bool has_local_reg = false;
    for (int j = 0; j < ngen; j++) {
      if (bus->getGenStatusByIdx(j) == 1 && bus->getIREG(j) == 0) {
        has_local_reg = true;
        break;
      }
    }
    if (has_local_reg) continue;

    // Find first online generator with remote regulation
    int ireg_bus = 0;
    double vs_target = 0.0;
    for (int j = 0; j < ngen; j++) {
      int ireg = bus->getIREG(j);
      if (bus->getGenStatusByIdx(j) == 1 && ireg != 0 && ireg != orig_idx) {
        ireg_bus = ireg;
        vs_target = bus->getVSByIdx(j);
        break;
      }
    }
    if (ireg_bus == 0) continue;  // No remote regulation on this bus

    // Look up remote bus voltage
    std::vector<int> remote_indices = p_network->getLocalBusIndices(ireg_bus);
    if (remote_indices.empty()) continue;  // Remote bus not in local partition

    gridpack::powerflow::PFBus *remote_bus =
      dynamic_cast<gridpack::powerflow::PFBus*>(
        p_network->getBus(remote_indices[0]).get());
    double v_remote = remote_bus->getVoltage();

    double dv = vs_target - v_remote;
    if (fabs(dv) > tol) {
      // Apply damping to prevent NR divergence when Q-limits interact
      // with voltage adjustments at multiple generators simultaneously
      // Limit step to prevent Q explosion when
      // multiple PV buses lose voltage control and cause large voltage deviations
      const double MAX_DV = 0.05;
      if (dv > MAX_DV) dv = MAX_DV;
      else if (dv < -MAX_DV) dv = -MAX_DV;
      bus->adjustVoltageForRemoteReg(dv);
      all_ok = false;
    }
  }

  return checkTrue(all_ok);
}

/**
 * Set up IREG remote voltage regulation via PV swap.
 * For each generator bus with IREG, make the remote bus PV (V=VS)
 * and the generator bus PQ (V free, Q from initial dispatch).
 * Must be called after load() and setExchange().
 */
void gridpack::powerflow::PFFactoryModule::setupIREGPointers()
{
  int numBus = p_network->numBuses();
  MPI_Comm comm = static_cast<MPI_Comm>(p_network->communicator());
  int nprocs = p_network->communicator().size();

  // Pass 1: Collect IREG swap requests from local active buses.
  // Also handle slack bus auto-correction.
  std::vector<int> swap_gen, swap_remote;
  std::vector<double> swap_vs;

  for (int i = 0; i < numBus; i++) {
    if (!p_network->getActiveBus(i)) continue;
    gridpack::powerflow::PFBus *bus =
      dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(i).get());

    // Slack bus cannot regulate a remote bus — auto-correct to local
    if (bus->getReferenceBus()) {
      int ngen = bus->getNumGenerators();
      int orig_idx = bus->getOriginalIndex();
      for (int j = 0; j < ngen; j++) {
        int ireg = bus->getIREG(j);
        if (bus->getGenStatusByIdx(j) == 1 && ireg != 0 && ireg != orig_idx) {
          printf("Warning: Slack bus %d has remote regulation of bus %d. "
                 "Slack bus can regulate itself only. Setting local regulation.\n",
                 orig_idx, ireg);
          gridpack::component::DataCollection *data = p_network->getBusData(i).get();
          double v_init = 1.0;
          data->getValue(BUS_VOLTAGE_MAG, &v_init);
          bus->setVoltageMag(v_init);
          break;
        }
      }
      continue;
    }

    if (!bus->isPV()) continue;
    int remote_bus_num = bus->getIREGRemoteBus();
    if (remote_bus_num == 0) continue;

    swap_gen.push_back(bus->getOriginalIndex());
    swap_remote.push_back(remote_bus_num);
    swap_vs.push_back(bus->getIREGVS());
  }

  // Gather all swap requests across processes
  int nlocal = swap_gen.size();
  int ntotal = 0;
  MPI_Allreduce(&nlocal, &ntotal, 1, MPI_INT, MPI_SUM, comm);

  if (ntotal == 0) return;

  std::vector<int> all_gen(ntotal), all_remote(ntotal);
  std::vector<double> all_vs(ntotal);
  std::vector<int> counts(nprocs), displs(nprocs);
  MPI_Allgather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
  displs[0] = 0;
  for (int p = 1; p < nprocs; p++) displs[p] = displs[p-1] + counts[p-1];

  MPI_Allgatherv(swap_gen.data(), nlocal, MPI_INT,
                 all_gen.data(), counts.data(), displs.data(), MPI_INT, comm);
  MPI_Allgatherv(swap_remote.data(), nlocal, MPI_INT,
                 all_remote.data(), counts.data(), displs.data(), MPI_INT, comm);
  MPI_Allgatherv(swap_vs.data(), nlocal, MPI_DOUBLE,
                 all_vs.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);

  // Pass 2: Apply swaps — each process handles buses it owns
  for (int s = 0; s < ntotal; s++) {
    // Set generator bus to PQ (active and ghost copies)
    std::vector<int> gindices = p_network->getLocalBusIndices(all_gen[s]);
    for (size_t g = 0; g < gindices.size(); g++) {
      gridpack::powerflow::PFBus *gbus =
        dynamic_cast<gridpack::powerflow::PFBus*>(
            p_network->getBus(gindices[g]).get());
      gbus->setIsPV(false);
      gbus->saveIsPVState();
    }

    // Set remote bus to PV at VS (active and ghost copies)
    std::vector<int> rindices = p_network->getLocalBusIndices(all_remote[s]);
    for (size_t r = 0; r < rindices.size(); r++) {
      gridpack::powerflow::PFBus *rbus =
        dynamic_cast<gridpack::powerflow::PFBus*>(
            p_network->getBus(rindices[r]).get());
      if (rbus->isPV() || rbus->getReferenceBus()) continue;
      rbus->setIsPV(true);
      rbus->saveIsPVState();
      rbus->setVoltageForIREG(all_vs[s]);
    }
  }
}

/**
 * Clear changes that were made for Q limit violations and reset
 * system to its original state
 */
void gridpack::powerflow::PFFactoryModule::clearQlimViolations()
{
  int numBus = p_network->numBuses();
  int i;
  bool bus_ok = true;
  for (i=0; i<numBus; i++) {
    if (p_network->getActiveBus(i)) {
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>
        (p_network->getBus(i).get());
      bus->clearQlim();
    }
  }
  p_network->updateBuses();
  for (i=0; i<numBus; i++) {
    if (!p_network->getActiveBus(i)) {
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>
        (p_network->getBus(i).get());
      bus->pushIsPV();
    }
  }
}

/**
 * Reinitialize voltages
 */
void gridpack::powerflow::PFFactoryModule::resetVoltages()
{
  int numBus = p_network->numBuses();
  int i;
  for (i=0; i<numBus; i++) {
    if (p_network->getActiveBus(i)) {
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>
        (p_network->getBus(i).get());
      bus->resetVoltage();
    }
  }
}

/**
 * Set the initial start mode for power flow solver
 */
void gridpack::powerflow::PFFactoryModule::setInitStartMode(InitStartMode mode)
{
  gridpack::powerflow::PFBus::setInitStartMode(mode);
}

/**
 * Scale generator real power. If zone less than 1 then scale all
 * generators in the area.
 * @param scale factor to scale real power generation
 * @param area index of area for scaling generation
 * @param zone index of zone for scaling generation
 */
void gridpack::powerflow::PFFactoryModule::scaleGeneratorRealPower(
    double scale, int area, int zone)
{
  int nbus = p_network->numBuses();
  int i, izone;
  for (i=0; i<nbus; i++) {
    gridpack::powerflow::PFBus *bus = p_network->getBus(i).get();
    if (zone > 0) {
      izone = bus->getZone();
    } else {
      izone = zone;
    }
    if (bus->getArea() == area && zone == izone) {
      std::vector<std::string> tags = bus->getGenerators();
      int j;
      for (j=0; j<tags.size(); j++) {
        bus->scaleGeneratorRealPower(tags[j],scale);
      }
    }
  }
}

/**
 * Scale load real power. If zone less than 1 then scale all
 * loads in the area.
 * @param scale factor to scale load real power
 * @param area index of area for scaling load
 * @param zone index of zone for scaling load
 */
void gridpack::powerflow::PFFactoryModule::scaleLoadPower(
    double scale, int area, int zone)
{
  int nbus = p_network->numBuses();
  int i, izone;
  for (i=0; i<nbus; i++) {
    gridpack::powerflow::PFBus *bus = p_network->getBus(i).get();
    if (zone > 0) {
      izone = bus->getZone();
    } else {
      izone = zone;
    }
    if (bus->getArea() == area && zone == izone) {
      std::vector<std::string> tags = bus->getLoads();
      int j;
      for (j=0; j<tags.size(); j++) {
        bus->scaleLoadPower(tags[j],scale);
      }
    }
  }
}

/**
 * Return the total real power load for all loads in the zone. If zone
 * less than 1, then return the total load for the area
 * @param area index of area
 * @param zone index of zone
 * @return total load
 */
double gridpack::powerflow::PFFactoryModule::getTotalLoadRealPower(int area, int zone)
{
  double ret = 0.0;
  int nbus = p_network->numBuses();
  int i, j, izone;
  for (i=0; i<nbus; i++) {
    gridpack::powerflow::PFBus *bus = p_network->getBus(i).get();
    if (zone > 0) {
      izone = bus->getZone();
    } else {
      izone = zone;
    }
    if (bus->getArea() == area && zone == izone) {
      std::vector<std::string> tags;
      std::vector<double> pl;
      std::vector<double> ql;
      std::vector<int> status;
      bus->getLoadPower(tags,pl,ql,status);
      for (j=0; j<tags.size(); j++) {
        if (static_cast<bool>(status[j])) {
          ret += pl[j];
        }
      }
    }
  }
  p_network->communicator().sum(&ret,1);
  return ret;
}

/**
 * Return the current real power generation and the maximum and minimum total
 * power generation for all generators in the zone. If zone is less than 1
 * then return values for all generators in the area
 * @param area index of area
 * @param zone index of zone
 * @param total total real power generation
 * @param pmin minimum allowable real power generation
 * @param pmax maximum available real power generation
 */
void gridpack::powerflow::PFFactoryModule::getGeneratorMargins(int area, int zone,
    double *total, double *pmin, double *pmax)
{
  *total = 0.0;
  *pmin = 0.0;
  *pmax = 0.0;
  int nbus = p_network->numBuses();
  int i, j, izone;
  for (i=0; i<nbus; i++) {
    gridpack::powerflow::PFBus *bus = p_network->getBus(i).get();
    if (zone > 0) {
      izone = bus->getZone();
    } else {
      izone = zone;
    }
    if (bus->getArea() == area && zone == izone) {
      std::vector<std::string> tags;
      std::vector<double> tcurrent;
      std::vector<double> tpmin;
      std::vector<double> tpmax;
      std::vector<int> status;
      bus->getGeneratorMargins(tags,tcurrent,tpmin,tpmax,status);
      for (j=0; j<tcurrent.size(); j++) {
        if (status[j] != 0) {
          *total += tcurrent[j];
          *pmin += tpmin[j];
          *pmax += tpmax[j];
        }
      }
    }
  }
}

/**
 * Reset power of loads and generators to original values
 */
void gridpack::powerflow::PFFactoryModule::resetPower()
{
  int nbus = p_network->numBuses();
  int i;
  for (i=0; i<nbus; i++) {
    p_network->getBus(i)->resetPower();
  }
}

/**
 * Set parameters for real time path rating diagnostics
 * @param src_area generation area
 * @param src_zone generation zone
 * @param load_area load area
 * @param load_zone load zone
 * @param gen_scale scale factor for generation
 * @param load_scale scale factor for loads
 */
void gridpack::powerflow::PFFactoryModule::setRTPRParams(
    int src_area, int src_zone, int load_area,
    int load_zone, double gen_scale, double load_scale)
{
  int nbus = p_network->numBuses();
  int i, j, izone;
  for (i=0; i<nbus; i++) {
    gridpack::powerflow::PFBus *bus = p_network->getBus(i).get();
    int tarea = bus->getArea();
    int tzone = bus->getZone();
    if (src_zone > 0) {
      izone = tzone;
    } else {
      izone = src_zone;
    }
    bus->setScale(1.0);
    if (tarea == src_area && src_zone == izone) {
      bus->setSource(true);
      bus->setScale(gen_scale);
    } else {
      bus->setSource(false);
    }
    if (load_zone > 0) {
      izone = tzone;
    } else {
      izone = load_zone;
    }
    if (tarea == load_area && load_zone == izone) {
      bus->setSink(true);
      bus->setScale(load_scale);
    } else {
      bus->setSink(false);
    }
  }
}

/**
 * Return vector describing all violations
 * @return violation vector
 */
std::vector<gridpack::powerflow::PFFactoryModule::Violation>
gridpack::powerflow::PFFactoryModule::getViolations()
{
  std::vector<gridpack::powerflow::PFFactoryModule::Violation> ret;
  gridpack::parallel::GlobalVector<gridpack::powerflow::PFFactoryModule::Violation>
    sumVec(p_network->communicator());
  int nproc = p_network->communicator().size();
  int me = p_network->communicator().rank();
  std::vector<int> sizes(nproc);
  int i;
  for (i=0; i<nproc; i++) sizes[i] = 0;
  sizes[me] = p_violations.size();

  p_network->communicator().sum(&sizes[0],nproc);
  int offset = 0;
  for (i=1; i<me; i++) offset += sizes[i];
  int total = 0;
  for (i=0; i<nproc; i++) total += sizes[i];
  if (total == 0) return ret;
  std::vector<int> idx;
  int last = offset+sizes[me];
  for (i=offset; i<last; i++) idx.push_back(i);
  sumVec.addElements(idx,p_violations);
  sumVec.upload();
  sumVec.getAllData(ret);
  return ret;
}

/**
 * Clear violation vector
 */
void gridpack::powerflow::PFFactoryModule::clearViolations()
{
  p_violations.clear();
}

/**
 * User rate B parameter for line overload violations
 * @param flag if true, use RATEB parameter
 */
void gridpack::powerflow::PFFactoryModule::useRateB(bool flag)
{
  if (flag) {
    p_rateB = true;
  } else {
    p_rateB = false;
  }
}

/**
 * Select rating tier for overload checks.
 */
void gridpack::powerflow::PFFactoryModule::setContingencyRating(
    const std::string& rating)
{
  if (rating == "A" || rating == "B" || rating == "C") {
    p_contingencyRating = rating;
  } else {
    p_contingencyRating = "A";
  }
  // Keep PFBranch's serialWrite("flow",...) denominator in sync so the .out
  // file's loading% matches _violations.csv / JSON loading_percent.
  gridpack::powerflow::PFBranch::setContingencyRating(p_contingencyRating);
}

/**
 * Rating for one line element under the current contingency tier
 * with A->B->C fallback when the picked tier is zero/missing.
 */
double gridpack::powerflow::PFFactoryModule::pickBranchRating(
    int branchLocalIdx, int elemIdx) const
{
  double a = 0.0, b = 0.0, c = 0.0;
  p_network->getBranchData(branchLocalIdx)->getValue(BRANCH_RATING_A, &a, elemIdx);
  p_network->getBranchData(branchLocalIdx)->getValue(BRANCH_RATING_B, &b, elemIdx);
  p_network->getBranchData(branchLocalIdx)->getValue(BRANCH_RATING_C, &c, elemIdx);
  if (p_contingencyRating == "A") return a;
  if (p_contingencyRating == "B") return (b > 0.0) ? b : a;
  // "C"
  if (c > 0.0) return c;
  if (b > 0.0) return b;
  return a;
}

/**
 * Check switched shunt violations and adjust shunt B values.
 * For buses with SWREM != 0, resolves remote bus voltage via getLocalBusIndices.
 * @return true if no violations found (all voltages within deadband)
 */
bool PFFactoryModule::checkSwitchedShuntViolations()
{
  int numBus = p_network->numBuses();
  int i;
  bool bus_ok = true;
  for (i = 0; i < numBus; i++) {
    if (!p_network->getActiveBus(i)) continue;
    gridpack::powerflow::PFBus *bus =
      dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(i).get());
    if (!bus->hasSwitchedShunt()) continue;

    // Determine controlled voltage
    double v_controlled = bus->getVoltage();  // Default: local control
    int swrem = bus->getSwitchedShuntRemoteBus();
    if (swrem > 0) {
      // Remote control: look up voltage at remote bus
      std::vector<int> rindices = p_network->getLocalBusIndices(swrem);
      if (rindices.size() > 0) {
        gridpack::powerflow::PFBus *rbus =
          dynamic_cast<gridpack::powerflow::PFBus*>(
              p_network->getBus(rindices[0]).get());
        v_controlled = rbus->getVoltage();
      }
    }

    if (bus->adjustSwitchedShunt(v_controlled)) {
      bus_ok = false;
    }
  }
  return checkTrue(bus_ok);
}

/**
 * Clear switched shunt adjustments and reset to BINIT state
 */
void PFFactoryModule::clearSwitchedShunts()
{
  int numBus = p_network->numBuses();
  for (int i = 0; i < numBus; i++) {
    if (!p_network->getActiveBus(i)) continue;
    gridpack::powerflow::PFBus *bus =
      dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(i).get());
    bus->resetSwitchedShunt();
  }
}

/**
 * Check LTC violations and adjust transformer tap ratios.
 * Iterates over branches, finds LTC-controlled transformers,
 * looks up controlled bus voltage, and adjusts tap if outside deadband.
 * @return true if no violations found
 */
bool PFFactoryModule::checkLTCViolations()
{
  int numBranch = p_network->numBranches();
  bool branch_ok = true;
  for (int i = 0; i < numBranch; i++) {
    if (!p_network->getActiveBranch(i)) continue;
    gridpack::powerflow::PFBranch *branch =
      dynamic_cast<gridpack::powerflow::PFBranch*>(p_network->getBranch(i).get());
    if (!branch->hasLTC()) continue;

    // Look up voltage at controlled bus
    int cont = branch->getLTCControlledBus();
    double v_controlled = 1.0;
    std::vector<int> rindices = p_network->getLocalBusIndices(cont);
    if (rindices.size() > 0) {
      gridpack::powerflow::PFBus *cbus =
        dynamic_cast<gridpack::powerflow::PFBus*>(
            p_network->getBus(rindices[0]).get());
      v_controlled = cbus->getVoltage();
    }

    if (branch->adjustLTC(v_controlled)) {
      branch_ok = false;
    }
  }
  return checkTrue(branch_ok);
}

/**
 * Clear LTC adjustments and reset taps to initial values
 */
void PFFactoryModule::clearLTCControls()
{
  int numBranch = p_network->numBranches();
  for (int i = 0; i < numBranch; i++) {
    if (!p_network->getActiveBranch(i)) continue;
    gridpack::powerflow::PFBranch *branch =
      dynamic_cast<gridpack::powerflow::PFBranch*>(p_network->getBranch(i).get());
    branch->resetLTC();
  }
}

/**
 * Compute net MW export for each area via tie-line flows.
 * For each branch connecting buses in different areas, the real power
 * flow from the "from" bus side is counted as export for that bus's area.
 * @param areaExport map from area number to net MW export (positive = export)
 */
void PFFactoryModule::computeAreaExport(std::map<int,double> &areaExport)
{
  areaExport.clear();
  int numBranch = p_network->numBranches();
  for (int i = 0; i < numBranch; i++) {
    if (!p_network->getActiveBranch(i)) continue;
    gridpack::powerflow::PFBranch *branch =
      dynamic_cast<gridpack::powerflow::PFBranch*>(p_network->getBranch(i).get());

    // Get endpoint buses
    int idx1, idx2;
    p_network->getBranchEndpoints(i, &idx1, &idx2);
    gridpack::powerflow::PFBus *bus1 =
      dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(idx1).get());
    gridpack::powerflow::PFBus *bus2 =
      dynamic_cast<gridpack::powerflow::PFBus*>(p_network->getBus(idx2).get());

    int area1 = bus1->getArea();
    int area2 = bus2->getArea();
    if (area1 == area2) continue;  // Not a tie-line

    // Sum power flow for each line element on this branch
    std::vector<std::string> tags = branch->getLineIDs();
    for (size_t j = 0; j < tags.size(); j++) {
      if (!branch->getBranchStatus(tags[j])) continue;
      gridpack::ComplexType s = branch->getComplexPower(tags[j]);
      double p_mw = real(s);  // MW flowing from bus1 to bus2
      areaExport[area1] += p_mw;   // Export from area1
      areaExport[area2] -= p_mw;   // Import to area2
    }
  }
}

// -------------------------------------------------------------
// Extensions for the GPU batch contingency path
// -------------------------------------------------------------

/**
 * Select the Jacobian layout on every bus and branch
 */
void PFFactoryModule::setJacobianFormulation(JacobianFormulation form)
{
  int numBus = p_network->numBuses();
  for (int i = 0; i < numBus; i++) {
    dynamic_cast<PFBus*>(p_network->getBus(i).get())
      ->setJacobianFormulation(form);
  }
  int numBranch = p_network->numBranches();
  for (int i = 0; i < numBranch; i++) {
    dynamic_cast<PFBranch*>(p_network->getBranch(i).get())
      ->setJacobianFormulation(form);
  }
}

/**
 * Report what the last setContingency() did to topology and the slack
 */
void PFFactoryModule::getContingencyEffects(ContingencyEffects *effects) const
{
  effects->island_count = p_islandCount;
  effects->lone_bus = p_hasLoneBus;
  effects->slack_transferred = p_slackTransferred;
  effects->slack_bus = p_currentSlackBusIdx;
  effects->isolated_buses = p_loneBusIndices;
  effects->isolated_buses.insert(effects->isolated_buses.end(),
      p_islandIsolatedBusIndices.begin(), p_islandIsolatedBusIndices.end());
}

/**
 * Superset bus type of a bus in its current state. Isolation wins over the
 * reference role, which wins over voltage control.
 */
int PFFactoryModule::supersetType(PFBus *bus)
{
  if (bus->isIsolated()) return SUPERSET_ISOLATED;
  if (bus->getReferenceBus()) return SUPERSET_REF;
  if (bus->isPV()) return SUPERSET_PV;
  return SUPERSET_PQ;
}

/**
 * PFBus::chkQlim() never converts a bus that has no generators. Such a bus
 * can still be PV: when a generator regulates the voltage of another bus,
 * setupIREGPointers() makes that other bus PV at the generator's set point
 * (and the generator's own bus PQ). Exporting unbounded limits for it gives
 * the batch path's check the same outcome. Without this, the check would
 * see zero capability and convert every remotely regulated bus.
 */
void PFFactoryModule::qlimBounds(PFBus *bus, double *qmax, double *qmin)
{
  if (bus->getNumGenerators() == 0) {
    *qmax = std::numeric_limits<double>::infinity();
    *qmin = -std::numeric_limits<double>::infinity();
    return;
  }
  bus->getOnlineGenQLimits(qmax, qmin);
}

/**
 * Sum of the admittance of all branch objects joining local buses k and m,
 * seen from k. Uses the values cached by the last PFBranch::setYBus().
 */
void PFFactoryModule::pairAdmittance(int k, int m, double *g, double *b)
{
  *g = 0.0;
  *b = 0.0;
  std::vector<int> nghbrs = p_network->getConnectedBranches(k);
  for (size_t j = 0; j < nghbrs.size(); j++) {
    int idx1, idx2;
    p_network->getBranchEndpoints(nghbrs[j], &idx1, &idx2);
    PFBranch *branch =
      dynamic_cast<PFBranch*>(p_network->getBranch(nghbrs[j]).get());
    if (!branch->isActiveAtLoad()) continue;
    gridpack::ComplexType y;
    if (idx1 == k && idx2 == m) {
      y = branch->getForwardYBus();
    } else if (idx2 == k && idx1 == m) {
      y = branch->getReverseYBus();
    } else {
      continue;
    }
    *g += real(y);
    *b += imag(y);
  }
}

/**
 * Export the network as it stands now (see header). Rows list the edges of
 * a bus in the order of its connected branches, which is the order
 * PFBus::rhsValues() sums them in.
 */
void PFFactoryModule::exportSupersetModel(SupersetModel *model)
{
  setYBus();
  setSBus();
  int numBus = p_network->numBuses();
  int numBranch = p_network->numBranches();
  model->sbase = 0.0;
  model->buses.clear();
  model->buses.resize(numBus);
  model->has_remote_regulation = false;
  model->has_switched_shunt = false;
  model->has_ltc = false;
  for (int i = 0; i < numBus; i++) {
    PFBus *bus = dynamic_cast<PFBus*>(p_network->getBus(i).get());
    SupersetBus &sb = model->buses[i];
    model->sbase = bus->getSBase();
    sb.original_index = bus->getOriginalIndex();
    sb.type = supersetType(bus);
    gridpack::ComplexType y = bus->getYBus();
    sb.g_diag = real(y);
    sb.b_diag = imag(y);
    bus->getScheduledInjection(&sb.p0, &sb.q0);
    sb.v_init = bus->getInitialVoltage();
    sb.theta_init = bus->getInitialAngle();
    sb.v_solved = bus->getVoltage();
    sb.theta_solved = bus->getPhase();
    bus->getOnlineLoadTotals(&sb.pl, &sb.ql, &sb.ip, &sb.iq, &sb.yp, &sb.yq);
    qlimBounds(bus, &sb.qmax, &sb.qmin);
    sb.remote_regulation = bus->hasActiveRemoteRegulation();
    sb.switched_shunt = bus->hasSwitchedShunt();
    if (sb.remote_regulation) model->has_remote_regulation = true;
    if (sb.switched_shunt) model->has_switched_shunt = true;
  }
  for (int i = 0; i < numBranch; i++) {
    PFBranch *branch = dynamic_cast<PFBranch*>(p_network->getBranch(i).get());
    if (branch->hasLTC()) model->has_ltc = true;
  }

  // Edges: one per directed bus pair joined by an in-service branch object
  model->row_start.assign(numBus + 1, 0);
  model->edge_col.clear();
  model->edge_g.clear();
  model->edge_b.clear();
  std::vector<std::map<int,int> > edge_of(numBus);
  for (int k = 0; k < numBus; k++) {
    model->row_start[k] = static_cast<int>(model->edge_col.size());
    std::vector<int> nghbrs = p_network->getConnectedBranches(k);
    for (size_t j = 0; j < nghbrs.size(); j++) {
      PFBranch *branch =
        dynamic_cast<PFBranch*>(p_network->getBranch(nghbrs[j]).get());
      if (!branch->isActiveAtLoad()) continue;
      int idx1, idx2;
      p_network->getBranchEndpoints(nghbrs[j], &idx1, &idx2);
      int m = (idx1 == k) ? idx2 : idx1;
      if (m == k || edge_of[k].count(m) > 0) continue;
      edge_of[k][m] = static_cast<int>(model->edge_col.size());
      double g, b;
      pairAdmittance(k, m, &g, &b);
      model->edge_col.push_back(m);
      model->edge_g.push_back(g);
      model->edge_b.push_back(b);
    }
  }
  model->row_start[numBus] = static_cast<int>(model->edge_col.size());
  model->edge_mate.assign(model->edge_col.size(), -1);
  for (int k = 0; k < numBus; k++) {
    for (int e = model->row_start[k]; e < model->row_start[k+1]; e++) {
      model->edge_mate[e] = edge_of[model->edge_col[e]][k];
    }
  }
  model->branch_edge.assign(numBranch, -1);
  for (int i = 0; i < numBranch; i++) {
    PFBranch *branch = dynamic_cast<PFBranch*>(p_network->getBranch(i).get());
    if (!branch->isActiveAtLoad()) continue;
    int idx1, idx2;
    p_network->getBranchEndpoints(i, &idx1, &idx2);
    if (idx1 == idx2) continue;
    model->branch_edge[i] = edge_of[idx1][idx2];
  }
}

/**
 * Read back absolute values of selected buses and branches (see header)
 */
void PFFactoryModule::captureCaseState(const SupersetModel &model,
    const std::vector<int> &buses, const std::vector<int> &branches,
    SupersetCaseState *state)
{
  state->buses.clear();
  state->edges.clear();
  std::vector<int> bus_list = buses;
  for (size_t j = 0; j < branches.size(); j++) {
    int i = branches[j];
    dynamic_cast<PFBranch*>(p_network->getBranch(i).get())->setYBus();
    int idx1, idx2;
    p_network->getBranchEndpoints(i, &idx1, &idx2);
    bus_list.push_back(idx1);
    bus_list.push_back(idx2);
  }
  std::sort(bus_list.begin(), bus_list.end());
  bus_list.erase(std::unique(bus_list.begin(), bus_list.end()),
      bus_list.end());
  for (size_t j = 0; j < bus_list.size(); j++) {
    PFBus *bus = dynamic_cast<PFBus*>(p_network->getBus(bus_list[j]).get());
    bus->setYBus();
    bus->setSBus();
    SupersetBusUpdate u;
    u.bus = bus_list[j];
    u.type = supersetType(bus);
    gridpack::ComplexType y = bus->getYBus();
    u.g_diag = real(y);
    u.b_diag = imag(y);
    bus->getScheduledInjection(&u.p0, &u.q0);
    qlimBounds(bus, &u.qmax, &u.qmin);
    u.remote_regulation = bus->hasActiveRemoteRegulation();
    state->buses.push_back(u);
  }
  for (size_t j = 0; j < branches.size(); j++) {
    int e = model.branch_edge[branches[j]];
    if (e < 0) continue;
    int idx1, idx2;
    p_network->getBranchEndpoints(branches[j], &idx1, &idx2);
    SupersetEdgeUpdate f, r;
    f.edge = e;
    pairAdmittance(idx1, idx2, &f.g, &f.b);
    r.edge = model.edge_mate[e];
    pairAdmittance(idx2, idx1, &r.g, &r.b);
    state->edges.push_back(f);
    state->edges.push_back(r);
  }
}

} // namespace powerflow
} // namespace gridpack
