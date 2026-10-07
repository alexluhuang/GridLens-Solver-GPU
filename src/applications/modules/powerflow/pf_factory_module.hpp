/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   pf_factory_module.hpp
 * @author Bruce Palmer
 * @date   2014-01-28 11:33:42 d3g096
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
 * - Added setupIREGPointers() for IREG PV bus swap with MPI support
 * @date  2026-04-05
 *
 * @brief
 *
 *
 */
// -------------------------------------------------------------

#ifndef _pf_factory_module_h_
#define _pf_factory_module_h_

#include "boost/smart_ptr/shared_ptr.hpp"
#include "gridpack/network/base_network.hpp"
#include "gridpack/factory/base_factory.hpp"
#include "gridpack/applications/components/pf_matrix/pf_components.hpp"
#include "gridpack/applications/modules/powerflow/pf_hvdc.hpp"
#include "gridpack/applications/modules/powerflow/pf_superset_model.hpp"

namespace gridpack {
namespace powerflow {

/// The type of network used in the powerflow application
typedef gridpack::network::BaseNetwork<PFBus, PFBranch > PFNetwork;

class PFFactoryModule
  : public gridpack::factory::BaseFactory<PFNetwork> {
  public:
    /**
     * Struct for storing information on violations in contingency calculations
     */
    struct Violation {
      bool line_violation; /* branch overload violation */
      bool bus_violation;  /* bus voltage violation */
      int bus1;  /* bus ID for voltage violations or from bus for branch violations */
      int bus2;  /* to bus ID for branch violations */
      char tag[3]; /* 2 character identifier */
    };
    /**
     * Basic constructor
     * @param network: network associated with factory
     */
    PFFactoryModule(NetworkPtr network);

    /**
     * Basic destructor
     */
    ~PFFactoryModule();

    /**
     * Create the admittance (Y-Bus) matrix
     */
    void setYBus(void);

    /**
     * Make SBus vector 
     */
    void setSBus(void);

    /**
      * Update pg of specified bus element based on their genID
      * @param name 
      * @param busID
      * @param genID
      * @param value
      */
//    void updatePg(std::string &name, int busID, std::string genID, double value);
    void updatePg(int busID, std::string genID, double value);
    void updateQg(int busID, std::string genID, double value);

    /**
     * Create the PQ 
     */
    void setPQ(void);

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
    bool checkLoneBus(std::ofstream *stream = NULL);

    /**
     * Set lone buses back to their original status.
     */
    void clearLoneBus();

    /**
     * Detect islands (disconnected subnetworks) in the network using BFS.
     * Mark buses in smaller islands as isolated to prevent singular Jacobian.
     * @param stream optional stream pointer for printing island info
     * @return number of islands found (1 = connected network, >1 = islanding)
     */
    int detectIslands(std::ofstream *stream = NULL);

    /**
     * Get the number of islands detected in the last call to detectIslands
     * @return number of islands (0 if detectIslands not called)
     */
    int getIslandCount() const;

    /**
     * Check if any lone buses were found in the last call to checkLoneBus
     * @return true if at least one lone bus was found
     */
    bool hasLoneBus() const;

    /**
     * Check if the reference (slack) bus has an online generator.
     * If not, transfer the slack function to the bus with the largest
     * online generator capacity.
     * @return true if a valid slack bus exists (or was transferred),
     *         false if no generator with real power capacity is available
     */
    bool checkAndTransferSlack();

    /**
     * Restore the original slack bus after a contingency.
     * Called by clearIslands() or unSetContingency().
     */
    void restoreSlack();

    /**
     * Check if slack bus generator output exceeds its capacity (Pmax).
     * Should be called after power flow solve.
     * @return true if within limits, false if Pgen > Pmax
     */
    bool checkSlackCapacity();

    /**
     * Clear island detection state and restore isolated status of buses
     * that were marked as isolated due to islanding
     */
    void clearIslands();

    /**
     * Set voltage limits on all buses
     * @param Vmin lower bound on voltages
     * @param Vmax upper bound on voltages
     */
    void setVoltageLimits(double Vmin, double Vmax);

    /**
     * Check to see if there are any voltage violations in the network
     * @param area only check for voltage violations in this area
     * @return true if no violations found
     */
    bool checkVoltageViolations();
    bool checkVoltageViolations(int area);

    /**
     * Set Q limit deadband (Mvar). PV->PQ switch only occurs when Q exceeds
     * limit by more than this amount.
     */
    void setQlimDeadband(double db) { p_qlim_deadband = db; }

    /**
     * Check to see if there are any Q limit violations in the network
     * @param area only check for Q limit violations in this area
     * @return true if no violations found
     */
    bool checkQlimViolations();
    bool checkQlimViolations(int area);

    /**
     * Adjust voltage setpoints for remote bus voltage regulation (IREG).
     * For each PV bus with generators that have IREG != 0 and IREG != own bus,
     * adjust the local bus voltage so that the remote bus voltage matches VS.
     * @param tol tolerance for remote voltage error (default 1e-4 pu)
     * @return true if all remote regulations are satisfied within tolerance
     */
    bool adjustRemoteRegulation(double tol = 1.0e-4);

    /**
     * Set up pointers for IREG PV buses to read remote bus voltage.
     * Must be called after setExchange() and initBusUpdate().
     */
    void setupIREGPointers();

    /**
     * Read the two-terminal dc lines from the network data and set the
     * converter injections at the current bus voltages
     */
    void loadHVDC();

    /**
     * Sequential ac/dc step: re-solve every dc line at the current ac
     * voltages of its converter buses and update the converter injections.
     * A line is blocked if it is out of service or a converter bus is
     * isolated
     * @param tol tolerance (pu) on the change in converter P and Q
     * @param max_change largest change in converter P or Q (pu)
     * @param notes messages for lines whose control mode changed
     * @return true if no converter injection changed by more than tol
     */
    bool updateHVDC(double tol, double *max_change = NULL,
        std::vector<std::string> *notes = NULL);

    /**
     * Set the converter injections at the start of a solve. Lines that are
     * out of service or have an isolated converter bus are blocked; the
     * others start from the reference operating point if one has been set,
     * otherwise they are solved at the current voltages
     */
    void startHVDC();

    /**
     * Use the current dc operating point as the starting point of later
     * solves (e.g. the base case for contingency calculations)
     */
    void setHVDCReference();

    /**
     * Number of two-terminal dc lines
     * @return number of dc lines
     */
    int numHVDCLines() const;

    /**
     * Names of the dc lines
     * @param active_only only lines scheduled to operate (MDC not 0)
     * @return dc line names
     */
    std::vector<std::string> getHVDCLineNames(bool active_only) const;

    /**
     * Set the status of a dc line; false takes it out of service
     * @param name dc line name
     * @param status new status
     * @param old previous status
     * @return false if no line has this name
     */
    bool setHVDCLineStatus(const std::string &name, bool status,
        bool *old = NULL);

    /**
     * Dc line data, status and latest operating point, indexed by line
     */
    const std::vector<HVDCLine>& getHVDCLines() const;
    const std::vector<bool>& getHVDCLineStatus() const;
    const std::vector<HVDCSolution>& getHVDCSolutions() const;

    /**
     * Index of a dc line in getHVDCLines()
     * @param name dc line name
     * @return index, or -1 if no line has this name
     */
    int getHVDCLineIndex(const std::string &name) const;

    /**
     * Operating point later solves start from: the reference if one is
     * set, otherwise the latest operating point
     */
    const std::vector<HVDCSolution>& getHVDCStartingPoint() const;

    /**
     * Set the dc line operating points, e.g. those of an external solution,
     * and the converter injections they give
     * @param solutions one operating point per dc line
     */
    void setHVDCSolutions(const std::vector<HVDCSolution> &solutions);

    /**
     * Clear changes that were made for Q limit violations and reset
     * system to its original state
     */
    void clearQlimViolations();

    /**
     * Check switched shunt violations and adjust shunt B values.
     * For buses with SWREM != 0, resolves remote bus voltage.
     * @return true if no violations found (all voltages within deadband)
     */
    bool checkSwitchedShuntViolations();

    /**
     * Clear switched shunt adjustments and reset to BINIT state
     */
    void clearSwitchedShunts();

    /**
     * Check LTC violations and adjust transformer tap ratios.
     * @return true if no violations found (all controlled voltages within deadband)
     */
    bool checkLTCViolations();

    /**
     * Clear LTC adjustments and reset taps to initial values
     */
    void clearLTCControls();

    /**
     * Compute net MW export for each area via tie-line flows.
     * @param areaExport map from area number to net MW export (positive = export)
     */
    void computeAreaExport(std::map<int,double> &areaExport);

    /**
     * Set "ignore" parameter on all buses with violations so that subsequent
     * checks are not counted as violations
     */
    void ignoreVoltageViolations();

    /**
     * Clear "ignore" parameter on all buses
     */
    void clearVoltageViolations();

    /**
     * Check to see if there are any line overload violations in
     * the network. The last call checks for overloads on specific lines.
     * @param area only check for voltage violations in this area
     * @param bus1 original index of "from" bus for branch
     * @param bus2 original index of "to" bus for branch
     * @param tags line IDs for individual lines
     * @param violations true if violation detected on branch, false otherwise
     * @return true if no violations found
     */
    bool checkLineOverloadViolations();
    bool checkLineOverloadViolations(int area);
    bool checkLineOverloadViolations(std::vector<int> &bus1, std::vector<int> &bus2,
        std::vector<std::string> &tags, std::vector<bool> &violations);

    /**
     * Set "ignore" parameter on all lines with violations so that subsequent
     * checks are not counted as violations
     */
    void ignoreLineOverloadViolations();

    /**
     * Clear "ignore" parameter on all lines
     */
    void clearLineOverloadViolations();

    /**
     * Reinitialize voltages
     */
    void resetVoltages();

    /**
     * Set the initial start mode for power flow solver
     * @param mode INIT_START_WARM (default): use voltage values from raw file
     *             INIT_START_FLAT: flat start (PV/Slack use VS, PQ use 1.0 pu, all angles 0)
     */
    void setInitStartMode(InitStartMode mode);

    /**
     * Scale generator real power. If zone less than 1 then scale all
     * generators in the area
     * @param scale factor to scale real power generation
     * @param area index of area for scaling generation
     * @param zone index of zone for scaling generation
     */
    void scaleGeneratorRealPower(double scale, int area, int zone);

    /**
     * Scale load real power. If zone less than 1 then scale all
     * loads in the area
     * @param scale factor to scale load real power
     * @param area index of area for scaling load
     * @param zone index of zone for scaling load
     * @return false if there is not enough capacity to change generation
     *         by requested amount
     */
    void scaleLoadPower(double scale, int area, int zone);

    /**
     * Return the total real power load for all loads in the zone. If zone
     * less than 1, then return the total load for the area
     * @param area index of area
     * @param zone index of zone
     * @return total load
     */
    double getTotalLoadRealPower(int area, int zone);

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
    void getGeneratorMargins(int area, int zone, double *total, double *pmin,
        double *pmax);

    /**
     * Reset power of loads and generators to original values
     */
    void resetPower();

    /**
     * Set parameters for real time path rating diagnostics
     * @param src_area generation area
     * @param src_zone generation zone
     * @param load_area load area
     * @param load_zone load zone
     * @param gen_scale scale factor for generation
     * @param load_scale scale factor for loads
     */
    void setRTPRParams(int src_area, int src_zone, int load_area,
            int load_zone, double gen_scale, double load_scale);

    /**
     * Return vector describing all violations
     * @return violation vector
     */
    std::vector<Violation> getViolations();

    /**
     * Clear violation vector
     */
    void clearViolations();

    /**
     * User rate B parameter for line overload violations
     * @param flag if true, use RATEB parameter
     */
    void useRateB(bool flag);

    /**
     * Select which rating tier drives overload checks. 'A', 'B', 'C';
     * A->B->C fallback when the picked tier is zero/missing.
     */
    void setContingencyRating(const std::string& rating);
    std::string getContingencyRating() const { return p_contingencyRating; }

    /**
     * Rating for one branch element under the current tier + fallback.
     */
    double pickBranchRating(int branchLocalIdx, int elemIdx) const;

    // ---------------------------------------------------------------
    // Extensions for the GPU batch contingency path. They are only called
    // when that path is enabled and never change solver behavior.
    // ---------------------------------------------------------------

    /**
     * Select the Jacobian layout on every bus and branch
     */
    void setJacobianFormulation(JacobianFormulation form);

    /**
     * Report what the last setContingency() did to topology and the slack
     */
    void getContingencyEffects(ContingencyEffects *effects) const;

    // ---------------------------------------------------------------
    // Shortcuts for cases whose topology is already known (GPU batch
    // path). Each leaves the network exactly as the full routine named
    // in its comment would, for the cases it is used on.

    /**
     * checkLoneBus() limited to the listed local buses. Equivalent when no
     * other bus can have lost its last in-service branch: the unmodified
     * network has no lone buses and only these buses' branches changed.
     */
    bool checkLoneBusAt(std::vector<int> buses);

    /**
     * What detectIslands() records for a network known to be one island
     */
    void setSingleIsland();

    /**
     * setYBus() on the listed buses and branches only. A bus's values
     * depend on its own data and its branches; a branch's on its own.
     */
    void setYBusAt(const std::vector<int> &buses,
        const std::vector<int> &branches);

    /**
     * The state checkLineOverloadViolations() leaves behind, without the
     * check: each flow it computes copies the exchanged voltage of the two
     * end buses into their internal state. The circuits it considers
     * (not ignored, positive rating) are found on the first call; ratings
     * and ignore flags must not change afterwards.
     */
    void touchLineCheckBuses();

    /**
     * clearQlim() on the listed local buses only. Equivalent to
     * clearQlimViolations() when every other bus is unchanged since the
     * last full clear (clearQlim() leaves a cleared bus as it is).
     */
    void clearQlimAt(const std::vector<int> &buses);

    /**
     * Export the network as it stands now. Called after the base case is
     * solved and the Q-limit changes are cleared, so that it describes the
     * state every contingency solve starts from (plus the base solution).
     * Recomputes admittances and scheduled injections from the components.
     */
    void exportSupersetModel(SupersetModel *model);

    /**
     * Read back the absolute values of selected buses and branches in the
     * current (contingency applied) state. End buses of listed branches are
     * included automatically. Admittances and injections are recomputed by
     * the components themselves (setYBus, setSBus).
     * @param model model returned by exportSupersetModel
     * @param buses local bus indices
     * @param branches local branch indices
     * @param state values of the listed buses and of the edges of the
     *        listed branches
     */
    void captureCaseState(const SupersetModel &model,
        const std::vector<int> &buses, const std::vector<int> &branches,
        SupersetCaseState *state);

  private:

    /**
     * Superset bus type of a bus in its current state
     */
    static int supersetType(PFBus *bus);

    /**
     * Reactive limits GridPACK's Q-limit check applies to a bus (MVAr):
     * the totals of its in-service generators, or no limits at all for a
     * bus without generators
     */
    static void qlimBounds(PFBus *bus, double *qmax, double *qmin);

    /**
     * Sum of the admittance of all branch objects joining local buses k and
     * m, seen from k (cached by setYBus)
     */
    void pairAdmittance(int k, int m, double *g, double *b);


    NetworkPtr p_network;
    std::vector<bool> p_saveIsolatedStatus;
    std::vector<int>  p_loneBusIndices;
    std::vector<bool> p_saveIslandIsolatedStatus;  // For island detection
    std::vector<int> p_islandIsolatedBusIndices;   // Local indices of buses isolated due to islanding
    int p_islandCount;  // Number of islands detected
    bool p_hasLoneBus;  // Whether any lone buses were found
    int p_originalSlackBusIdx;  // Local index of original slack bus
    int p_currentSlackBusIdx;   // Local index of current slack bus (may differ after transfer)
    bool p_slackTransferred;    // Whether slack was transferred during contingency

    std::vector<Violation> p_violations;

    // Circuits checkLineOverloadViolations() computes a flow for: branch
    // local index and the circuit whose status getComplexPower() checks
    struct CheckedCircuit {
      int branch;
      int status_index;
    };
    std::vector<CheckedCircuit> p_checkedCircuits;
    bool p_checkedCircuitsReady = false;

    bool p_rateB;
    std::string p_contingencyRating;  // "A" | "B" | "C" (default "A")
    double p_qlim_deadband;  // Q deadband (Mvar) for PV->PQ switch

    std::vector<HVDCLine> p_hvdc_lines;         // two-terminal dc lines
    std::vector<bool> p_hvdc_status;            // false when out of service
    std::vector<HVDCSolution> p_hvdc_solution;  // latest operating point
    std::vector<HVDCSolution> p_hvdc_reference; // starting point of solves
    bool p_hvdc_have_reference;
    std::vector<int> p_hvdc_buses;              // converter ac buses

    /**
     * Voltage magnitude and isolation flag (1 or 0) of each converter bus,
     * contributed by the process that owns the bus
     * @param state 2 entries per converter bus, in p_hvdc_buses order
     */
    void gatherHVDCBusState(std::vector<double> &state);

    /**
     * Set the injections of the converter buses from the dc line
     * operating points in p_hvdc_solution
     */
    void applyHVDCInjections();
};

} // powerflow
} // gridpack
#endif
