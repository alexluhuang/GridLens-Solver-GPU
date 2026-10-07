/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   ca_driver.cpp
 * @author Bruce Palmer
 * @date   2017-12-08 13:12:46 d3g096
 *
 * @updated Yousu Chen
 * - N-1 auto-generation for branch and generator contingencies
 * - Automatic slack bus transfer and capacity check
 * - Q-limit support integration
 * @date  2026-01-31
 *
 * @updated Yousu Chen
 * - csv_flat / csv_delta per-(contingency,branch) outputs
 * - monitorBranchesFile / monitorAreas / monitorKvMin/Max filters (all formats)
 * @date  2026-06-21
 *
 * @brief Driver for contingency analysis calculation that make use of the
 *        powerflow module to implement individual power flow simulations for
 *        each contingency. The different contingencies are distributed across
 *        separate communicators using the task manager.
 *
 *
 */
// -------------------------------------------------------------

#include "gridpack/include/gridpack.hpp"
#include "gridpack/applications/modules/powerflow/pf_app_module.hpp"
#include "gridpack/utilities/results_exporter.hpp"
#include "ca_driver.hpp"
#include "ca_rows.hpp"
#include "ca_parallel_write.hpp"
#include "ca_parquet.hpp"
#include "gridpack/applications/modules/batch_pf/host/batch_path.hpp"
#include "gridpack/applications/modules/batch_pf/host/reconcile.hpp"

#include <boost/scoped_ptr.hpp>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace {

/**
 * Label used for a contingency type in the output files
 * @param event contingency
 * @return "branch", "generator" or "hvdc"
 */
const char* contingencyTypeName(const gridpack::powerflow::Contingency &event)
{
  if (event.p_type == gridpack::powerflow::Branch) return "branch";
  if (event.p_type == gridpack::powerflow::Generator) return "generator";
  if (event.p_type == gridpack::powerflow::DCLine) return "hvdc";
  return "unknown";
}

}  // namespace

// Statistical-summary output (vmag.txt, pflow.txt, etc.) used to be controlled
// by a USE_STATBLOCK build-time macro; it is now a runtime XML option,
// `Configuration.Contingency_analysis.writeStats`, defaulting to true to
// preserve existing behavior.
// Sets up multiple communicators so that individual contingency calculations
// can be run concurrently

/**
 * Basic constructor
 */
gridpack::contingency_analysis::CADriver::CADriver(void)
{
}

/**
 * Basic destructor
 */
gridpack::contingency_analysis::CADriver::~CADriver(void)
{
}

/**
 * Get list of contingencies from external file
 * @param cursor pointer to contingencies in input deck
 * @return vector of contingencies
 */
std::vector<gridpack::powerflow::Contingency>
  gridpack::contingency_analysis::CADriver::getContingencies(
      gridpack::utility::Configuration::ChildCursors contingencies)
{
  // The contingencies ChildCursors argument is a vector of configuration
  // pointers. Each element in the vector is pointing at a seperate Contingency
  // block within the Contingencies block in the input file.
  std::vector<gridpack::powerflow::Contingency> ret;
  int size = contingencies.size();
  int i, idx;
  // Create string utilities object to help parse file
  gridpack::utility::StringUtils utils;
  // Loop over all child cursors
  for (idx = 0; idx < size; idx++) {
    std::string ca_type;
    contingencies[idx]->get("contingencyType",&ca_type);
    // Contingency name is used to direct output to different files for each
    // contingency
    std::string ca_name;
    contingencies[idx]->get("contingencyName",&ca_name);
    if (ca_type == "Line") {
      std::string buses;
      contingencies[idx]->get("contingencyLineBuses",&buses);
      std::string names;
      if (!contingencies[idx]->get("CKT",&names)) {
        contingencies[idx]->get("contingencyLineNames",&names);
      }
      // Tokenize bus string to get a list of individual buses
      std::vector<std::string> string_vec = utils.blankTokenizer(buses);
      // Convert buses from character strings to ints
      std::vector<int> bus_ids;
      for (i=0; i<string_vec.size(); i++) {
        bus_ids.push_back(atoi(string_vec[i].c_str()));
      }
      string_vec.clear();
      // Tokenize names string to get a list of individual line tags
      string_vec = utils.blankTokenizer(names);
      std::vector<std::string> line_names;
      // clean up line tags so that they are exactly two characters
      for (i=0; i<string_vec.size(); i++) {
        line_names.push_back(utils.clean2Char(string_vec[i]));
      }
      // Check to make sure we found everything
      if (bus_ids.size() == 2*line_names.size()) {
        // Add contingency parameters to contingency struct
        gridpack::powerflow::Contingency contingency;
        contingency.p_name = ca_name;
        contingency.p_type = Branch;
        int i;
        for (i = 0; i < line_names.size(); i++) {
          contingency.p_from.push_back(bus_ids[2*i]);
          contingency.p_to.push_back(bus_ids[2*i+1]);
          contingency.p_ckt.push_back(line_names[i]);
          contingency.p_saveLineStatus.push_back(true);
        }
        // Add branch contingency to contingency list
        ret.push_back(contingency);
      }
    } else if (ca_type == "Generator") {
      std::string buses;
      contingencies[idx]->get("contingencyBuses",&buses);
      std::string gens;
      if (!contingencies[idx]->get("GenID",&gens)) {
        contingencies[idx]->get("contingencyGenerators",&gens);
      }
      // Tokenize bus string to get a list of individual buses
      std::vector<std::string> string_vec = utils.blankTokenizer(buses);
      std::vector<int> bus_ids;
      // Convert buses from character strings to ints
      for (i=0; i<string_vec.size(); i++) {
        bus_ids.push_back(atoi(string_vec[i].c_str()));
      }
      string_vec.clear();
      // Tokenize gens string to get a list of individual generator tags
      string_vec = utils.blankTokenizer(gens);
      std::vector<std::string> gen_ids;
      // clean up generator tags so that they are exactly two characters
      for (i=0; i<string_vec.size(); i++) {
        gen_ids.push_back(utils.clean2Char(string_vec[i]));
      }
      // Check to make sure we found everything
      if (bus_ids.size() == gen_ids.size()) {
        gridpack::powerflow::Contingency contingency;
        contingency.p_name = ca_name;
        contingency.p_type = Generator;
        int i;
        for (i = 0; i < bus_ids.size(); i++) {
          contingency.p_busid.push_back(bus_ids[i]);
          contingency.p_genid.push_back(gen_ids[i]);
          contingency.p_saveGenStatus.push_back(true);
        }
        // Add generator contingency to contingency list
        ret.push_back(contingency);
      }
    } else if (ca_type == "DCLine" || ca_type == "HVDC") {
      // Two-terminal dc lines (poles) to block. Names may contain blanks,
      // so several lines are separated by ';' or ','
      std::string names;
      contingencies[idx]->get("contingencyDCLines",&names);
      gridpack::powerflow::Contingency contingency;
      contingency.p_name = ca_name;
      contingency.p_type = gridpack::powerflow::DCLine;
      std::string cur;
      for (size_t k = 0; k <= names.size(); k++) {
        if (k == names.size() || names[k] == ';' || names[k] == ',') {
          std::string nm = gridpack::powerflow::normalizeHVDCName(cur);
          if (!nm.empty()) {
            contingency.p_dclines.push_back(nm);
            contingency.p_saveDCLineStatus.push_back(true);
          }
          cur.clear();
        } else {
          cur += names[k];
        }
      }
      if (!contingency.p_dclines.empty()) ret.push_back(contingency);
    }
  }
  return ret;
}

/**
 * Auto-generate N-1 contingencies from the network
 * @param pf_app power flow application module with loaded network
 * @param gen_branches generate branch contingencies
 * @param gen_generators generate generator contingencies
 * @return vector of auto-generated contingencies
 */
std::vector<gridpack::powerflow::Contingency>
  gridpack::contingency_analysis::CADriver::generateN1Contingencies(
      gridpack::powerflow::PFAppModule &pf_app,
      bool gen_branches, bool gen_generators, bool gen_dclines)
{
  std::vector<gridpack::powerflow::Contingency> ret;
  gridpack::utility::StringUtils utils;

  // Generate N-1 branch contingencies
  if (gen_branches) {
    std::vector<std::string> branch_data = pf_app.writeBranchString("flow_str");
    int branch_count = 0;

    for (size_t i=0; i<branch_data.size(); i++) {
      std::vector<std::string> tokens = utils.blankTokenizer(branch_data[i]);
      if (tokens.size()%8 != 0) {
        continue;  // Skip malformed data
      }

      int nline = tokens.size()/8;
      for (int j=0; j<nline; j++) {
        int from_bus = atoi(tokens[j*8].c_str());
        int to_bus = atoi(tokens[j*8+1].c_str());
        std::string ckt_id = tokens[j*8+2];

        // Create contingency for this branch
        gridpack::powerflow::Contingency contingency;
        char name_buf[64];
        snprintf(name_buf, sizeof(name_buf), "BR_%d_%d_%s", from_bus, to_bus,
                utils.clean2Char(ckt_id).c_str());
        contingency.p_name = name_buf;
        contingency.p_type = Branch;
        contingency.p_from.push_back(from_bus);
        contingency.p_to.push_back(to_bus);
        contingency.p_ckt.push_back(utils.clean2Char(ckt_id));
        contingency.p_saveLineStatus.push_back(true);

        ret.push_back(contingency);
        branch_count++;
      }
    }

    if (gridpack::parallel::Communicator().rank() == 0) {
      printf("Auto-generated %d N-1 branch contingencies\n", branch_count);
    }
  }

  // Generate N-1 generator contingencies
  if (gen_generators) {
    std::vector<std::string> gen_data = pf_app.writeBusString("power");
    int gen_count = 0;

    for (size_t i=0; i<gen_data.size(); i++) {
      std::vector<std::string> tokens = utils.blankTokenizer(gen_data[i]);
      if (tokens.size()%4 != 0) {
        continue;  // Skip malformed data
      }

      int ngen = tokens.size()/4;
      for (int j=0; j<ngen; j++) {
        int bus_id = atoi(tokens[j*4].c_str());
        std::string gen_id = tokens[j*4+1];

        // Create contingency for this generator
        gridpack::powerflow::Contingency contingency;
        char name_buf[64];
        snprintf(name_buf, sizeof(name_buf), "GN_%d_%s", bus_id,
                utils.clean2Char(gen_id).c_str());
        contingency.p_name = name_buf;
        contingency.p_type = Generator;
        contingency.p_busid.push_back(bus_id);
        contingency.p_genid.push_back(utils.clean2Char(gen_id));
        contingency.p_saveGenStatus.push_back(true);

        ret.push_back(contingency);
        gen_count++;
      }
    }

    if (gridpack::parallel::Communicator().rank() == 0) {
      printf("Auto-generated %d N-1 generator contingencies\n", gen_count);
    }
  }

  // Generate N-1 two-terminal dc line (pole) contingencies
  if (gen_dclines) {
    std::vector<std::string> names = pf_app.getHVDCLineNames(true);
    for (size_t i=0; i<names.size(); i++) {
      gridpack::powerflow::Contingency contingency;
      std::string tag = names[i];
      std::replace(tag.begin(), tag.end(), ' ', '_');
      contingency.p_name = "DC_" + tag;
      contingency.p_type = gridpack::powerflow::DCLine;
      contingency.p_dclines.push_back(names[i]);
      contingency.p_saveDCLineStatus.push_back(true);
      ret.push_back(contingency);
    }
    if (gridpack::parallel::Communicator().rank() == 0) {
      printf("Auto-generated %d N-1 dc line contingencies\n",
          static_cast<int>(names.size()));
    }
  }

  return ret;
}

/**
 * Check if a contingency is a duplicate of any in the existing list
 * @param contingency the contingency to check
 * @param existing_list vector of existing contingencies
 * @return true if duplicate found, false otherwise
 */
bool gridpack::contingency_analysis::CADriver::isDuplicateContingency(
    const gridpack::powerflow::Contingency &contingency,
    const std::vector<gridpack::powerflow::Contingency> &existing_list)
{
  for (size_t i = 0; i < existing_list.size(); i++) {
    const gridpack::powerflow::Contingency &existing = existing_list[i];

    // Check if same type
    if (existing.p_type != contingency.p_type) {
      continue;
    }

    if (contingency.p_type == Branch) {
      // For branch contingencies, check if same branches are tripped
      // A duplicate means all branches match (order doesn't matter)
      if (existing.p_from.size() != contingency.p_from.size()) {
        continue;
      }

      bool all_match = true;
      for (size_t j = 0; j < contingency.p_from.size(); j++) {
        bool found = false;
        for (size_t k = 0; k < existing.p_from.size(); k++) {
          if (contingency.p_from[j] == existing.p_from[k] &&
              contingency.p_to[j] == existing.p_to[k] &&
              contingency.p_ckt[j] == existing.p_ckt[k]) {
            found = true;
            break;
          }
        }
        if (!found) {
          all_match = false;
          break;
        }
      }

      if (all_match) {
        return true;  // Duplicate found
      }

    } else if (contingency.p_type == Generator) {
      // For generator contingencies, check if same generators are tripped
      if (existing.p_busid.size() != contingency.p_busid.size()) {
        continue;
      }

      bool all_match = true;
      for (size_t j = 0; j < contingency.p_busid.size(); j++) {
        bool found = false;
        for (size_t k = 0; k < existing.p_busid.size(); k++) {
          if (contingency.p_busid[j] == existing.p_busid[k] &&
              contingency.p_genid[j] == existing.p_genid[k]) {
            found = true;
            break;
          }
        }
        if (!found) {
          all_match = false;
          break;
        }
      }

      if (all_match) {
        return true;  // Duplicate found
      }
    } else if (contingency.p_type == gridpack::powerflow::DCLine) {
      // Same set of dc lines (order doesn't matter)
      std::set<std::string> a(contingency.p_dclines.begin(),
          contingency.p_dclines.end());
      std::set<std::string> b(existing.p_dclines.begin(),
          existing.p_dclines.end());
      if (a == b) return true;
    }
  }

  return false;  // Not a duplicate
}

/**
 * Execute application. argc and argv are standard runtime parameters
 */
void gridpack::contingency_analysis::CADriver::execute(int argc, char** argv)
{
  // Create world communicator for entire simulation
  gridpack::parallel::Communicator world;

  // Get timer instance for timing entire calculation
  gridpack::utility::CoarseTimer *timer =
    gridpack::utility::CoarseTimer::instance();
  int t_total = timer->createCategory("Total Application");
  timer->start(t_total);

  // Read configuration file (user specified, otherwise assume that it is
  // call input.xml)
  gridpack::utility::Configuration *config
    = gridpack::utility::Configuration::configuration();
  if (argc >= 2 && argv[1] != NULL) {
    char inputfile[256];
    snprintf(inputfile, sizeof(inputfile),"%s",argv[1]);
    config->open(inputfile,world);
  } else {
    config->open("input.xml",world);
  }
  // Optional GPU batch path (Contingency_analysis/GPUBatch). Settings are
  // read now, while the main input file is the open configuration; without
  // a GPUBatch block nothing below changes.
  gridpack::batchpf::BatchPath gpuPath(config, world);

  // Get size of group (communicator) that individual contingency calculations
  // will run on and create a task communicator. Each process is part of only
  // one task communicator, even though the world communicator is broken up into
  // many task communicators
  gridpack::utility::Configuration::CursorPtr cursor;
  cursor = config->getCursor("Configuration.Contingency_analysis");
  int grp_size;
  double Vmin, Vmax;
  // Check to find out if files should be printed for individual power flow
  // calculations
  bool print_calcs;
  std::string tmp_bool;
  gridpack::utility::StringUtils util;
  if (!cursor->get("printCalcFiles",&tmp_bool)) {
    print_calcs = true;
  } else {
    util.toLower(tmp_bool);
    if (tmp_bool == "false") {
      print_calcs = false;
    } else {
      print_calcs = true;
    }
  }
  // Statistical-summary output (vmag.txt, pflow.txt, etc. via StatBlock).
  // Default true to preserve existing behavior; set false to skip the
  // per-case StatBlock work and the 13 post-loop global writes.
  bool write_stats = true;
  if (cursor->get("writeStats",&tmp_bool)) {
    util.toLower(tmp_bool);
    write_stats = (tmp_bool != "false");
  }
  // groupSize is forced to 1: an outaged branch may straddle a multi-rank
  // partition. Scale by adding ranks, not by widening a group.
  if (!cursor->get("groupSize",&grp_size)) {
    grp_size = 1;
  }
  if (grp_size != 1) {
    if (world.rank() == 0) {
      printf("WARNING: groupSize=%d is not supported (a contingency branch "
             "may span the partition boundary); using groupSize=1\n",
             grp_size);
    }
    grp_size = 1;
  }
  if (!cursor->get("minVoltage",&Vmin)) {
    Vmin = 0.9;
  }
  if (!cursor->get("maxVoltage",&Vmax)) {
    Vmax = 1.1;
  }
  // Check for Q limit violations (qlim: true=enabled, false=disabled)
  bool check_Qlim = cursor->get("qlim", true);
  double qlim_deadband = cursor->get("qlimDeadband", 0.1);
  // Output format: "text" (default), "json", "csv", "csv_flat", "csv_delta",
  // "parquet" (csv_delta's branch table as two Parquet files, ca_parquet.hpp).
  std::string outputFormat = "text";
  cursor->get("outputFormat", &outputFormat);
  if (outputFormat == "parquet" &&
      !gridpack::contingency_analysis::ParquetFlows::available()) {
    if (world.rank() == 0) {
      std::cout << "ERROR: outputFormat='parquet' needs Parquet support, which"
                   " this ca.x was built without (Apache Arrow's Parquet C++"
                   " library). Aborting.\n" << std::flush;
    }
    world.barrier();
    MPI_Abort(static_cast<MPI_Comm>(world), 1);
  }
  // parquet mode is csv_delta mode with another writer for the branch table
  const bool parquetOut = (outputFormat == "parquet");
  const bool deltaRows = (outputFormat == "csv_delta" || parquetOut);
  if (outputFormat != "text" && outputFormat != "json" &&
      outputFormat != "csv"  && outputFormat != "csv_flat" &&
      outputFormat != "csv_delta" && !parquetOut) {
    if (world.rank() == 0) {
      printf("ERROR: unrecognized outputFormat='%s'. "
             "Must be one of: text, json, csv, csv_flat, csv_delta, parquet. "
             "Aborting.\n",
             outputFormat.c_str());
    }
    world.barrier();
    MPI_Abort(static_cast<MPI_Comm>(world), 1);
  }
  std::string outputFile = "ca_results";
  cursor->get("outputFile", &outputFile);
  // Optional CSV allowlist (from_bus,to_bus,ckt). Empty -> emit all.
  std::string monitorBranchesFile;
  cursor->get("monitorBranchesFile", &monitorBranchesFile);
  // Optional area/kV gates. Empty/zero/missing -> no restriction on that
  // dimension. Filters AND together with monitorBranchesFile.
  std::string monitorAreasStr;
  cursor->get("monitorAreas", &monitorAreasStr);
  double monitorKvMin = 0.0;
  cursor->get("monitorKvMin", &monitorKvMin);
  double monitorKvMax = 0.0;
  cursor->get("monitorKvMax", &monitorKvMax);
  // Split on blanks, tabs, newlines, commas or semicolons; skip bad tokens.
  std::set<int> monitorAreas;
  {
    std::string cur;
    std::vector<std::string> tok;
    for (size_t i = 0; i <= monitorAreasStr.size(); i++) {
      char c = (i < monitorAreasStr.size()) ? monitorAreasStr[i] : ' ';
      bool sep = (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                  c == ',' || c == ';');
      if (sep) {
        if (!cur.empty()) { tok.push_back(cur); cur.clear(); }
      } else {
        cur += c;
      }
    }
    for (size_t i = 0; i < tok.size(); i++) {
      char *end = NULL;
      long v = strtol(tok[i].c_str(), &end, 10);
      if (end == tok[i].c_str() || *end != '\0' || v <= 0) {
        if (world.rank() == 0) {
          printf("WARNING: monitorAreas token '%s' is not a positive integer; "
                 "ignored\n", tok[i].c_str());
        }
        continue;
      }
      monitorAreas.insert(static_cast<int>(v));
    }
  }
  // Any monitor filter configured; gates every output format.
  bool haveMonitorFilter = !monitorAreas.empty() ||
                           monitorKvMin > 0.0 || monitorKvMax > 0.0 ||
                           !monitorBranchesFile.empty();
  // Loading% denominator for all CA outputs (.out, _violations.csv, JSON,
  // csv_flat, csv_delta). A|B|C, default A to match PW/PSSE convention.
  // A->B->C fallback if the requested tier is zero/missing.
  std::string contingencyRating = "A";
  cursor->get("contingencyRating", &contingencyRating);
  util.toUpper(contingencyRating);
  if (contingencyRating != "A" && contingencyRating != "B" &&
      contingencyRating != "C") {
    if (world.rank() == 0) {
      printf("WARNING: contingencyRating='%s' not A/B/C; defaulting to A\n",
             contingencyRating.c_str());
    }
    contingencyRating = "A";
  }
  // Severity threshold for violation reporting; loading% > threshold*100
  // is flagged. Default 1.0 (100% of rate).
  double violationSeverityThreshold = 1.0;
  cursor->get("violationSeverityThreshold", &violationSeverityThreshold);
  if (violationSeverityThreshold <= 0.0) violationSeverityThreshold = 1.0;
  // Cap on top_severe_contingencies and roster arrays in _summary.json.
  int topN = 10;
  cursor->get("topN", &topN);
  if (topN < 1) topN = 1;
  if (topN > 10000) topN = 10000;
  // Weights for composite_pi = piBranchWeight*branch_pi + piVoltageWeight*voltage_pi.
  double piBranchWeight = 1.0, piVoltageWeight = 1.0;
  cursor->get("piBranchWeight",  &piBranchWeight);
  cursor->get("piVoltageWeight", &piVoltageWeight);
  if (piBranchWeight  < 0.0) piBranchWeight  = 0.0;
  if (piVoltageWeight < 0.0) piVoltageWeight = 0.0;

  // Set static flag for PFBus class BEFORE network creation.
  // This controls how Q values are reported in output functions:
  // - When check_Qlim = false: output uses calculated Q from p_Qinj
  // - When check_Qlim = true: output uses p_qg (set by chkQlim())
  gridpack::powerflow::PFBus::setQlim(check_Qlim);
  gridpack::powerflow::PFBus::setQlimDeadband(qlim_deadband);
  gridpack::parallel::Communicator task_comm = world.divide(grp_size);

  // Create powerflow applications on each task communicator
  boost::shared_ptr<gridpack::powerflow::PFNetwork>
    pf_network(new gridpack::powerflow::PFNetwork(task_comm));
  gridpack::powerflow::PFAppModule pf_app;
  // Read in the network from an external file and partition it over the
  // processors in the task communicator. This will read in power flow
  // parameters from the Powerflow block in the input
  // Study phases, reported by GridPACK's coarse timer at the end of the run
  // alongside its power flow categories, so the CPU and GPU paths can be
  // compared phase by phase (guide 8.12)
  const int t_read = timer->createCategory("CA: Read Network");
  const int t_base = timer->createCategory("CA: Base Case");
  const int t_list = timer->createCategory("CA: Case List and Output Setup");
  const int t_cases = timer->createCategory("CA: Solve and Report Cases");
  const int t_merge = timer->createCategory("CA: Merge Output Files");
  // Inside the case loop: the work done for every case, summed per rank.
  // On the GPU path the solve is replaced by injecting the GPU result, and
  // the loop time not covered here is waiting for results and scheduling.
  const int t_case_apply = timer->createCategory("CA case: Apply Outage");
  const int t_case_solve = timer->createCategory("CA case: CPU Solve");
  const int t_case_inject = timer->createCategory("CA case: Inject GPU Result");
  const int t_case_report = timer->createCategory("CA case: Check and Report");
  const int t_case_rows = timer->createCategory("CA case: Write Table Rows");
  const int t_case_restore = timer->createCategory("CA case: Restore Network");
  timer->start(t_read);
  pf_app.readNetwork(pf_network,config);
  // Finish initializing the network
  pf_app.initialize();
  timer->stop(t_read);

  // Build (number -> name) lookup tables for area, zone, owner.
  // Keyed on the PSS/E-assigned number (not contiguous), used when
  // emitting per-(branch,contingency) CSV rows so each row carries
  // human-readable area/zone/owner names alongside the numbers.
  std::map<int, std::string> area_name_by_num;
  std::map<int, std::string> zone_name_by_num;
  std::map<int, std::string> owner_name_by_num;
  {
    boost::shared_ptr<gridpack::component::DataCollection> netdata =
      pf_network->getNetworkData();
    int aT = 0, zT = 0, oT = 0;
    netdata->getValue(AREA_TOTAL,  &aT);
    netdata->getValue(ZONE_TOTAL,  &zT);
    netdata->getValue(OWNER_TOTAL, &oT);
    for (int i = 0; i < aT; i++) {
      int n = 0; std::string s;
      netdata->getValue(AREAINTG_NUMBER, &n, i);
      netdata->getValue(AREAINTG_NAME,   &s, i);
      area_name_by_num[n] = s;
    }
    for (int i = 0; i < zT; i++) {
      int n = 0; std::string s;
      netdata->getValue(ZONE_NUMBER, &n, i);
      netdata->getValue(ZONE_NAME,   &s, i);
      zone_name_by_num[n] = s;
    }
    for (int i = 0; i < oT; i++) {
      int n = 0; std::string s;
      netdata->getValue(OWNER_NUMBER, &n, i);
      netdata->getValue(OWNER_NAME,   &s, i);
      owner_name_by_num[n] = s;
    }
  }

  // Per-rank bus metadata + wide/long branch-row outputs for csv_flat
  // (long-form one row per branch per case) and csv_delta (wide-form one
  // row per branch per case joining base and cont state). Both share the
  // bus_meta load, the buses sidecar, and the per-rank .part-file gather.
  bool wantBusSidecar = (outputFormat == "csv_flat" || deltaRows);
  struct BusMeta {
    std::string name;
    double basekv;
    int    area, zone, owner;
  };
  std::map<int, BusMeta> bus_meta;
  if (wantBusSidecar) {
    int nBus = pf_network->numBuses();
    for (int i = 0; i < nBus; i++) {
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>(pf_network->getBus(i).get());
      if (!bus) continue;
      int orig = pf_network->getOriginalBusIndex(i);
      BusMeta m;
      m.name   = bus->getBusName();
      m.basekv = bus->getBaseKV();
      m.area   = bus->getArea();
      m.zone   = bus->getZone();
      m.owner  = bus->getOwner();
      bus_meta[orig] = m;
    }
  }
  // (area, base kV) per bus, all-gathered over task_comm so filter lookups
  // work on gathered rows.
  struct BusAreaKv { int area; double basekv; };
  std::map<int, BusAreaKv> bus_ak;
  if (wantBusSidecar || haveMonitorFilter) {
    std::vector<int> lid, larea;
    std::vector<double> lkv;
    int nBus = pf_network->numBuses();
    for (int i = 0; i < nBus; i++) {
      if (!pf_network->getActiveBus(i)) continue;
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>(pf_network->getBus(i).get());
      if (!bus) continue;
      lid.push_back(pf_network->getOriginalBusIndex(i));
      larea.push_back(bus->getArea());
      lkv.push_back(bus->getBaseKV());
    }
    MPI_Comm tc = static_cast<MPI_Comm>(task_comm);
    int tsize = task_comm.size();
    int nloc = static_cast<int>(lid.size());
    std::vector<int> counts(tsize, 0), displs(tsize, 0);
    MPI_Allgather(&nloc, 1, MPI_INT, &counts[0], 1, MPI_INT, tc);
    int tot = 0;
    for (int p = 0; p < tsize; p++) { displs[p] = tot; tot += counts[p]; }
    std::vector<int> gid(tot > 0 ? tot : 1), garea(tot > 0 ? tot : 1);
    std::vector<double> gkv(tot > 0 ? tot : 1);
    int *lid_p   = nloc > 0 ? &lid[0]   : NULL;
    int *larea_p = nloc > 0 ? &larea[0] : NULL;
    double *lkv_p = nloc > 0 ? &lkv[0]  : NULL;
    MPI_Allgatherv(lid_p,   nloc, MPI_INT,    &gid[0],   &counts[0], &displs[0], MPI_INT,    tc);
    MPI_Allgatherv(larea_p, nloc, MPI_INT,    &garea[0], &counts[0], &displs[0], MPI_INT,    tc);
    MPI_Allgatherv(lkv_p,   nloc, MPI_DOUBLE, &gkv[0],   &counts[0], &displs[0], MPI_DOUBLE, tc);
    for (int i = 0; i < tot; i++) {
      BusAreaKv a;
      a.area = garea[i];
      a.basekv = gkv[i];
      bus_ak[gid[i]] = a;
    }
  }

  // Strip surrounding single quotes (PSS/E style) and outer whitespace.
  auto trim_quoted = [](const std::string &in) -> std::string {
    std::string s = in;
    size_t a = s.find_first_not_of(" \t");
    size_t b = s.find_last_not_of(" \t");
    if (a == std::string::npos) return std::string();
    s = s.substr(a, b - a + 1);
    if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'') {
      s = s.substr(1, s.size() - 2);
    }
    a = s.find_first_not_of(" \t");
    b = s.find_last_not_of(" \t");
    return (a == std::string::npos) ? std::string() : s.substr(a, b - a + 1);
  };
  auto lookup_name = [&](const std::map<int, std::string> &m, int n) -> std::string {
    std::map<int, std::string>::const_iterator it = m.find(n);
    if (it == m.end()) return std::string();
    return trim_quoted(it->second);
  };

  // Per-rank bus metadata sidecar (deduped by world rank 0 after the loop).
  if (wantBusSidecar) {
    std::ostringstream oss;
    oss << outputFile << "_buses." << world.rank() << ".part";
    std::ofstream fbus(oss.str().c_str(),
                       std::ios::out | std::ios::trunc | std::ios::binary);
    fbus << std::fixed;
    for (std::map<int, BusMeta>::const_iterator it = bus_meta.begin();
         it != bus_meta.end(); ++it) {
      const BusMeta &m = it->second;
      fbus << it->first << ","
           << trim_quoted(m.name) << ","
           << std::setprecision(2) << m.basekv << ","
           << m.area << "," << m.zone << "," << m.owner << ","
           << lookup_name(area_name_by_num,  m.area)  << ","
           << lookup_name(zone_name_by_num,  m.zone)  << ","
           << lookup_name(owner_name_by_num, m.owner)
           << "\n";
    }
    fbus.close();
  }

  // Per-rank streaming output file. Opened on first row written so non-csv_flat
  // runs and ranks that produce no rows leave nothing behind.
  std::string flatPartPath;
  if (outputFormat == "csv_flat") {
    std::ostringstream oss;
    oss << outputFile << "_flat." << world.rank() << ".part";
    flatPartPath = oss.str();
  }
  std::ofstream flatPart;
  size_t flatRowCount = 0;

  // Per-rank _violations.csv stream. Populated by every output format that
  // knows about violations (json/csv via populateViolations; csv_flat/csv_delta
  // inline where loading_pct is already computed). Emit-branch and emit-voltage
  // helpers below open the file lazily.
  std::string violPartPath;
  {
    std::ostringstream oss;
    oss << outputFile << "_violations." << world.rank() << ".part";
    violPartPath = oss.str();
  }
  std::ofstream violPart;
  size_t violRowCount = 0;
  // Running summary state; captureFlatRows / captureDeltaRows / populateViolations
  // all funnel through the emit helpers, so this stays consistent regardless of
  // outputFormat.
  struct WorstBranchState {
    double loading_pct = 0.0;
    int from = 0, to = 0;
    std::string ckt;
    std::string ct_name;
  };
  struct WorstVoltageState {
    double v_pu = 1.0;
    double dev_pu = 0.0;   // signed
    int bus_id = 0;
    std::string ct_name;
  };
  WorstBranchState worstBr;
  WorstVoltageState worstVLo;
  worstVLo.v_pu = 1e9;   // start high so any real low value beats it
  WorstVoltageState worstVHi;
  worstVHi.v_pu = -1e9;
  std::set<std::string> ctsWithBranchViol;
  std::set<std::string> ctsWithVoltageViol;
  // Rank-local per-ct composite indices; reduced to world 0 in summary.
  std::map<std::string, double> ctPi;             // sum(mva/rate)^2, all monitored branches
  std::map<std::string, double> ctVpi;            // sum((v-1)/dv)^2, all energized buses
  std::map<std::string, double> ctVdev;           // sum(v-limit)^2, violated buses only
  std::map<std::string, double> ctWorstLoading;   // max loading_pct among violated branches
  std::map<std::string, double> ctWorstVdev;      // max |v - limit| among violated buses
  std::map<std::string, double> ctWorstVpu;       // v_pu that produced ctWorstVdev
  auto accumBranchPi = [&](const std::string &ct_name,
                           double mva, double rate) {
    if (rate <= 0.0) return;
    double r = mva / rate;
    ctPi[ct_name] += r * r;
  };
  // Textbook voltage PI: normalize by the direction-aware half-band so a bus
  // at Vmin or Vmax contributes 1.0 even when limits are asymmetric.
  // Skip isolated (v<=0) and NaN/inf buses so numerical failures on a single
  // bus don't poison the accumulator.
  auto accumVoltagePi = [&](const std::string &ct_name, double v_pu) {
    if (v_pu <= 0.0 || !std::isfinite(v_pu)) return;
    double denom = (v_pu < 1.0) ? (1.0 - Vmin) : (Vmax - 1.0);
    if (denom <= 0.0) return;
    double r = (v_pu - 1.0) / denom;
    ctVpi[ct_name] += r * r;
  };
  // Legacy deviation metric: (v-limit)^2 accrued only for violated buses.
  auto accumVoltageDev = [&](const std::string &ct_name, double dev_pu) {
    ctVdev[ct_name] += dev_pu * dev_pu;
  };
  auto openViolPart = [&]() {
    if (violPart.is_open()) return;
    violPart.open(violPartPath.c_str(), std::ios::out | std::ios::trunc);
    violPart << std::fixed;
  };
  // Row schema (same 11 columns for branch and voltage rows; unused fields blank):
  //   event_idx,contingency,type,element,mva_or_vpu,rate_or_limit,loading_percent,
  //   base_mva,delta,severity
  // For branch: element = "from-to-ckt", mva_or_vpu = MVA,      rate_or_limit = rate,
  //             loading_percent = %,     base_mva/delta populated,        severity.
  // For voltage: element = "bus_id",     mva_or_vpu = v_pu,     rate_or_limit = "low/high",
  //             loading_percent = "",    base_mva = "",  delta = signed dev pu.
  auto emitBranchViolation = [&](int event_idx, const std::string &ct_name,
                                 int from, int to, const std::string &ckt,
                                 double mva, double rate,
                                 double loading_pct, double base_mva) {
    openViolPart();
    const char *sev = (loading_pct >= 105.0) ? "critical" : "warning";
    violPart << event_idx << "," << ct_name << ",branch,"
             << from << "-" << to << "-" << ckt << ","
             << std::setprecision(4) << mva << ","
             << std::setprecision(4) << rate << ","
             << std::setprecision(2) << loading_pct << ","
             << std::setprecision(4) << base_mva << ","
             << std::setprecision(4) << (mva - base_mva) << ","
             << sev << "\n";
    violRowCount++;
    if (loading_pct > worstBr.loading_pct) {
      worstBr.loading_pct = loading_pct;
      worstBr.from = from; worstBr.to = to;
      worstBr.ckt = ckt;
      worstBr.ct_name = ct_name;
    }
    ctsWithBranchViol.insert(ct_name);
    double &pc = ctWorstLoading[ct_name];
    if (loading_pct > pc) pc = loading_pct;
  };
  auto emitVoltageViolation = [&](int event_idx, const std::string &ct_name,
                                  int bus_id, double v_pu,
                                  double lo, double hi) {
    openViolPart();
    double dev = (v_pu < lo) ? (v_pu - lo)
               : (v_pu > hi) ? (v_pu - hi)
               : 0.0;
    if (dev == 0.0) return;
    const char *sev = (std::abs(dev) >= 0.05) ? "critical" : "warning";
    // rate_or_limit column holds "low_pu:high_pu" for voltage rows.
    std::ostringstream limits;
    limits << std::setprecision(4) << std::fixed << lo << ":" << hi;
    violPart << event_idx << "," << ct_name << ",voltage,"
             << bus_id << ","
             << std::setprecision(6) << v_pu << ","
             << limits.str() << ","
             << ","                                 // loading_percent blank
             << ","                                 // base_mva blank
             << std::setprecision(6) << dev << ","
             << sev << "\n";
    violRowCount++;
    if (dev < 0.0 && v_pu < worstVLo.v_pu) {
      worstVLo.v_pu = v_pu;
      worstVLo.dev_pu = dev;
      worstVLo.bus_id = bus_id;
      worstVLo.ct_name = ct_name;
    }
    if (dev > 0.0 && v_pu > worstVHi.v_pu) {
      worstVHi.v_pu = v_pu;
      worstVHi.dev_pu = dev;
      worstVHi.bus_id = bus_id;
      worstVHi.ct_name = ct_name;
    }
    ctsWithVoltageViol.insert(ct_name);
    accumVoltageDev(ct_name, dev);
    double absDev = std::abs(dev);
    double &d = ctWorstVdev[ct_name];
    if (absDev > d) { d = absDev; ctWorstVpu[ct_name] = v_pu; }
  };

  // (from, to, ckt) key shared by the monitor allowlist and base_cache.
  struct BranchKey {
    int from, to;
    std::string ckt;
    bool operator<(const BranchKey &o) const {
      if (from != o.from) return from < o.from;
      if (to   != o.to  ) return to   < o.to;
      return ckt < o.ckt;
    }
  };

  // Per-branch rate-A/B/C from the parsed network data. Keyed by (from,to,ckt)
  // so the csv_flat / csv_delta emit paths can pick the configured rating.
  // Built once before the contingency loop. Each rank only sees its own
  // active+ghost branches; that's fine -- the emit path is also rank-local.
  struct BranchRates {
    double rate_a, rate_b, rate_c;
  };
  std::map<BranchKey, BranchRates> branch_rates;
  if (outputFormat == "csv_flat" || deltaRows) {
    int nBranch = pf_network->numBranches();
    for (int i = 0; i < nBranch; i++) {
      boost::shared_ptr<gridpack::component::DataCollection> bd =
        pf_network->getBranchData(i);
      if (!bd) continue;
      int from = 0, to = 0, nelems = 0;
      bd->getValue(BRANCH_FROMBUS, &from);
      bd->getValue(BRANCH_TOBUS,   &to);
      if (!bd->getValue(BRANCH_NUM_ELEMENTS, &nelems)) continue;
      for (int k = 0; k < nelems; k++) {
        std::string ckt;
        if (!bd->getValue(BRANCH_CKT, &ckt, k)) continue;
        // Trim leading/trailing whitespace and PSS/E surrounding quotes.
        size_t a = ckt.find_first_not_of(" \t");
        size_t b = ckt.find_last_not_of(" \t");
        ckt = (a == std::string::npos) ? std::string()
                                       : ckt.substr(a, b - a + 1);
        if (ckt.size() >= 2 && ckt.front() == '\'' && ckt.back() == '\'') {
          ckt = ckt.substr(1, ckt.size() - 2);
          a = ckt.find_first_not_of(" \t");
          b = ckt.find_last_not_of(" \t");
          ckt = (a == std::string::npos) ? std::string()
                                         : ckt.substr(a, b - a + 1);
        }
        BranchRates r;
        r.rate_a = 0.0; r.rate_b = 0.0; r.rate_c = 0.0;
        bd->getValue(BRANCH_RATING_A, &r.rate_a, k);
        bd->getValue(BRANCH_RATING_B, &r.rate_b, k);
        bd->getValue(BRANCH_RATING_C, &r.rate_c, k);
        BranchKey key;
        key.from = from; key.to = to; key.ckt = ckt;
        branch_rates[key] = r;
      }
    }
  }
  // Base case always uses rate-A (PSS/E "normal" rating). Contingency rows
  // use whichever the user picked, with A->B->C fallback if zero/missing.
  auto pickContRate = [&](const BranchRates &r) -> double {
    if (contingencyRating == "A") {
      return r.rate_a;
    }
    if (contingencyRating == "B") {
      return (r.rate_b > 0.0) ? r.rate_b : r.rate_a;
    }
    if (r.rate_c > 0.0) return r.rate_c;
    if (r.rate_b > 0.0) return r.rate_b;
    return r.rate_a;
  };

  // Monitor allowlist parsed from monitorBranchesFile. Empty -> emit all.
  std::set<BranchKey> monitorSet;
  if (!monitorBranchesFile.empty()) {
    std::ifstream fin(monitorBranchesFile.c_str());
    if (!fin.is_open()) {
      if (world.rank() == 0) {
        printf("WARNING: monitorBranchesFile '%s' not found; emitting all branches\n",
               monitorBranchesFile.c_str());
      }
    } else {
      std::string line;
      size_t lineNo = 0;
      while (std::getline(fin, line)) {
        lineNo++;
        // Strip trailing CR (Windows line endings).
        while (!line.empty() && (line[line.size()-1] == '\r' ||
                                 line[line.size()-1] == '\n')) {
          line.resize(line.size()-1);
        }
        // Skip blank lines and comments.
        size_t firstNon = line.find_first_not_of(" \t");
        if (firstNon == std::string::npos) continue;
        if (line[firstNon] == '#') continue;
        // Tokenize on commas.
        std::vector<std::string> tok;
        size_t pos = 0;
        while (pos <= line.size()) {
          size_t comma = line.find(',', pos);
          std::string t = (comma == std::string::npos)
                          ? line.substr(pos)
                          : line.substr(pos, comma - pos);
          size_t a = t.find_first_not_of(" \t");
          size_t b = t.find_last_not_of(" \t");
          tok.push_back((a == std::string::npos) ? std::string()
                                                 : t.substr(a, b - a + 1));
          if (comma == std::string::npos) break;
          pos = comma + 1;
        }
        if (tok.size() < 3) continue;
        // Skip header row: any non-numeric first field.
        if (tok[0].empty()) continue;
        bool numeric = true;
        for (size_t ci = 0; ci < tok[0].size(); ci++) {
          char c = tok[0][ci];
          if (!(c >= '0' && c <= '9') && c != '-' && c != '+') {
            numeric = false; break;
          }
        }
        if (!numeric) continue;
        BranchKey k;
        k.from = atoi(tok[0].c_str());
        k.to   = atoi(tok[1].c_str());
        k.ckt  = tok[2];
        monitorSet.insert(k);
      }
      if (world.rank() == 0) {
        printf("Monitor allowlist: %zu branches loaded from %s\n",
               monitorSet.size(), monitorBranchesFile.c_str());
      }
    }
  }
  // Endpoints of allowlisted branches; gates voltage rows under an allowlist.
  std::set<int> monitorBusSet;
  for (std::set<BranchKey>::const_iterator it = monitorSet.begin();
       it != monitorSet.end(); ++it) {
    monitorBusSet.insert(it->from);
    monitorBusSet.insert(it->to);
  }
  // Area/kV gate. Either-endpoint match for areas (catches tie-lines).
  // kV is gated on max(kv_from, kv_to) so a 138/13.8 stepdown counts as 138.
  // Empty area set / zero kV bound = unrestricted on that dimension.
  auto passesAreaKv = [&](int area_from, int area_to,
                          double kv_from, double kv_to) {
    if (!monitorAreas.empty()) {
      if (monitorAreas.find(area_from) == monitorAreas.end() &&
          monitorAreas.find(area_to)   == monitorAreas.end()) {
        return false;
      }
    }
    double kv_max = (kv_from > kv_to) ? kv_from : kv_to;
    if (monitorKvMin > 0.0 && kv_max < monitorKvMin) return false;
    if (monitorKvMax > 0.0 && kv_max > monitorKvMax) return false;
    return true;
  };
  // When monitorBranchesFile presents, it overrides area/kV criteria.
  bool haveAreaKvFilter = !monitorAreas.empty() ||
                          monitorKvMin > 0.0 ||
                          monitorKvMax > 0.0;
  if (!monitorSet.empty() && haveAreaKvFilter) {
    if (world.rank() == 0) {
      printf("WARNING: monitorBranchesFile is set; ignoring "
             "monitorAreas/monitorKvMin/monitorKvMax\n");
    }
    monitorAreas.clear();
    monitorKvMin = 0.0;
    monitorKvMax = 0.0;
    haveAreaKvFilter = false;
  }
  if (world.rank() == 0) {
    if (!monitorAreas.empty()) {
      printf("Monitor areas filter: %zu areas:", monitorAreas.size());
      for (std::set<int>::const_iterator it = monitorAreas.begin();
           it != monitorAreas.end(); ++it) printf(" %d", *it);
      printf("\n");
    }
    if (monitorKvMin > 0.0 || monitorKvMax > 0.0) {
      printf("Monitor kV filter: min=%.2f max=%.2f (0 means unbounded)\n",
             monitorKvMin, monitorKvMax);
    }
    // List the areas and base kV levels present in the case.
    if (haveAreaKvFilter) {
      std::map<int, int> area_cnt;
      std::map<double, int> kv_cnt;
      for (std::map<int, BusAreaKv>::const_iterator it = bus_ak.begin();
           it != bus_ak.end(); ++it) {
        area_cnt[it->second.area]++;
        kv_cnt[it->second.basekv]++;
      }
      const size_t maxShow = 25;
      size_t n = 0;
      printf("Case areas (buses):");
      for (std::map<int, int>::const_iterator it = area_cnt.begin();
           it != area_cnt.end() && n < maxShow; ++it, ++n)
        printf(" %d(%d)", it->first, it->second);
      if (area_cnt.size() > maxShow)
        printf(" ... %zu areas total", area_cnt.size());
      printf("\n");
      n = 0;
      printf("Case base kV levels (buses):");
      for (std::map<double, int>::const_iterator it = kv_cnt.begin();
           it != kv_cnt.end() && n < maxShow; ++it, ++n)
        printf(" %g(%d)", it->first, it->second);
      if (kv_cnt.size() > maxShow)
        printf(" ... %zu levels total", kv_cnt.size());
      printf("\n");
    }
  }
  auto busAreaKv = [&](int bus, int &area, double &kv) {
    std::map<int, BusAreaKv>::const_iterator it = bus_ak.find(bus);
    area = (it != bus_ak.end()) ? it->second.area   : 0;
    kv   = (it != bus_ak.end()) ? it->second.basekv : 0.0;
  };
  // Branch monitor predicate shared by every output path (ckt padding trimmed).
  auto branchMonitored = [&](int from, int to, const std::string &ckt) -> bool {
    if (!monitorSet.empty()) {
      BranchKey k;
      k.from = from; k.to = to; k.ckt = ckt;
      while (!k.ckt.empty() && (k.ckt[k.ckt.size()-1] == ' ' ||
                                k.ckt[k.ckt.size()-1] == '\t'))
        k.ckt.resize(k.ckt.size()-1);
      return monitorSet.find(k) != monitorSet.end();
    }
    if (!haveAreaKvFilter) return true;
    int af = 0, at = 0;
    double kf = 0.0, kt = 0.0;
    busAreaKv(from, af, kf);
    busAreaKv(to,   at, kt);
    return passesAreaKv(af, at, kf, kt);
  };
  // Bus counterpart: allowlist endpoint, else own area / base kV.
  auto busMonitored = [&](int bus) -> bool {
    if (!monitorSet.empty()) {
      return monitorBusSet.find(bus) != monitorBusSet.end();
    }
    if (!haveAreaKvFilter) return true;
    int area = 0;
    double kv = 0.0;
    busAreaKv(bus, area, kv);
    if (!monitorAreas.empty() && monitorAreas.find(area) == monitorAreas.end())
      return false;
    if (monitorKvMin > 0.0 && kv < monitorKvMin) return false;
    if (monitorKvMax > 0.0 && kv > monitorKvMax) return false;
    return true;
  };

  struct BaseFlow {
    double p_mw, q_mvar, mva, loading_pct;
    double base_rate, cont_rate;
    double v_from_pu, v_to_pu, ang_from_deg, ang_to_deg;
    double base_kv_from, base_kv_to;
    int    area_from, area_to;
  };
  std::map<BranchKey, BaseFlow> base_cache;
  std::string deltaPartPath;
  if (outputFormat == "csv_delta") {
    std::ostringstream oss;
    oss << outputFile << "_delta." << world.rank() << ".part";
    deltaPartPath = oss.str();
  }
  std::ofstream deltaPart;
  // parquet: this rank's part of <outputFile>_flows.parquet
  std::string flowsPartPath;
  std::unique_ptr<gridpack::contingency_analysis::ParquetFlows> flowsPart;
  if (parquetOut) {
    std::ostringstream oss;
    oss << outputFile << "_flows." << world.rank() << ".parquet.part";
    flowsPartPath = oss.str();
    flowsPart = std::make_unique<gridpack::contingency_analysis::ParquetFlows>(
        flowsPartPath, world.rank());
  }
  size_t deltaRowCount = 0;
  size_t deltaSkipCount = 0;

  // Convergence sidecar rows.
  struct ConvRow {
    int    event_idx;
    std::string name;
    std::string type;
    gridpack::utility::ConvergenceSummary cs;
    std::string status;
  };
  std::vector<ConvRow> localConvRows;
  // _convergence.csv: written for every outputFormat.
  bool emitConv = true;

  // ---- Flat and delta table rows, read straight from the solved state ----
  // These rows used to be built from text: writeBusString("vr_str") and
  // writeBranchString("flow_str") formatted every bus and branch, gathered
  // the strings through a Global Array, and the driver parsed each one back
  // with sscanf. Reading the components directly gives the same values
  // (round13 reproduces the print-and-parse rounding) in the same order
  // (buses by number, branches in global order), so the files stay
  // byte-identical, several times faster. As before, a branch object
  // contributes only its first circuit: its string held one line per circuit
  // and only the first line was parsed. RowBranch::regular is false in the
  // rare case where a tag would not read back as one token; such branches
  // still go through the exact text route.
  using gridpack::contingency_analysis::round13;
  using gridpack::contingency_analysis::scanToken;
  using gridpack::contingency_analysis::RowText;
  using CircuitFlow = gridpack::powerflow::PFBranch::CircuitFlow;
  struct RowBus {
    int local;
    int id;
    bool monitored;
  };
  struct RowBranch {
    int local = 0;
    int from = 0, to = 0;
    bool regular = true;
    std::string ckt;    // the first tag as "%15s" read it (csv_delta key)
    std::string ckt3;   // its first three characters (csv_flat)
    int from_slot = -1, to_slot = -1;           // positions in rowBuses
    bool mon_flat = false, mon_delta = false;
    const BranchRates *rates_flat = nullptr;    // branch_rates entry, if any
    const BranchRates *rates_delta = nullptr;
    int base_slot = -1;                         // rowBase entry, csv_delta
    // this case's values, as the text route read them
    double p = 0.0, q = 0.0, rate_a = 0.0;
    int viol = 0;
  };
  std::vector<RowBus> rowBuses;
  std::map<int, int> rowSlotOfId;
  std::vector<double> rowBusV, rowBusA;
  std::vector<RowBranch> rowBranches;
  std::vector<BaseFlow> rowBase;
  std::map<BranchKey, int> rowBaseIndex;
  std::vector<BranchKey> rowBaseKey;   // key of each rowBase entry
  std::vector<CircuitFlow> rowCircuits;
  RowText rowText;
  bool rowsPrepared = false;
  auto slotOf = [&](int id) -> int {
    std::map<int, int>::const_iterator it = rowSlotOfId.find(id);
    return (it == rowSlotOfId.end()) ? -1 : it->second;
  };
  auto ratesOf = [&](int from, int to, const std::string &ckt) -> const BranchRates * {
    BranchKey key;
    key.from = from; key.to = to; key.ckt = ckt;
    std::map<BranchKey, BranchRates>::const_iterator it = branch_rates.find(key);
    return (it == branch_rates.end()) ? nullptr : &it->second;
  };
  // Keys, monitor decisions and rate lookups of one branch's current token
  auto resolveBranch = [&](RowBranch &rb) {
    rb.ckt3 = rb.ckt.substr(0, 3);
    rb.from_slot = slotOf(rb.from);
    rb.to_slot = slotOf(rb.to);
    rb.mon_flat = branchMonitored(rb.from, rb.to, rb.ckt3);
    rb.mon_delta = branchMonitored(rb.from, rb.to, rb.ckt);
    rb.rates_flat = ratesOf(rb.from, rb.to, rb.ckt3);
    rb.rates_delta = ratesOf(rb.from, rb.to, rb.ckt);
  };
  auto prepareRows = [&]() {
    if (rowsPrepared) return;
    rowsPrepared = true;
    // Buses as writeStrings listed them (global order), keyed by number
    // with the last one winning, then visited by number like the std::map
    std::vector<std::pair<int, int> > order;
    for (int i = 0; i < pf_network->numBuses(); i++) {
      if (pf_network->getActiveBus(i)) {
        order.push_back(std::make_pair(pf_network->getGlobalBusIndex(i), i));
      }
    }
    std::sort(order.begin(), order.end());
    std::map<int, int> localOfId;
    for (size_t k = 0; k < order.size(); k++) {
      localOfId[pf_network->getBus(order[k].second)->getOriginalIndex()] = order[k].second;
    }
    for (std::map<int, int>::const_iterator it = localOfId.begin(); it != localOfId.end(); ++it) {
      rowSlotOfId[it->first] = static_cast<int>(rowBuses.size());
      RowBus rb{};
      rb.local = it->second;
      rb.id = it->first;
      rb.monitored = busMonitored(it->first);
      rowBuses.push_back(rb);
    }
    rowBusV.assign(rowBuses.size(), 0.0);
    rowBusA.assign(rowBuses.size(), 0.0);
    // Branches serialWrite writes (active at load, with circuits), global order
    order.clear();
    for (int i = 0; i < pf_network->numBranches(); i++) {
      if (pf_network->getActiveBranch(i)) {
        order.push_back(std::make_pair(pf_network->getGlobalBranchIndex(i), i));
      }
    }
    std::sort(order.begin(), order.end());
    for (size_t k = 0; k < order.size(); k++) {
      gridpack::powerflow::PFBranch *br = pf_network->getBranch(order[k].second).get();
      if (!br->isActiveAtLoad()) continue;
      std::vector<std::string> tags = br->getLineTags();
      if (tags.empty()) continue;
      RowBranch rb;
      rb.local = order[k].second;
      rb.from = br->getBus1OriginalIndex();
      rb.to = br->getBus2OriginalIndex();
      rb.regular = scanToken(tags[0], &rb.ckt);
      if (rb.regular) resolveBranch(rb);
      rowBranches.push_back(rb);
    }
  };
  // This case's bus voltages and angles
  auto readBuses = [&]() {
    for (size_t k = 0; k < rowBuses.size(); k++) {
      double angle = 0.0, vmag = 0.0;
      pf_network->getBus(rowBuses[k].local)->stateValues(&angle, &vmag);
      rowBusA[k] = round13(angle);
      rowBusV[k] = round13(vmag);
    }
  };
  // This case's values of one branch's first circuit; false if the text
  // route would have skipped the branch
  auto readBranch = [&](RowBranch &rb) -> bool {
    gridpack::powerflow::PFBranch *br = pf_network->getBranch(rb.local).get();
    if (!br->flowValues(&rowCircuits) || rowCircuits.empty()) return false;
    if (rb.regular) {
      const CircuitFlow &c = rowCircuits[0];
      rb.p = round13(c.p);
      rb.q = round13(c.q);
      rb.rate_a = round13(c.rate_a);
      rb.viol = c.viol;
      return true;
    }
    // The exact text route: the string serialWrite("flow_str") builds
    // (2048 bytes, as the branch IO allocates) and the same sscanf
    std::string text;
    char line[256];
    for (size_t i = 0; i < rowCircuits.size(); i++) {
      const CircuitFlow &c = rowCircuits[i];
      std::snprintf(line, sizeof(line), "%6d %6d %s %20.12e %20.12e %20.12e %20.12e %1d\n",
                    rb.from, rb.to, c.tag.c_str(), c.p, c.q, c.perf, c.rate_a, c.viol);
      if (text.size() + std::strlen(line) < 2048) text += line;
    }
    char ckt_buf[16] = {0};
    int from = 0, to = 0;
    double perf = 0.0;
    if (sscanf(text.c_str(), "%d %d %15s %lf %lf %lf %lf %d",
               &from, &to, ckt_buf, &rb.p, &rb.q, &perf, &rb.rate_a, &rb.viol) != 8) {
      return false;
    }
    rb.from = from;
    rb.to = to;
    rb.ckt = ckt_buf;
    while (!rb.ckt.empty() && rb.ckt[rb.ckt.size()-1] == ' ') rb.ckt.resize(rb.ckt.size()-1);
    resolveBranch(rb);
    return true;
  };
  // Voltage PI and violations on monitored buses, in bus-number order. The
  // sum keeps the old expressions and order, so the summary is unchanged.
  auto voltageChecks = [&](int event_idx, const std::string &ct_name) {
    double *vpi = nullptr;
    for (size_t k = 0; k < rowBuses.size(); k++) {
      if (!rowBuses[k].monitored) continue;
      double v_pu = rowBusV[k];
      if (v_pu > 0.0 && std::isfinite(v_pu)) {
        double denom = (v_pu < 1.0) ? (1.0 - Vmin) : (Vmax - 1.0);
        if (denom > 0.0) {
          double r = (v_pu - 1.0) / denom;
          if (!vpi) vpi = &ctVpi[ct_name];
          *vpi += r * r;
        }
      }
      if (v_pu <= 0.0 || !std::isfinite(v_pu)) continue;
      if (v_pu < Vmin || v_pu > Vmax) {
        emitVoltageViolation(event_idx, ct_name, rowBuses[k].id, v_pu, Vmin, Vmax);
      }
    }
  };
  auto addBranchPi = [&](double **pi, const std::string &ct_name, double mva, double rate) {
    if (rate <= 0.0) return;
    double r = mva / rate;
    if (!*pi) *pi = &ctPi[ct_name];
    **pi += r * r;
  };

  // Stream one CSV row per branch into the rank's .part file. Called once per
  // converged case (base + each contingency).
  auto captureFlatRows = [&](int event_idx, const std::string &name,
                             bool emit, bool is_base) {
    if (!emit || task_comm.rank() != 0) return;
    if (!flatPart.is_open()) {
      flatPart.open(flatPartPath.c_str(), std::ios::out | std::ios::trunc);
      flatPart << std::fixed;
    }
    prepareRows();
    readBuses();
    // Names are cut to 23 characters, as the text route's 24-byte buffer did
    constexpr size_t ct_chars = 23;
    const std::string ct_name = name.substr(0, std::min(name.find('\0'), ct_chars));
    // Voltage PI and violations on monitored buses, contingency rows only.
    if (!is_base) voltageChecks(event_idx, ct_name);
    double *pi = nullptr;
    rowText.clear();
    for (size_t bi = 0; bi < rowBranches.size(); bi++) {
      RowBranch &rb = rowBranches[bi];
      if (!readBranch(rb)) continue;
      if (!rb.mon_flat) continue;
      double p = rb.p, q = rb.q;
      double rate_sel = rb.rate_a;
      if (rb.rates_flat) {
        rate_sel = is_base ? rb.rates_flat->rate_a : pickContRate(*rb.rates_flat);
      }
      double flow_mva    = std::sqrt(p*p + q*q);
      double loading_pct = (rate_sel > 0.0) ? (flow_mva / rate_sel) * 100.0 : 0.0;
      if (!is_base) addBranchPi(&pi, ct_name, flow_mva, rate_sel);
      // Stream to _violations.csv for csv_flat runs (contingency rows only).
      // csv_flat keeps no base-case table, so base_mva stays 0.
      if (!is_base && loading_pct > violationSeverityThreshold * 100.0) {
        emitBranchViolation(event_idx, ct_name, rb.from, rb.to, rb.ckt3,
                            flow_mva, rate_sel, loading_pct, 0.0);
      }
      double v_from       = (rb.from_slot >= 0) ? rowBusV[rb.from_slot] : 0.0;
      double ang_from_deg = (rb.from_slot >= 0) ? rowBusA[rb.from_slot] : 0.0;
      double v_to         = (rb.to_slot >= 0) ? rowBusV[rb.to_slot] : 0.0;
      double ang_to_deg   = (rb.to_slot >= 0) ? rowBusA[rb.to_slot] : 0.0;
      rowText.add(event_idx).add(',').add(ct_name).add(',')
             .add(rb.from).add(',').add(rb.to).add(',').add(rb.ckt3).add(',')
             .fixed(p, 4).add(',')
             .fixed(q, 4).add(',')
             .fixed(flow_mva, 4).add(',')
             .fixed(rate_sel, 4).add(',')
             .fixed(loading_pct, 2).add(',')
             .add(rb.viol).add(',')
             .fixed(v_from, 6).add(',')
             .fixed(v_to, 6).add(',')
             .fixed(ang_from_deg, 4).add(',')
             .fixed(ang_to_deg, 4)
             .add('\n');
      flatRowCount++;
    }
    flatPart.write(rowText.str().data(), static_cast<std::streamsize>(rowText.str().size()));
  };

  // Base-case branch state for the csv_delta join, read once after the base
  // solve. As with the std::map it replaces, a later branch with the same
  // (from, to, circuit) key overwrites an earlier one.
  auto populateBaseCache = [&]() {
    if (task_comm.rank() != 0) return;
    prepareRows();
    readBuses();
    for (size_t bi = 0; bi < rowBranches.size(); bi++) {
      RowBranch &rb = rowBranches[bi];
      if (!readBranch(rb)) continue;
      if (!rb.mon_delta) continue;
      double p = rb.p, q = rb.q;
      double base_rate = rb.rate_a, cont_rate = rb.rate_a;
      if (rb.rates_delta) {
        base_rate = rb.rates_delta->rate_a;
        cont_rate = pickContRate(*rb.rates_delta);
      }
      BaseFlow bf;
      bf.p_mw        = p;
      bf.q_mvar      = q;
      bf.mva         = std::sqrt(p*p + q*q);
      bf.base_rate   = base_rate;
      bf.cont_rate   = cont_rate;
      bf.loading_pct = (base_rate > 0.0) ? (bf.mva / base_rate) * 100.0 : 0.0;
      bf.v_from_pu     = (rb.from_slot >= 0) ? rowBusV[rb.from_slot] : 0.0;
      bf.ang_from_deg  = (rb.from_slot >= 0) ? rowBusA[rb.from_slot] : 0.0;
      bf.v_to_pu       = (rb.to_slot >= 0) ? rowBusV[rb.to_slot] : 0.0;
      bf.ang_to_deg    = (rb.to_slot >= 0) ? rowBusA[rb.to_slot] : 0.0;
      busAreaKv(rb.from, bf.area_from, bf.base_kv_from);
      busAreaKv(rb.to,   bf.area_to,   bf.base_kv_to);
      BranchKey k;
      k.from = rb.from; k.to = rb.to; k.ckt = rb.ckt;
      std::map<BranchKey, int>::const_iterator it = rowBaseIndex.find(k);
      if (it == rowBaseIndex.end()) {
        rowBaseIndex[k] = static_cast<int>(rowBase.size());
        rowBase.push_back(bf);
        rowBaseKey.push_back(k);
      } else {
        rowBase[it->second] = bf;
      }
    }
    for (size_t bi = 0; bi < rowBranches.size(); bi++) {
      RowBranch &rb = rowBranches[bi];
      if (!rb.regular) continue;
      BranchKey k;
      k.from = rb.from; k.to = rb.to; k.ckt = rb.ckt;
      std::map<BranchKey, int>::const_iterator it = rowBaseIndex.find(k);
      rb.base_slot = (it == rowBaseIndex.end()) ? -1 : it->second;
    }
  };

  // Wide-form (base+cont on same row) capture for csv_delta. Mirrors
  // captureFlatRows but joins each branch with its base-case entry. Branches
  // without one are counted in deltaSkipCount and skipped silently. (The
  // cont_event_facility column stays disabled: join event_idx against
  // <outputFile>_contingencies.csv, which names every outaged element.)
  auto captureDeltaRows = [&](int event_idx,
                              const gridpack::powerflow::Contingency &evt,
                              bool emit) {
    if (!emit || task_comm.rank() != 0) return;
    if (!parquetOut && !deltaPart.is_open()) {
      deltaPart.open(deltaPartPath.c_str(), std::ios::out | std::ios::trunc);
      deltaPart << std::fixed;
    }
    prepareRows();
    std::string ct_name = evt.p_name;
    while (!ct_name.empty() && ct_name[ct_name.size()-1] == ' ')
      ct_name.resize(ct_name.size()-1);
    const char *type_str = contingencyTypeName(evt);
    readBuses();
    // Voltage PI and violations on monitored buses.
    voltageChecks(event_idx, ct_name);
    double *pi = nullptr;
    rowText.clear();
    for (size_t bi = 0; bi < rowBranches.size(); bi++) {
      RowBranch &rb = rowBranches[bi];
      if (!readBranch(rb)) continue;
      if (!rb.mon_delta) continue;
      int slot = rb.base_slot;
      if (!rb.regular) {
        BranchKey k;
        k.from = rb.from; k.to = rb.to; k.ckt = rb.ckt;
        std::map<BranchKey, int>::const_iterator it = rowBaseIndex.find(k);
        slot = (it == rowBaseIndex.end()) ? -1 : it->second;
      }
      if (slot < 0) { deltaSkipCount++; continue; }
      const BaseFlow &bf = rowBase[slot];
      double p = rb.p, q = rb.q;
      double cont_mva     = std::sqrt(p*p + q*q);
      double cont_loading = (bf.cont_rate > 0.0) ? (cont_mva / bf.cont_rate) * 100.0 : 0.0;
      addBranchPi(&pi, ct_name, cont_mva, bf.cont_rate);
      // Stream to _violations.csv (delta path knows base_mva already).
      if (cont_loading > violationSeverityThreshold * 100.0) {
        emitBranchViolation(event_idx, ct_name, rb.from, rb.to, rb.ckt,
                            cont_mva, bf.cont_rate, cont_loading, bf.mva);
      }
      double v_from_c = (rb.from_slot >= 0) ? rowBusV[rb.from_slot] : 0.0;
      double a_from_c = (rb.from_slot >= 0) ? rowBusA[rb.from_slot] : 0.0;
      double v_to_c   = (rb.to_slot >= 0) ? rowBusV[rb.to_slot] : 0.0;
      double a_to_c   = (rb.to_slot >= 0) ? rowBusA[rb.to_slot] : 0.0;
      double d_ang_b  = bf.ang_from_deg - bf.ang_to_deg;
      double d_ang_c  = a_from_c - a_to_c;
      // Across-branch drop, same convention as the angle deltas above.
      double d_v_b    = bf.v_from_pu - bf.v_to_pu;
      double d_v_c    = v_from_c - v_to_c;
      if (parquetOut) {
        // The same values; the base-case columns are in _branches.parquet
        // and the differences are differences of stored columns
        flowsPart->add(event_idx, slot, p, q, cont_mva, cont_loading,
                       v_from_c, v_to_c, a_from_c, a_to_c);
        deltaRowCount++;
        continue;
      }
      rowText.add(event_idx).add(',').add(ct_name).add(',').add(type_str).add(',')
             .add(rb.from).add(',').add(rb.to).add(',').add(rb.ckt).add(',')
             .fixed(bf.base_kv_from, 2).add(',')
             .fixed(bf.base_kv_to, 2).add(',')
             .add(bf.area_from).add(',').add(bf.area_to).add(',')
             .fixed(bf.base_rate, 4).add(',')
             .fixed(bf.cont_rate, 4).add(',')
             .fixed(bf.p_mw, 4).add(',')
             .fixed(p, 4).add(',')
             .fixed(bf.q_mvar, 4).add(',')
             .fixed(q, 4).add(',')
             .fixed(bf.mva, 4).add(',')
             .fixed(cont_mva, 4).add(',')
             .fixed(bf.loading_pct, 2).add(',')
             .fixed(cont_loading, 2).add(',')
             .fixed(bf.v_from_pu, 6).add(',')
             .fixed(v_from_c, 6).add(',')
             .fixed(bf.v_to_pu, 6).add(',')
             .fixed(v_to_c, 6).add(',')
             .fixed(bf.ang_from_deg, 4).add(',')
             .fixed(a_from_c, 4).add(',')
             .fixed(bf.ang_to_deg, 4).add(',')
             .fixed(a_to_c, 4).add(',')
             .fixed(d_v_b, 6).add(',')
             .fixed(d_v_c, 6).add(',')
             .fixed(d_ang_b, 4).add(',')
             .fixed(d_ang_c, 4)
             .add('\n');
      deltaRowCount++;
    }
    if (parquetOut) {
      flowsPart->endCase();
    } else {
      deltaPart.write(rowText.str().data(), static_cast<std::streamsize>(rowText.str().size()));
    }
  };

  timer->start(t_base);
  //  Set minimum and maximum voltage limits on all buses
  pf_app.setVoltageLimits(Vmin, Vmax);
  // Route CA violation checks and loadingPercent through the same rating tier.
  pf_app.setContingencyRating(contingencyRating);
  // Solve the base power flow on every task communicator. Abort if it fails.
  bool baseSolveOk = false;
  try {
    baseSolveOk = pf_app.solve();
    if (baseSolveOk && check_Qlim && !pf_app.checkQlimViolations()) {
      baseSolveOk = pf_app.solve();
    }
  } catch (const std::exception &e) {
    if (world.rank() == 0) {
      printf("ERROR: base-case solve threw exception: %s\n", e.what());
    }
    baseSolveOk = false;
  } catch (...) {
    if (world.rank() == 0) {
      printf("ERROR: base-case solve threw unknown exception\n");
    }
    baseSolveOk = false;
  }
  if (!baseSolveOk) {
    if (world.rank() == 0) {
      gridpack::utility::ConvergenceSummary cs = pf_app.getConvergence();
      printf("ERROR: base case did not converge "
             "(iterations=%d, final_tol=%.6e, "
             "max_p_bus=%d max_p_mismatch=%.4f, "
             "max_q_bus=%d max_q_mismatch=%.4f). "
             "Aborting contingency analysis.\n",
             cs.iterations, cs.finalTolerance,
             cs.finalMismatch.maxPBus, cs.finalMismatch.maxPMismatch,
             cs.finalMismatch.maxQBus, cs.finalMismatch.maxQMismatch);
    }
    world.barrier();
    MPI_Abort(static_cast<MPI_Comm>(world), 1);
  }
  // Operating point of the two-terminal dc lines in the base case
  if (world.rank() == 0) pf_app.writeHVDCSummary();
  // Contingency solves start their dc lines from the base-case operating
  // point, as the ac solution does from the base voltages
  pf_app.setHVDCReference();
  // Suppress voltage violations already present at base.
  pf_app.ignoreVoltageViolations();

  // Flag non-monitored elements "ignore" so the violation checks and .out
  // listings follow the filter.
  if (haveMonitorFilter) {
    long cnt[4] = { 0, 0, 0, 0 };   // mon buses, buses, mon elems, elems
    int nBus = pf_network->numBuses();
    for (int i = 0; i < nBus; i++) {
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>(pf_network->getBus(i).get());
      if (!bus) continue;
      bool mon = busMonitored(pf_network->getOriginalBusIndex(i));
      if (!mon) bus->setIgnore(true);
      if (pf_network->getActiveBus(i)) { cnt[1]++; if (mon) cnt[0]++; }
    }
    int nBranch = pf_network->numBranches();
    for (int i = 0; i < nBranch; i++) {
      gridpack::powerflow::PFBranch *br =
        dynamic_cast<gridpack::powerflow::PFBranch*>(pf_network->getBranch(i).get());
      if (!br) continue;
      int from = br->getBus1OriginalIndex();
      int to   = br->getBus2OriginalIndex();
      std::vector<std::string> tags = br->getLineTags();
      bool active = pf_network->getActiveBranch(i);
      for (size_t t = 0; t < tags.size(); t++) {
        bool mon = branchMonitored(from, to, tags[t]);
        if (!mon) br->setIgnore(tags[t], true);
        if (active) { cnt[3]++; if (mon) cnt[2]++; }
      }
    }
    long tot[4] = { 0, 0, 0, 0 };
    MPI_Allreduce(cnt, tot, 4, MPI_LONG, MPI_SUM,
                  static_cast<MPI_Comm>(task_comm));
    if (world.rank() == 0) {
      printf("Monitor filter: %ld of %ld branch elements and %ld of %ld buses "
             "monitored\n", tot[2], tot[3], tot[0], tot[1]);
      if (tot[2] == 0 && tot[0] == 0) {
        printf("WARNING: monitor filter matches nothing; all outputs will be "
               "empty. Check monitorAreas/monitorKvMin/monitorKvMax/"
               "monitorBranchesFile against the case.\n");
      }
    }
  }
  // Drop non-monitored elements from a collected result set.
  auto filterResults = [&](gridpack::utility::PowerFlowResults &r) {
    if (!haveMonitorFilter) return;
    std::vector<gridpack::utility::BusResult> buses;
    for (size_t i = 0; i < r.buses.size(); i++) {
      if (busMonitored(r.buses[i].busId)) buses.push_back(r.buses[i]);
    }
    r.buses.swap(buses);
    std::vector<gridpack::utility::BranchResult> branches;
    for (size_t i = 0; i < r.branches.size(); i++) {
      const gridpack::utility::BranchResult &b = r.branches[i];
      if (branchMonitored(b.fromBus, b.toBus, b.circuitId)) branches.push_back(b);
    }
    r.branches.swap(branches);
    std::vector<gridpack::utility::GeneratorResult> gens;
    for (size_t i = 0; i < r.generators.size(); i++) {
      if (busMonitored(r.generators[i].busId)) gens.push_back(r.generators[i]);
    }
    r.generators.swap(gens);
  };

  // Collect base case results for export. csv_flat captures rows directly
  // in the hot loop and skips the heavyweight collectResults() path.
  gridpack::utility::PowerFlowResults baseCaseResults;
  if (outputFormat == "json" || outputFormat == "csv" ||
      outputFormat == "text") {
    baseCaseResults = pf_app.collectResults();
    filterResults(baseCaseResults);
  }
  if (outputFormat == "csv_flat") {
    // The base case is replicated on every task communicator. captureFlatRows
    // calls writeBusString/writeBranchString which are task_comm collectives,
    // so every task_comm participates -- but only world rank 0 emits rows so
    // the base case isn't duplicated in the final file.
    captureFlatRows(0, std::string("base_case"), world.rank() == 0, true);
  }
  if (deltaRows) {
    // Cache base-case branch state on every rank for the contingency join.
    populateBaseCache();
  }
  if (parquetOut && world.rank() == 0) {
    // The base-case columns of csv_delta, once per branch; flows rows refer
    // to them by branch_id (the position in this table)
    std::vector<gridpack::contingency_analysis::ParquetBranchRow> table;
    for (size_t s = 0; s < rowBase.size(); s++) {
      const BaseFlow &bf = rowBase[s];
      gridpack::contingency_analysis::ParquetBranchRow r;
      r.branch_id = static_cast<int32_t>(s);
      r.from_bus = rowBaseKey[s].from;
      r.to_bus = rowBaseKey[s].to;
      r.ckt = rowBaseKey[s].ckt;
      r.base_kv_from = bf.base_kv_from;
      r.base_kv_to = bf.base_kv_to;
      r.area_from = bf.area_from;
      r.area_to = bf.area_to;
      r.base_rate_mva = bf.base_rate;
      r.cont_rate_mva = bf.cont_rate;
      r.base_p_mw = bf.p_mw;
      r.base_q_mvar = bf.q_mvar;
      r.base_mva = bf.mva;
      r.base_loading_pct = bf.loading_pct;
      r.v_from_base = bf.v_from_pu;
      r.v_to_base = bf.v_to_pu;
      r.ang_from_base = bf.ang_from_deg;
      r.ang_to_base = bf.ang_to_deg;
      table.push_back(r);
    }
    const std::string path = outputFile + "_branches.parquet";
    try {
      gridpack::contingency_analysis::ParquetFlows::writeBranches(path, table);
    } catch (const std::exception &e) {
      std::cout << "ERROR: " << e.what() << '\n' << std::flush;
      MPI_Abort(static_cast<MPI_Comm>(world), 1);
    }
    std::cout << "[parquet] wrote " << table.size() << " rows to " << path << '\n';
  }

  timer->stop(t_base);
  timer->start(t_list);
  // Check if auto-generation of N-1 contingencies is enabled
  // FullBranchN1: generate N-1 contingencies for all branches
  // FullGeneratorN1: generate N-1 contingencies for all generators
  // FullHVDCN1: generate N-1 contingencies for all two-terminal dc lines
  // (each record is one pole)
  cursor = config->getCursor("Configuration.Contingency_analysis");
  bool full_branch_n1 = false;
  bool full_generator_n1 = false;
  bool full_hvdc_n1 = false;

  if (!cursor->get("FullBranchN1",&tmp_bool)) {
    full_branch_n1 = false;
  } else {
    util.toLower(tmp_bool);
    full_branch_n1 = (tmp_bool == "true");
  }

  if (!cursor->get("FullGeneratorN1",&tmp_bool)) {
    full_generator_n1 = false;
  } else {
    util.toLower(tmp_bool);
    full_generator_n1 = (tmp_bool == "true");
  }

  if (cursor->get("FullHVDCN1",&tmp_bool)) {
    util.toLower(tmp_bool);
    full_hvdc_n1 = (tmp_bool == "true");
  }

  bool auto_generate_n1 = full_branch_n1 || full_generator_n1 ||
    full_hvdc_n1;

  std::vector<gridpack::powerflow::Contingency> events;
  int auto_generated_count = 0;
  int file_loaded_count = 0;
  int duplicates_skipped = 0;

  // Step 1: Auto-generate N-1 contingencies if requested
  if (auto_generate_n1) {
    if (world.rank() == 0) {
      printf("\n==================================================================\n");
      printf("Auto-generating N-1 contingencies from network\n");
      printf("  FullBranchN1: %s\n", full_branch_n1 ? "YES" : "NO");
      printf("  FullGeneratorN1: %s\n", full_generator_n1 ? "YES" : "NO");
      printf("  FullHVDCN1: %s\n", full_hvdc_n1 ? "YES" : "NO");
      printf("==================================================================\n\n");
    }
    events = generateN1Contingencies(pf_app, full_branch_n1, full_generator_n1,
        full_hvdc_n1);
    auto_generated_count = events.size();
  }

  // Step 2: Load contingencies from file if specified
  // This allows combining auto-generated N-1 with custom N-2+ contingencies
  std::string contingencyfile;
  bool has_contingency_file = cursor->get("contingencyList",&contingencyfile);

  if (has_contingency_file || !auto_generate_n1) {
    // Set default filename if not specified
    if (!has_contingency_file) {
      contingencyfile = "contingencies.xml";
    }

    if (world.rank() == 0) {
      if (auto_generate_n1) {
        printf("Loading additional contingencies from file: %s\n", contingencyfile.c_str());
        printf("(Duplicates of auto-generated contingencies will be skipped)\n\n");
      } else {
        printf("Contingency List: %s\n", contingencyfile.c_str());
      }
    }

    // Open contingency file
    bool ok = config->open(contingencyfile,world);

    if (ok) {
      // Get a list of contingencies from file
      cursor = config->getCursor(
          "ContingencyList.Contingency_analysis.Contingencies");
      gridpack::utility::Configuration::ChildCursors contingencies;
      if (cursor) cursor->children(contingencies);
      std::vector<gridpack::powerflow::Contingency> file_contingencies =
          getContingencies(contingencies);

      // If auto-generation was used, check for duplicates before adding
      if (auto_generate_n1) {
        for (size_t i = 0; i < file_contingencies.size(); i++) {
          if (isDuplicateContingency(file_contingencies[i], events)) {
            duplicates_skipped++;
            if (world.rank() == 0) {
              printf("  Skipping duplicate: %s\n", file_contingencies[i].p_name.c_str());
            }
          } else {
            events.push_back(file_contingencies[i]);
            file_loaded_count++;
          }
        }
        if (world.rank() == 0 && file_loaded_count > 0) {
          printf("\nAdded %d unique contingencies from file\n", file_loaded_count);
          if (duplicates_skipped > 0) {
            printf("Skipped %d duplicates\n", duplicates_skipped);
          }
        }
      } else {
        // No auto-generation, just use file contingencies
        events = file_contingencies;
        file_loaded_count = events.size();
      }
    }
  }

  // Print summary
  if (world.rank() == 0) {
    printf("\n==================================================================\n");
    printf("Total contingencies to analyze: %d\n", (int)events.size());
    if (auto_generate_n1) {
      printf("  Auto-generated: %d\n", auto_generated_count);
      if (file_loaded_count > 0) {
        printf("  From file: %d\n", file_loaded_count);
      }
      if (duplicates_skipped > 0) {
        printf("  Duplicates skipped: %d\n", duplicates_skipped);
      }
    }
    printf("==================================================================\n\n");
  }

  // event_idx lookup table: one row per contingency (0 = base case); N-k
  // element ids are ';'-separated within the columns.
  if (world.rank() == 0) {
    std::string ctgFile = outputFile + "_contingencies.csv";
    std::ofstream cout_ctg(ctgFile.c_str(), std::ios::out | std::ios::trunc);
    cout_ctg << "event_idx,contingency,type,n_elements,"
                "from_bus,to_bus,circuit_id,gen_bus,gen_id,dc_line\n";
    size_t ctgRows = 0;
    cout_ctg << "0,base_case,base,0,,,,,,\n";
    ctgRows++;
    // Trim clean2Char padding so ids join against the other CSVs.
    auto rtrim = [](const std::string &in) -> std::string {
      std::string t = in;
      while (!t.empty() && (t[t.size()-1] == ' ' || t[t.size()-1] == '\t'))
        t.resize(t.size()-1);
      return t;
    };
    for (size_t ei = 0; ei < events.size(); ei++) {
      const gridpack::powerflow::Contingency &e = events[ei];
      int event_idx = static_cast<int>(ei) + 1;
      std::string nm = rtrim(e.p_name);
      size_t n = 0;
      const char *ty = contingencyTypeName(e);
      if (e.p_type == Branch) n = e.p_from.size();
      else if (e.p_type == Generator) n = e.p_busid.size();
      else if (e.p_type == gridpack::powerflow::DCLine) n = e.p_dclines.size();
      std::ostringstream c_from, c_to, c_ckt, c_gbus, c_gid, c_dc;
      for (size_t j = 0; j < n; j++) {
        const char *sep = (j > 0) ? ";" : "";
        if (e.p_type == Branch) {
          c_from << sep << e.p_from[j];
          c_to   << sep << e.p_to[j];
          c_ckt  << sep << rtrim(e.p_ckt[j]);
        } else if (e.p_type == gridpack::powerflow::DCLine) {
          std::string dc = e.p_dclines[j];
          std::replace(dc.begin(), dc.end(), ',', ' ');
          c_dc << sep << dc;
        } else {
          c_gbus << sep << e.p_busid[j];
          c_gid  << sep << rtrim(e.p_genid[j]);
        }
      }
      // Empty events still get a row so every event_idx decodes.
      cout_ctg << event_idx << "," << nm << "," << ty << "," << n << ","
               << c_from.str() << "," << c_to.str() << "," << c_ckt.str() << ","
               << c_gbus.str() << "," << c_gid.str() << "," << c_dc.str()
               << "\n";
      ctgRows++;
    }
    cout_ctg.close();
    printf("[contingencies] wrote %zu rows to %s\n", ctgRows, ctgFile.c_str());
  }

  // Print contingency details (gated on printCalcFiles; noisy for large lists)
  if (print_calcs && world.rank() == 0) {
    int idx;
    for (idx = 0; idx < events.size(); idx++) {
      printf("Name: %s\n",events[idx].p_name.c_str());
      if (events[idx].p_type == Branch) {
        int nlines = events[idx].p_from.size();
        int j;
        for (j=0; j<nlines; j++) {
          printf(" Line: (from) %d (to) %d (line) \'%s\'\n",
              events[idx].p_from[j],events[idx].p_to[j],
              events[idx].p_ckt[j].c_str());
        }
      } else if (events[idx].p_type == Generator) {
        int nbus = events[idx].p_busid.size();
        int j;
        for (j=0; j<nbus; j++) {
          printf(" Generator: (bus) %d (generator ID) \'%s\'\n",
              events[idx].p_busid[j],events[idx].p_genid[j].c_str());
        }
      } else if (events[idx].p_type == gridpack::powerflow::DCLine) {
        for (size_t j=0; j<events[idx].p_dclines.size(); j++) {
          printf(" DC line: \'%s\'\n", events[idx].p_dclines[j].c_str());
        }
      }
    }
  }


  // Set up task manager on the world communicator. The number of tasks is
  // equal to the number of contingencies
  gridpack::parallel::TaskManager taskmgr(world);
  int ntasks = events.size();
  taskmgr.set(ntasks);

  // Get bus voltage information for base case
  int i, j;
  // StatBlock objects and the per-case scratch vectors live across the
  // contingency loop so they are declared up here, regardless of whether
  // statistics output is enabled.
  boost::scoped_ptr<gridpack::analysis::StatBlock> vmag_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> vang_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> pgen_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> qgen_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> pflow_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> qflow_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> perf_stats;
  std::vector<std::string> v_vals;
  int nsize = 0;
  std::vector<double> vmag, vang, pgen, qgen, pflow, qflow, perf;
  std::vector<int> mask, mag_mask;
  int t_store = timer->createCategory("Store Statistics");
  if (write_stats) {
    timer->start(t_store);
    v_vals = pf_app.writeBusString("vr_str");
    nsize = v_vals.size();
    std::vector<int> mag_ids;
    std::vector<int> ids;
    std::vector<std::string> mag_tags;
    std::vector<std::string> tags;
    // Find bus IDs and create a dummy tag label and get voltage magnitude
    // and angle for base case
    for (i=0; i<nsize; i++) {
      std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
      if (!busMonitored(atoi(tokens[0].c_str()))) continue;
      int not_isolated = atoi(tokens[3].c_str());
      if (not_isolated == 1) {
        mag_ids.push_back(atoi(tokens[0].c_str()));
        mag_tags.push_back("1 ");
        vmag.push_back(atof(tokens[2].c_str()));
        if (atoi(tokens[4].c_str()) != 0) {
          mag_mask.push_back(2);
        } else {
          mag_mask.push_back(1);
        }
      }
      ids.push_back(atoi(tokens[0].c_str()));
      tags.push_back("1 ");
      vang.push_back(atof(tokens[1].c_str()));
      mask.push_back(1);
    }
    int nmags = vmag.size();
    int nangs = vang.size();
    world.max(&nmags,1);
    world.max(&nangs,1);
    // Create StatBlock objects for voltage magnitude and angles and add
    // bus IDs to it as well as base case values. Row counts follow the
    // monitor filter; a zero-row block would fail inside GA, so skip it.
    if (nmags > 0) {
      vmag_stats.reset(new gridpack::analysis::StatBlock(world,nmags,ntasks+1));
    }
    if (nangs > 0) {
      vang_stats.reset(new gridpack::analysis::StatBlock(world,nangs,ntasks+1));
    }
    if (world.rank() == 0) {
      if (vmag_stats) {
        vmag_stats->addRowLabels(mag_ids, mag_tags);
        vmag_stats->addColumnValues(0,vmag,mag_mask);
      }
      if (vang_stats) {
        vang_stats->addRowLabels(ids, tags);
        vang_stats->addColumnValues(0,vang,mask);
      }
    }
    // Get generator power information
    v_vals.clear();
    ids.clear();
    tags.clear();
    mask.clear();
    v_vals = pf_app.writeBusString("power");
    nsize = v_vals.size();
    // Find bus IDs and tags for generators and evaluate Pg and Qg for base case
    for (i=0; i<nsize; i++) {
      std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
      if (tokens.size()%4 != 0) {
        printf("Incorrect generator listing\n");
        continue;
      }
      int ngen = tokens.size()/4;
      for (j=0; j<ngen; j++) {
        if (!busMonitored(atoi(tokens[j*4].c_str()))) continue;
        ids.push_back(atoi(tokens[j*4].c_str()));
        tags.push_back(tokens[j*4+1]);
        pgen.push_back(atof(tokens[j*4+2].c_str()));
        qgen.push_back(atof(tokens[j*4+3].c_str()));
        mask.push_back(1);
      }
    }
    nsize = pgen.size();
    world.max(&nsize,1);
    // Create StatBlock objects for Pg and Qg and add labels and base case values
    if (nsize > 0) {
      pgen_stats.reset(new gridpack::analysis::StatBlock(world,nsize,ntasks+1));
      qgen_stats.reset(new gridpack::analysis::StatBlock(world,nsize,ntasks+1));
    }
    if (world.rank() == 0 && pgen_stats) {
      pgen_stats->addRowLabels(ids, tags);
      qgen_stats->addRowLabels(ids, tags);
      pgen_stats->addColumnValues(0,pgen,mask);
      qgen_stats->addColumnValues(0,qgen,mask);
    }

    // Find flow parameters for all branch lines
    v_vals.clear();
    ids.clear();
    tags.clear();
    mask.clear();
    std::vector<int> id1;
    std::vector<int> id2;
    std::vector<double> pmin, pmax;
    v_vals = pf_app.writeBranchString("flow_str");
    nsize = v_vals.size();
    // Parse branch line endpoints as well as line IDs and values of P and Q for
    // base case
    for (i=0; i<nsize; i++) {
      std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
      if (tokens.size()%8 != 0) {
        printf("Incorrect branch power flow listing\n");
        continue;
      }
      int nline = tokens.size()/8;
      for (j=0; j<nline; j++) {
        if (!branchMonitored(atoi(tokens[j*8].c_str()),
                             atoi(tokens[j*8+1].c_str()),
                             tokens[j*8+2])) continue;
        id1.push_back(atoi(tokens[j*8].c_str()));
        id2.push_back(atoi(tokens[j*8+1].c_str()));
        tags.push_back(tokens[j*8+2]);
        pflow.push_back(atof(tokens[j*8+3].c_str()));
        qflow.push_back(atof(tokens[j*8+4].c_str()));
        perf.push_back(atof(tokens[j*8+5].c_str()));
        pmin.push_back(-atof(tokens[j*8+6].c_str()));
        pmax.push_back(atof(tokens[j*8+6].c_str()));
        if (atoi(tokens[j*8+7].c_str()) == 0) {
          mask.push_back(1);
        } else {
          mask.push_back(2);
        }
      }
    }
    nsize = pflow.size();
    world.max(&nsize,1);
    // Create StatBlock objects for flow parameters and add labels and base case
    // values
    if (nsize > 0) {
      pflow_stats.reset(new gridpack::analysis::StatBlock(world,nsize,ntasks+1));
      qflow_stats.reset(new gridpack::analysis::StatBlock(world,nsize,ntasks+1));
      perf_stats.reset(new gridpack::analysis::StatBlock(world,nsize,ntasks+1));
    }
    if (world.rank() == 0 && pflow_stats) {
      pflow_stats->addRowLabels(id1, id2, tags);
      qflow_stats->addRowLabels(id1, id2, tags);
      perf_stats->addRowLabels(id1, id2, tags);
      pflow_stats->addColumnValues(0,pflow,mask);
      qflow_stats->addColumnValues(0,qflow,mask);
      perf_stats->addColumnValues(0,perf,mask);
      pflow_stats->addRowMinValue(pmin);
      qflow_stats->addRowMinValue(pmin);
      pflow_stats->addRowMaxValue(pmax);
      qflow_stats->addRowMaxValue(pmax);
    }
    timer->stop(t_store);
  }
  // Write a mask-0 column for a contingency that produced no usable solution
  // (islanded, no slack, diverged, slack overload) so every task fills its
  // StatBlock column and the statistics only see valid values.
  auto addFailedStatColumns = [&](int task_id) {
    if (!write_stats) return;
    timer->start(t_store);
    vmag.clear();
    vang.clear();
    mask.clear();
    mag_mask.clear();
    v_vals.clear();
    v_vals = pf_app.writeBusString("vfail_str");
    nsize = v_vals.size();
    for (i=0; i<nsize; i++) {
      std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
      if (!busMonitored(atoi(tokens[0].c_str()))) continue;
      int not_isolated = atoi(tokens[3].c_str());
      if (not_isolated == 1) {
        vmag.push_back(0.0);
        mag_mask.push_back(0);
      }
      vang.push_back(0.0);
      mask.push_back(0);
    }
    if (task_comm.rank() == 0) {
      if (vmag_stats) vmag_stats->addColumnValues(task_id+1,vmag,mag_mask);
      if (vang_stats) vang_stats->addColumnValues(task_id+1,vang,mask);
    }
    pgen.clear();
    qgen.clear();
    mask.clear();
    v_vals.clear();
    v_vals = pf_app.writeBusString("pfail_str");
    nsize = v_vals.size();
    for (i=0; i<nsize; i++) {
      std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
      if (tokens.size()%4 != 0) {
        printf("Incorrect generator listing\n");
        continue;
      }
      int ngen = tokens.size()/4;
      for (j=0; j<ngen; j++) {
        if (!busMonitored(atoi(tokens[j*4].c_str()))) continue;
        pgen.push_back(0.0);
        qgen.push_back(0.0);
        mask.push_back(0);
      }
    }
    if (task_comm.rank() == 0 && pgen_stats) {
      pgen_stats->addColumnValues(task_id+1,pgen,mask);
      qgen_stats->addColumnValues(task_id+1,qgen,mask);
    }
    pflow.clear();
    qflow.clear();
    perf.clear();
    mask.clear();
    v_vals.clear();
    v_vals = pf_app.writeBranchString("fail_str");
    nsize = v_vals.size();
    for (i=0; i<nsize; i++) {
      std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
      if (tokens.size()%8 != 0) {
        printf("Incorrect branch power flow listing\n");
        continue;
      }
      int nline = tokens.size()/8;
      for (j=0; j<nline; j++) {
        if (!branchMonitored(atoi(tokens[j*8].c_str()),
                             atoi(tokens[j*8+1].c_str()),
                             tokens[j*8+2])) continue;
        pflow.push_back(0.0);
        qflow.push_back(0.0);
        perf.push_back(0.0);
        mask.push_back(0);
      }
    }
    if (task_comm.rank() == 0 && pflow_stats) {
      pflow_stats->addColumnValues(task_id+1,pflow,mask);
      qflow_stats->addColumnValues(task_id+1,qflow,mask);
      perf_stats->addColumnValues(task_id+1,perf,mask);
    }
    timer->stop(t_store);
  };

  if (check_Qlim) pf_app.clearQlimViolations();
  // Clear any Q limit warnings from base case before starting contingencies
  gridpack::powerflow::PFBus::clearQlimWarnings();

  // Local contingency results storage for JSON/CSV export
  std::vector<gridpack::utility::ContingencyResult> localContingencies;

  // Base-case per-element MVA lookup, keyed on (from,to,ckt).
  // Populated on rank 0 of each task_comm (only place collectResults ran).
  // Used to fill BranchViolation.baseMva/deltaMva during contingency reporting.
  std::map<BranchKey, double> baseMvaByKey;
  if (outputFormat == "json" || outputFormat == "csv" ||
      outputFormat == "text") {
    for (size_t bi = 0; bi < baseCaseResults.branches.size(); bi++) {
      const gridpack::utility::BranchResult &br = baseCaseResults.branches[bi];
      BranchKey k; k.from = br.fromBus; k.to = br.toBus; k.ckt = br.circuitId;
      while (!k.ckt.empty() && k.ckt[k.ckt.size()-1] == ' ') k.ckt.resize(k.ckt.size()-1);
      double mva = (br.mvaFrom > br.mvaTo) ? br.mvaFrom : br.mvaTo;
      baseMvaByKey[k] = mva;
    }
  }

  // Fill BranchViolation/VoltageViolation arrays from a solved ct result,
  // and stream the same rows to <outputFile>_violations.<rank>.part.
  // event_idx of 0 is reserved for base case; contingencies get task_id+1.
  auto populateViolations = [&](gridpack::utility::ContingencyResult &ct,
                                int event_idx) {
    const double threshPct = violationSeverityThreshold * 100.0;
    // Gate PI on task_comm rank 0 so groupSize>1 doesn't double-count.
    bool accumPi = (task_comm.rank() == 0);
    for (size_t bi = 0; bi < ct.solution.branches.size(); bi++) {
      const gridpack::utility::BranchResult &br = ct.solution.branches[bi];
      if (accumPi) {
        double mva_br = (br.mvaFrom > br.mvaTo) ? br.mvaFrom : br.mvaTo;
        accumBranchPi(ct.name, mva_br, br.rateSelected);
      }
      if (br.loadingPercent <= threshPct) continue;
      gridpack::utility::BranchViolation v;
      v.fromBus = br.fromBus;
      v.toBus = br.toBus;
      v.circuitId = br.circuitId;
      v.mva = (br.mvaFrom > br.mvaTo) ? br.mvaFrom : br.mvaTo;
      v.rate = br.rateSelected;
      v.loadingPercent = br.loadingPercent;
      BranchKey k; k.from = br.fromBus; k.to = br.toBus; k.ckt = br.circuitId;
      while (!k.ckt.empty() && k.ckt[k.ckt.size()-1] == ' ') k.ckt.resize(k.ckt.size()-1);
      std::map<BranchKey, double>::const_iterator it = baseMvaByKey.find(k);
      v.baseMva = (it != baseMvaByKey.end()) ? it->second : 0.0;
      v.deltaMva = v.mva - v.baseMva;
      v.severity = (br.loadingPercent >= 105.0) ? "critical" : "warning";
      ct.branchViolations.push_back(v);
      emitBranchViolation(event_idx, ct.name, v.fromBus, v.toBus, v.circuitId,
                          v.mva, v.rate, v.loadingPercent, v.baseMva);
    }
    for (size_t bi = 0; bi < ct.solution.buses.size(); bi++) {
      const gridpack::utility::BusResult &b = ct.solution.buses[bi];
      double v_pu = b.voltage;
      if (v_pu <= 0.0) continue;   // Skip isolated / not-solved buses.
      // Voltage PI accrues on every energized bus (textbook form),
      // gated on task_comm rank 0 to avoid double-count under groupSize>1.
      if (accumPi) accumVoltagePi(ct.name, v_pu);
      bool lo = v_pu < Vmin, hi = v_pu > Vmax;
      if (!lo && !hi) continue;
      gridpack::utility::VoltageViolation vv;
      vv.busId = b.busId;
      vv.vPu = v_pu;
      vv.limitLow = Vmin;
      vv.limitHigh = Vmax;
      vv.deviationPu = lo ? (v_pu - Vmin) : (v_pu - Vmax);
      double dev = std::abs(vv.deviationPu);
      vv.severity = (dev >= 0.05) ? "critical" : "warning";
      ct.voltageViolations.push_back(vv);
      emitVoltageViolation(event_idx, ct.name, vv.busId, vv.vPu,
                           vv.limitLow, vv.limitHigh);
    }
  };

  // Convergence row recorder; indexes events[task_id].
  auto recordConv = [&](int task_id, const char *status,
                        const std::string &) {
    if (!emitConv) return;
    if (task_comm.rank() != 0) return;
    ConvRow r;
    r.event_idx = task_id + 1;
    r.name      = events[task_id].p_name;
    r.type      = contingencyTypeName(events[task_id]);
    r.cs        = pf_app.getConvergence();
    r.status    = status;
    localConvRows.push_back(r);
  };

  // Evaluate contingencies using the task manager
  int task_id;
  char sbuf[512];
  std::vector<double> voltageResetReference;
  voltageResetReference.reserve(pf_network->numBuses());
  for (int b = 0; b < pf_network->numBuses(); ++b) {
    const auto *bus = dynamic_cast<gridpack::powerflow::PFBus *>(pf_network->getBus(b).get());
    voltageResetReference.push_back(bus->getInitialVoltage());
  }
  // One contingency, from setting it to restoring the network. gpu is null
  // for GridPACK's own solve; otherwise it carries a solution computed by
  // the GPU batch path, which is loaded instead of solving so that the same
  // checks and writers report it.
  // GPU cases whose topology the batch classifier already knows skip the
  // GridPACK work that only the solver and the text reports read: the
  // voltage reset, the full lone-bus and island searches, recomputing
  // every admittance and injection, and the violation checks (whose
  // results only the json, csv and text formats and printCalcFiles use).
  // Every output stays byte-identical (P2).
  const bool knownCaseReports = !print_calcs && !write_stats &&
      task_comm.size() == 1 && (outputFormat == "csv_flat" || deltaRows);
  auto processCase = [&](int task_id,
                         const gridpack::batchpf::GpuCaseResult *gpu) {
    gridpack::batchpf::CaseReport report;
    const bool known = gpu && gpu->known_topology && knownCaseReports;
    if (print_calcs) printf("Executing task %d on process %d\n",task_id,world.rank());
    // Trim trailing spaces from contingency name for filename
    std::string fname = events[task_id].p_name;
    size_t end = fname.find_last_not_of(' ');
    if (end != std::string::npos) fname = fname.substr(0, end + 1);
    snprintf(sbuf, sizeof(sbuf),"%s.out",fname.c_str());
    // Open a new file, based on the contingency name, to store results from
    // this particular contingency calculation
    if (print_calcs) pf_app.open(sbuf);
    // Write out information to the top of the output file providing some
    // information on the contingency
    snprintf(sbuf, sizeof(sbuf),"\nRunning task on %d processes\n",task_comm.size());
    if (print_calcs) pf_app.writeHeader(sbuf);
    if (events[task_id].p_type == Branch) {
      int nlines = events[task_id].p_from.size();
      int j;
      for (j=0; j<nlines; j++) {
        snprintf(sbuf, sizeof(sbuf)," Line: (from) %d (to) %d (line) \'%s\'\n",
            events[task_id].p_from[j],events[task_id].p_to[j],
            events[task_id].p_ckt[j].c_str());
        if (print_calcs) printf("p[%d] Line: (from) %d (to) %d (line) \'%s\'\n",
            pf_network->communicator().rank(),
            events[task_id].p_from[j],events[task_id].p_to[j],
            events[task_id].p_ckt[j].c_str());
      }
    } else if (events[task_id].p_type == Generator) {
      int nbus = events[task_id].p_busid.size();
      int j;
      for (j=0; j<nbus; j++) {
        snprintf(sbuf, sizeof(sbuf)," Generator: (bus) %d (generator ID) \'%s\'\n",
            events[task_id].p_busid[j],events[task_id].p_genid[j].c_str());
        if (print_calcs) printf("p[%d] Generator: (bus) %d (generator ID) \'%s\'\n",
            pf_network->communicator().rank(),
            events[task_id].p_busid[j],events[task_id].p_genid[j].c_str());
      }
    } else if (events[task_id].p_type == gridpack::powerflow::DCLine) {
      std::string lines;
      for (size_t j=0; j<events[task_id].p_dclines.size(); j++) {
        if (j > 0) lines += ", ";
        lines += events[task_id].p_dclines[j];
      }
      snprintf(sbuf, sizeof(sbuf), " DC line: %s\n", lines.c_str());
      if (print_calcs) printf("p[%d] DC line: %s\n",
          pf_network->communicator().rank(), lines.c_str());
    }
    if (print_calcs) pf_app.writeHeader(sbuf);
    timer->start(t_case_apply);
    bool contingencyFound = false;
    if (known) {
      // The GPU solution sets every voltage, so no reset is needed
      contingencyFound = pf_app.setKnownContingency(events[task_id]);
    } else {
      // Reset all voltages back to their original values
      pf_app.resetVoltages();
      // Sync ghost bus data after voltage reset to ensure branches connected to
      // ghost buses use the correct reset voltages in power flow calculation
      pf_network->updateBuses();
      // Set contingency
      contingencyFound = pf_app.setContingency(events[task_id]);
    }
    if (!contingencyFound) {
      printf("WARNING: Contingency '%s' - elements not found or no valid slack bus\n",
             events[task_id].p_name.c_str());
    }
    // Check for islanding before attempting to solve
    // Note: lone bus isolation is handled separately as a warning, not a failure
    int islandCount = pf_app.getIslandCount();
    bool hasLoneBus = pf_app.hasLoneBus();
    bool islandDetected = (islandCount > 1);
    // Skip power flow if contingency setup failed (no valid slack) or islanding detected
    bool slackCapacityOk = true;  // Will be checked after solve
    bool solveOk = false;
    bool hasSolveRecord = false;
    timer->stop(t_case_apply);
    if (gpu) {
      timer->start(t_case_inject);
      // The converter injections the GPU solution was found with
      if (!gpu->dc.empty()) pf_app.setHVDCSolutions(gpu->dc);
      if (known) {
        pf_app.setKnownExternalSolution(events[task_id], gpu->v, gpu->theta,
                                        gpu->qlim_conversion, gpu->q_required,
                                        gpu->convergence);
      } else {
        pf_app.setExternalSolution(gpu->v, gpu->theta, gpu->qlim_conversion,
                                   gpu->q_required, gpu->convergence);
      }
      solveOk = true;
      hasSolveRecord = true;
      timer->stop(t_case_inject);
    } else if (contingencyFound && !islandDetected) {
      timer->start(t_case_solve);
      try {
        solveOk = pf_app.solve();
        if (solveOk && check_Qlim && !pf_app.checkQlimViolations()) {
          pf_app.solve();
        }
        hasSolveRecord = true;
      } catch (const std::exception& e) {
        printf("p[%d] hit exception: %s\n", world.rank(), e.what());
        printf("Solver failure\n");
        solveOk = false;
      } catch (...) {
        printf("p[%d] hit unknown exception during solve\n", world.rank());
        solveOk = false;
      }
      timer->stop(t_case_solve);
    }
    timer->start(t_case_report);
    if (solveOk) {
      // Write PV->PQ conversion warnings to output file
      if (print_calcs && check_Qlim) {
        std::vector<std::string>& warnings = gridpack::powerflow::PFBus::getQlimWarnings();
        for (size_t w = 0; w < warnings.size(); w++) {
          pf_app.print(warnings[w].c_str());
        }
      }
      // Check if slack bus generator exceeds capacity
      slackCapacityOk = pf_app.checkSlackCapacity();
      if (!slackCapacityOk) {
        // Slack generator exceeds Pmax - insufficient generation capacity
        // This is treated as a failure, similar to divergence
        if (outputFormat == "json" || outputFormat == "csv") {
          gridpack::utility::ContingencyResult ctResult;
          ctResult.name = events[task_id].p_name;
          ctResult.type = contingencyTypeName(events[task_id]);
          ctResult.hasVoltageViolation = false;
          ctResult.hasBranchViolation = false;
          ctResult.solution.convergence = pf_app.getConvergence();
          ctResult.solution.convergence.converged = false;
          localContingencies.push_back(ctResult);
        }
        recordConv(task_id, "SLACK_OVERLOAD", std::string());
        snprintf(sbuf, sizeof(sbuf),"\nInsufficient generation capacity for contingency %s\n",
            events[task_id].p_name.c_str());
        if (print_calcs) pf_app.print(sbuf);
        addFailedStatColumns(task_id);
      } else {
        // Power flow solved and slack within capacity
        // If power flow solution is successful, write out voltages and currents
        if (print_calcs) pf_app.write();
        // Check for violations
        bool ok1 = true;
        bool ok2 = true;
        if (known) {
          pf_app.touchLineCheckBuses();
        } else {
          ok1 = pf_app.checkVoltageViolations();
          ok2 = pf_app.checkLineOverloadViolations();
        }
        bool ok = ok1 && ok2;
        // text mode runs the summary path but discards the per-ct struct.
        if (outputFormat == "json" || outputFormat == "csv" ||
            outputFormat == "text") {
          gridpack::utility::ContingencyResult ctResult;
          ctResult.name = events[task_id].p_name;
          ctResult.type = contingencyTypeName(events[task_id]);
          ctResult.hasVoltageViolation = !ok1;
          ctResult.hasBranchViolation = !ok2;
          ctResult.solution = pf_app.collectResults();
          filterResults(ctResult.solution);
          populateViolations(ctResult, static_cast<int>(task_id) + 1);
          if (outputFormat != "text") localContingencies.push_back(ctResult);
        }
        timer->start(t_case_rows);
        if (outputFormat == "csv_flat") {
          captureFlatRows(task_id + 1, events[task_id].p_name, true, false);
        }
        if (deltaRows) {
          captureDeltaRows(task_id + 1, events[task_id], true);
        }
        timer->stop(t_case_rows);
        recordConv(task_id, "OK", std::string());
      // Include results of violation checks in output
      if (ok) {
        snprintf(sbuf, sizeof(sbuf),"\nNo violation for contingency %s\n",
            events[task_id].p_name.c_str());
      }
      // Report bus voltage violations
      if (!ok1) {
        snprintf(sbuf, sizeof(sbuf),"\nBus Violation for contingency %s\n",
            events[task_id].p_name.c_str());
      } else if (!ok) {
        snprintf(sbuf, sizeof(sbuf),"\nNo Bus Violation for contingency %s\n",
            events[task_id].p_name.c_str());
      }
      if (print_calcs) pf_app.print(sbuf);
      if (print_calcs) pf_app.writeCABus();
      // Report branch overload violations
      if (!ok2) {
        // Keep in step with the row format in PFBranch::serialWrite("flow").
        snprintf(sbuf, sizeof(sbuf),"\nBranch Violation for contingency %s\n"
            "  From Bus    To Bus   CKT       P_from       Q_from"
            "     MVA_from         P_to         Q_to       MVA_to"
            "       Rate   Loading%%\n",
            events[task_id].p_name.c_str());
      } else if (!ok) {
        snprintf(sbuf, sizeof(sbuf),"\nNo Branch Violation for contingency %s\n",
            events[task_id].p_name.c_str());
      }

      if (print_calcs) pf_app.print(sbuf);
      if (print_calcs) pf_app.writeCABranch();
      // Get strings of data from power flow calculation and parse them to
      // extract numerical values. Store these values in vectors and then
      // add them to StatBlock objects
      if (write_stats) {
        timer->start(t_store);
        vmag.clear();
        vang.clear();
        mask.clear();
        mag_mask.clear();
        v_vals.clear();
        v_vals = pf_app.writeBusString("vr_str");
        nsize = v_vals.size();
        for (i=0; i<nsize; i++) {
          std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
          if (!busMonitored(atoi(tokens[0].c_str()))) continue;
          int not_isolated = atoi(tokens[3].c_str());
          if (not_isolated == 1) {
            vmag.push_back(atof(tokens[2].c_str()));
            if (atoi(tokens[4].c_str()) != 0) {
              mag_mask.push_back(2);
            } else {
              mag_mask.push_back(1);
            }
          }
          vang.push_back(atof(tokens[1].c_str()));
          mask.push_back(1);
        }
        if (task_comm.rank() == 0) {
          if (vmag_stats) vmag_stats->addColumnValues(task_id+1,vmag,mag_mask);
          if (vang_stats) vang_stats->addColumnValues(task_id+1,vang,mask);
        }
        pgen.clear();
        qgen.clear();
        mask.clear();
        v_vals.clear();
        v_vals = pf_app.writeBusString("power");
        nsize = v_vals.size();
        for (i=0; i<nsize; i++) {
          std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
          if (tokens.size()%4 != 0) {
            printf("Incorrect generator listing\n");
            continue;
          }
          int ngen = tokens.size()/4;
          for (j=0; j<ngen; j++) {
            if (!busMonitored(atoi(tokens[j*4].c_str()))) continue;
            pgen.push_back(atof(tokens[j*4+2].c_str()));
            qgen.push_back(atof(tokens[j*4+3].c_str()));
            mask.push_back(1);
          }
        }
        if (task_comm.rank() == 0 && pgen_stats) {
          pgen_stats->addColumnValues(task_id+1,pgen,mask);
          qgen_stats->addColumnValues(task_id+1,qgen,mask);
        }
        pflow.clear();
        qflow.clear();
        perf.clear();
        mask.clear();
        v_vals.clear();
        v_vals = pf_app.writeBranchString("flow_str");
        nsize = v_vals.size();
        for (i=0; i<nsize; i++) {
          std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
          if (tokens.size()%8 != 0) {
            printf("Incorrect branch power flow listing\n");
            continue;
          }
          int nline = tokens.size()/8;
          for (j=0; j<nline; j++) {
            if (!branchMonitored(atoi(tokens[j*8].c_str()),
                                 atoi(tokens[j*8+1].c_str()),
                                 tokens[j*8+2])) continue;
            pflow.push_back(atof(tokens[j*8+3].c_str()));
            qflow.push_back(atof(tokens[j*8+4].c_str()));
            perf.push_back(atof(tokens[j*8+5].c_str()));
            if (atoi(tokens[j*8+7].c_str()) == 0) {
              mask.push_back(1);
            } else {
              mask.push_back(2);
            }
          }
        }
        if (task_comm.rank() == 0 && pflow_stats) {
          pflow_stats->addColumnValues(task_id+1,pflow,mask);
          qflow_stats->addColumnValues(task_id+1,qflow,mask);
          perf_stats->addColumnValues(task_id+1,perf,mask);
        }
        timer->stop(t_store);
      }
        // Note: clearQlimViolations() moved after unSetContingency() below
      }  // end slackCapacityOk block
    } else {
      if (outputFormat == "json" || outputFormat == "csv") {
        gridpack::utility::ContingencyResult ctResult;
        ctResult.name = events[task_id].p_name;
        ctResult.type = contingencyTypeName(events[task_id]);
        ctResult.hasVoltageViolation = false;
        ctResult.hasBranchViolation = false;
        ctResult.solution.convergence = pf_app.getConvergence();
        ctResult.solution.convergence.converged = false;
        localContingencies.push_back(ctResult);
      }
      {
        const char *st;
        if (islandDetected) {
          st = "ISLANDED";
        } else if (!contingencyFound) {
          st = "NO_SLACK";
        } else {
          st = "DIVERGED";
        }
        recordConv(task_id, st, std::string());
      }
      if (islandDetected) {
        snprintf(sbuf, sizeof(sbuf),"\nIslanding detected for contingency %s (%d islands)\n",
            events[task_id].p_name.c_str(), islandCount);
      } else if (!contingencyFound) {
        snprintf(sbuf, sizeof(sbuf),"\nNo valid slack bus for contingency %s\n",
            events[task_id].p_name.c_str());
      } else {
        snprintf(sbuf, sizeof(sbuf),"\nDivergent for contingency %s\n",
            events[task_id].p_name.c_str());
      }
      if (print_calcs) pf_app.print(sbuf);
      addFailedStatColumns(task_id);
    }
    if (gpuPath.active()) {
      using gridpack::batchpf::ReportStatus;
      if (islandDetected) report.status = ReportStatus::Islanded;
      else if (!contingencyFound) report.status = ReportStatus::NoSlack;
      else if (!solveOk) report.status = ReportStatus::Diverged;
      else if (!slackCapacityOk) report.status = ReportStatus::SlackOverload;
      else report.status = ReportStatus::Ok;
      if (hasSolveRecord) {
        const auto cs = pf_app.getConvergence();
        report.iterations = cs.iterations;
        report.final_tolerance = cs.finalTolerance;
      }
      // Cleanup restores PV flags, so capture the solved classification first.
      for (int b = 0; b < pf_network->numBuses(); ++b) {
        if (!pf_network->getActiveBus(b)) continue;
        auto *bus = dynamic_cast<gridpack::powerflow::PFBus *>(pf_network->getBus(b).get());
        if (bus->isIsolated() || bus->getReferenceBus()) continue;
        if (bus->isPV()) ++report.pv_buses;
        else ++report.pq_buses;
      }
    }
    timer->stop(t_case_report);
    timer->start(t_case_restore);
    // Return network to its original base case state
    // Clear Q limit violations AFTER unSetContingency so generators are restored first.
    // This ensures clearQlim() sees the correct generator status when deciding
    // whether to restore p_isPV (PV bus status).
    if (known) {
      pf_app.unSetKnownContingency(events[task_id]);
      if (check_Qlim) pf_app.clearKnownQlimViolations(events[task_id], gpu->qlim_conversion);
    } else {
      pf_app.unSetContingency(events[task_id]);
      if (check_Qlim) pf_app.clearQlimViolations();
    }
    // Clear Q limit warnings for next contingency
    gridpack::powerflow::PFBus::clearQlimWarnings();
    // Remote regulation can change resetVoltages()'s reference. Each outage
    // starts from the prepared grid, including when acceleration is disabled.
    for (int b = 0; b < static_cast<int>(voltageResetReference.size()); ++b) {
      auto *bus = dynamic_cast<gridpack::powerflow::PFBus *>(pf_network->getBus(b).get());
      if (bus->getInitialVoltage() != voltageResetReference[b]) {
        bus->setVoltageMag(voltageResetReference[b]);
      }
    }
    timer->stop(t_case_restore);
    // Close output file for this contingency
    if (print_calcs) pf_app.close();
    return report;
  };
  timer->stop(t_list);
  if (gpuPath.active()) {
    gpuPath.prepare(pf_app, pf_network, events, check_Qlim, outputFile);
  }
  timer->start(t_cases);
  if (gpuPath.active()) {
    gpuPath.run(processCase);
    gpuPath.finish();
  } else {
    // nextTask returns the same task_id on all processors in task_comm.
    // When the calculation runs out of task, nextTask will return false.
    while (taskmgr.nextTask(task_comm, &task_id)) {
      processCase(task_id, nullptr);
    }
  }
  timer->stop(t_cases);
  timer->start(t_merge);
  // csv_flat / csv_delta: each rank streamed rows to its .part file during
  // the loop. Close, sync, then world rank 0 writes header + concatenates.
  if (outputFormat == "csv_flat") {
    if (flatPart.is_open()) flatPart.close();
  }
  if (outputFormat == "csv_delta") {
    if (deltaPart.is_open()) deltaPart.close();
  }
  if (parquetOut) {
    // Every rank writes one consecutive range of events, taken from all
    // ranks' part files, to <outputFile>_flows/part-<rank>.parquet. Reading
    // the files in name order gives event order (guide R5), and the
    // compression runs on all ranks at once.
    using gridpack::contingency_analysis::ParquetFlows;
    const MPI_Comm comm = static_cast<MPI_Comm>(world);
    std::vector<long long> local;
    try {
      flowsPart->close();
      for (const ParquetFlows::RowGroup &g : flowsPart->rowGroups()) {
        local.push_back(g.event_idx);
        local.push_back(g.part);
        local.push_back(g.index);
        local.push_back(g.rows);
      }
    } catch (const std::exception &e) {
      std::cout << "ERROR: " << e.what() << '\n' << std::flush;
      MPI_Abort(comm, 1);
    }
    int nlocal = static_cast<int>(local.size());
    std::vector<int> counts(world.size()), displs(world.size(), 0);
    MPI_Allgather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    for (int p = 1; p < world.size(); p++) displs[p] = displs[p - 1] + counts[p - 1];
    std::vector<long long> all(displs.back() + counts.back());
    MPI_Allgatherv(local.data(), nlocal, MPI_LONG_LONG, all.data(), counts.data(),
                   displs.data(), MPI_LONG_LONG, comm);
    std::vector<ParquetFlows::RowGroup> groups;
    long long totalRows = 0;
    for (size_t k = 0; k + 3 < all.size(); k += 4) {
      ParquetFlows::RowGroup g;
      g.event_idx = static_cast<int32_t>(all[k]);
      g.part = static_cast<int32_t>(all[k + 1]);
      g.index = static_cast<int32_t>(all[k + 2]);
      g.rows = all[k + 3];
      totalRows += g.rows;
      groups.push_back(g);
    }
    // Rank order then row-group order for equal events, as gathered
    std::stable_sort(groups.begin(), groups.end(),
                     [](const ParquetFlows::RowGroup &a, const ParquetFlows::RowGroup &b) {
                       return a.event_idx < b.event_idx;
                     });
    // Consecutive ranges of about equal rows; a case is never split
    std::vector<ParquetFlows::RowGroup> mine;
    const long long share = (totalRows + world.size() - 1) / std::max(1, world.size());
    long long before = 0;
    for (const ParquetFlows::RowGroup &g : groups) {
      const int owner = (share > 0) ? static_cast<int>(std::min<long long>(
                                          before / share, world.size() - 1)) : 0;
      if (owner == world.rank()) mine.push_back(g);
      before += g.rows;
    }
    std::vector<std::string> parts;
    for (int p = 0; p < world.size(); p++) {
      std::ostringstream oss;
      oss << outputFile << "_flows." << p << ".parquet.part";
      parts.push_back(oss.str());
    }
    const std::string flowsDir = outputFile + "_flows";
    if (world.rank() == 0) {
      // A fresh directory: remove part files of an earlier run
      std::error_code ec;
      std::filesystem::create_directories(flowsDir, ec);
      for (const auto &entry : std::filesystem::directory_iterator(flowsDir, ec)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("part-", 0) == 0 && entry.path().extension() == ".parquet") {
          std::filesystem::remove(entry.path(), ec);
        }
      }
    }
    world.sync();
    long written = 0;
    if (!mine.empty()) {
      constexpr int rank_digits = 5;
      std::ostringstream name;
      name << "/part-" << std::setw(rank_digits) << std::setfill('0') << world.rank()
           << ".parquet";
      try {
        written = ParquetFlows::writeRange(parts, mine, flowsDir + name.str());
      } catch (const std::exception &e) {
        std::cout << "ERROR: " << e.what() << '\n' << std::flush;
        MPI_Abort(comm, 1);
      }
    }
    world.sync();
    std::remove(flowsPartPath.c_str());
    long files = mine.empty() ? 0 : 1;
    world.sum(&written, 1);
    world.sum(&files, 1);
    if (world.rank() == 0) {
      std::cout << "[parquet] wrote " << written << " rows to " << flowsDir << "/ ("
                << files << " files)\n";
    }
  }
  // The tables are written by all ranks at once, each copying its own part
  // file's rows into place (ca_parallel_write.hpp): event order on the batch
  // path (guide R5), rank order otherwise, byte for byte as rank 0 alone
  // wrote them before
  const MPI_Comm writeComm = static_cast<MPI_Comm>(world);
  auto writeTable = [&](const char *suffix, const char *header, const char *tag,
                        const char *outName) {
    std::ostringstream oss;
    oss << outputFile << suffix << world.rank() << ".part";
    const std::string outFile = outputFile + outName;
    const long long rows = gridpack::contingency_analysis::writePartsInParallel(
        writeComm, outFile, header, oss.str(), gpuPath.active());
    if (world.rank() == 0) {
      std::cout << "[" << tag << "] wrote " << rows << " rows to " << outFile << '\n';
    }
  };
  if (wantBusSidecar) {
    world.sync();
    if (outputFormat == "csv_flat") {
      writeTable("_flat.",
                 "event_idx,contingency,from_bus,to_bus,circuit_id,"
                 "p_from_mw,q_from_mvar,mva_from,rate_mva,loading_percent,"
                 "viol,v_from_pu,v_to_pu,ang_from_deg,ang_to_deg\n",
                 "csv_flat",
                 "_flat.csv");
    }
    if (outputFormat == "csv_delta") {
      writeTable("_delta.",
                 "event_idx,contingency,type,from_bus,to_bus,ckt,"
                 "base_kv_from,base_kv_to,area_from,area_to,base_rate_mva,cont_rate_mva,"
                 "base_p_mw,cont_p_mw,base_q_mvar,cont_q_mvar,"
                 "base_mva,cont_mva,base_loading_pct,cont_loading_pct,"
                 "v_from_base,v_from_cont,v_to_base,v_to_cont,"
                 "ang_from_base,ang_from_cont,ang_to_base,ang_to_cont,"
                 // ",cont_event_facility" here if re-enabling the column
                 "d_v_base,d_v_cont,d_angle_base,d_angle_cont\n",
                 "csv_delta",
                 "_delta.csv");
    }
    if (world.rank() == 0) {
      // Bus metadata sidecar (deduped by bus_id, first writer wins).
      std::string busFile = outputFile + "_buses.csv";
      std::ofstream bout(busFile.c_str(),
                         std::ios::out | std::ios::trunc | std::ios::binary);
      bout << "bus_id,bus_name,base_kv,area,zone,owner,"
              "area_name,zone_name,owner_name\n";
      std::set<int> seen_bus;
      size_t bus_rows = 0;
      for (int p = 0; p < world.size(); p++) {
        std::ostringstream oss;
        oss << outputFile << "_buses." << p << ".part";
        std::string part = oss.str();
        std::ifstream fin(part.c_str());
        if (!fin) continue;
        std::string line;
        while (std::getline(fin, line)) {
          if (line.empty()) continue;
          size_t comma = line.find(',');
          if (comma == std::string::npos) continue;
          int bus_id = std::atoi(line.substr(0, comma).c_str());
          if (seen_bus.insert(bus_id).second) {
            bout << line << "\n";
            bus_rows++;
          }
        }
        fin.close();
        std::remove(part.c_str());
      }
      bout.close();
      printf("[buses] wrote %zu rows to %s\n", bus_rows, busFile.c_str());
    }
  }
  // <outputFile>_violations.csv from the per-rank parts, for every
  // outputFormat; a rank that emitted no rows has no part file.
  if (violPart.is_open()) violPart.close();
  world.sync();
  writeTable("_violations.",
             "event_idx,contingency,type,element,mva_or_vpu,rate_or_limit,"
             "loading_percent,base_mva,delta,severity\n",
             "violations",
             "_violations.csv");

  // Aggregate skip count across ranks for diagnostics.
  if (deltaRows) {
    long localSkip = static_cast<long>(deltaSkipCount);
    long totalSkip = localSkip;
    world.sum(&totalSkip, 1);
    if (world.rank() == 0 && totalSkip > 0) {
      std::cout << "[" << outputFormat << "] " << totalSkip
                << " branch rows had no base-cache match\n";
    }
  }

  // Aggregate per-run summary across all ranks. Emit <outputFile>_summary.json
  // on rank 0. Counters and worst-of values follow the same shape commercial
  // tools use in their contingency reports.
  {
    // 0: total_ct  1: converged  2: cts_with_branch_viol  3: cts_with_voltage_viol
    // 4: islanded  5: no_slack   6: diverged                7: slack_overload
    long localCounters[8] = { 0 };
    if (!localContingencies.empty()) {
      // json/csv paths: authoritative per-ct list.
      for (size_t ci = 0; ci < localContingencies.size(); ci++) {
        const gridpack::utility::ContingencyResult &ct = localContingencies[ci];
        localCounters[0] += 1;
        if (ct.solution.convergence.converged) localCounters[1] += 1;
      }
    } else {
      // text/csv_flat/csv_delta: count from convergence rows.
      localCounters[0] = static_cast<long>(localConvRows.size());
      long conv = 0;
      for (size_t i = 0; i < localConvRows.size(); i++) {
        if (localConvRows[i].status == "OK") conv++;
      }
      localCounters[1] = conv;
    }
    localCounters[2] = static_cast<long>(ctsWithBranchViol.size());
    localCounters[3] = static_cast<long>(ctsWithVoltageViol.size());
    // Per-status breakdown from convergence rows (populated in every mode).
    for (size_t i = 0; i < localConvRows.size(); i++) {
      const std::string &st = localConvRows[i].status;
      if      (st == "ISLANDED")       localCounters[4] += 1;
      else if (st == "NO_SLACK")       localCounters[5] += 1;
      else if (st == "DIVERGED")       localCounters[6] += 1;
      else if (st == "SLACK_OVERLOAD") localCounters[7] += 1;
    }
    long totalCounters[8] = { 0 };
    for (int i = 0; i < 8; i++) totalCounters[i] = localCounters[i];
    world.sum(&totalCounters[0], 8);
    long localViolRows  = static_cast<long>(violRowCount);
    long totalViolRows  = localViolRows;
    world.sum(&totalViolRows, 1);

    struct WorstBranchWire {
      double loading_pct;
      int from, to;
      char ckt[4];
      char ct_name[32];
    };
    struct WorstVoltageWire {
      double v_pu;
      double dev_pu;
      int bus_id;
      char ct_name[32];
    };
    WorstBranchWire wbLo;
    wbLo.loading_pct = worstBr.loading_pct;
    wbLo.from = worstBr.from; wbLo.to = worstBr.to;
    std::strncpy(wbLo.ckt, worstBr.ckt.c_str(), 3); wbLo.ckt[3] = '\0';
    std::strncpy(wbLo.ct_name, worstBr.ct_name.c_str(), 31); wbLo.ct_name[31] = '\0';
    WorstVoltageWire wvLo;
    wvLo.v_pu = worstVLo.v_pu; wvLo.dev_pu = worstVLo.dev_pu;
    wvLo.bus_id = worstVLo.bus_id;
    std::strncpy(wvLo.ct_name, worstVLo.ct_name.c_str(), 31); wvLo.ct_name[31] = '\0';
    WorstVoltageWire wvHi;
    wvHi.v_pu = worstVHi.v_pu; wvHi.dev_pu = worstVHi.dev_pu;
    wvHi.bus_id = worstVHi.bus_id;
    std::strncpy(wvHi.ct_name, worstVHi.ct_name.c_str(), 31); wvHi.ct_name[31] = '\0';
    MPI_Comm mpi_comm = static_cast<MPI_Comm>(world);
    if (world.rank() == 0) {
      for (int p = 1; p < world.size(); p++) {
        WorstBranchWire  otherB;
        WorstVoltageWire otherLo, otherHi;
        MPI_Recv(&otherB,  sizeof(otherB),  MPI_BYTE, p, 30, mpi_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&otherLo, sizeof(otherLo), MPI_BYTE, p, 31, mpi_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&otherHi, sizeof(otherHi), MPI_BYTE, p, 32, mpi_comm, MPI_STATUS_IGNORE);
        if (otherB.loading_pct > wbLo.loading_pct) wbLo = otherB;
        if (otherLo.dev_pu < wvLo.dev_pu)          wvLo = otherLo;
        if (otherHi.dev_pu > wvHi.dev_pu)          wvHi = otherHi;
      }
    } else {
      MPI_Send(&wbLo, sizeof(wbLo), MPI_BYTE, 0, 30, mpi_comm);
      MPI_Send(&wvLo, sizeof(wvLo), MPI_BYTE, 0, 31, mpi_comm);
      MPI_Send(&wvHi, sizeof(wvHi), MPI_BYTE, 0, 32, mpi_comm);
    }

    // Gather per-ct PI/VPI/Vdev/rosters + per-ct worst-single-element data.
    // Blob line: name pi vpi vdev worstLoad worstVdevAbs worstVpu br_flag v_flag
    std::map<std::string, double> aggPi;
    std::map<std::string, double> aggVpi;
    std::map<std::string, double> aggVdev;
    std::map<std::string, double> aggWorstLoading;
    std::map<std::string, double> aggWorstVdev;    // unsigned max |v-limit|
    std::map<std::string, double> aggWorstVpu;     // v_pu that produced aggWorstVdev
    std::set<std::string> aggBranchViol;
    std::set<std::string> aggVoltageViol;
    {
      std::ostringstream localOut;
      localOut << std::setprecision(10);
      std::set<std::string> names;
      for (std::map<std::string,double>::const_iterator it = ctPi.begin();
           it != ctPi.end(); ++it) names.insert(it->first);
      for (std::map<std::string,double>::const_iterator it = ctVpi.begin();
           it != ctVpi.end(); ++it) names.insert(it->first);
      for (std::map<std::string,double>::const_iterator it = ctVdev.begin();
           it != ctVdev.end(); ++it) names.insert(it->first);
      for (std::set<std::string>::const_iterator it = ctsWithBranchViol.begin();
           it != ctsWithBranchViol.end(); ++it) names.insert(*it);
      for (std::set<std::string>::const_iterator it = ctsWithVoltageViol.begin();
           it != ctsWithVoltageViol.end(); ++it) names.insert(*it);
      auto lookup = [](const std::map<std::string,double> &m,
                       const std::string &k) -> double {
        std::map<std::string,double>::const_iterator it = m.find(k);
        return (it == m.end()) ? 0.0 : it->second;
      };
      for (std::set<std::string>::const_iterator it = names.begin();
           it != names.end(); ++it) {
        const std::string &nm = *it;
        double pi    = lookup(ctPi,  nm);
        double vpi   = lookup(ctVpi, nm);
        double vdev  = lookup(ctVdev, nm);
        double wL    = lookup(ctWorstLoading, nm);
        double wDabs = lookup(ctWorstVdev, nm);
        double wV    = lookup(ctWorstVpu,  nm);
        int brFlag = ctsWithBranchViol.count(nm)  ? 1 : 0;
        int vFlag  = ctsWithVoltageViol.count(nm) ? 1 : 0;
        localOut << nm << '\t' << pi << '\t' << vpi << '\t' << vdev << '\t'
                 << wL << '\t' << wDabs << '\t' << wV << '\t'
                 << brFlag << '\t' << vFlag << '\n';
      }
      std::string localBlob = localOut.str();
      auto splitTabs = [](const std::string &line,
                          std::vector<std::string> &out) {
        out.clear();
        size_t pos = 0;
        while (pos <= line.size()) {
          size_t t = line.find('\t', pos);
          if (t == std::string::npos) {
            out.push_back(line.substr(pos));
            break;
          }
          out.push_back(line.substr(pos, t - pos));
          pos = t + 1;
        }
      };
      auto absorb = [&](const std::string &blob) {
        size_t pos = 0;
        std::vector<std::string> fields;
        while (pos < blob.size()) {
          size_t eol = blob.find('\n', pos);
          if (eol == std::string::npos) break;
          std::string line = blob.substr(pos, eol - pos);
          pos = eol + 1;
          splitTabs(line, fields);
          if (fields.size() < 9) continue;
          const std::string &nm = fields[0];
          double pi    = std::atof(fields[1].c_str());
          double vpi   = std::atof(fields[2].c_str());
          double vdev  = std::atof(fields[3].c_str());
          double wL    = std::atof(fields[4].c_str());
          double wDabs = std::atof(fields[5].c_str());
          double wV    = std::atof(fields[6].c_str());
          int brF      = std::atoi(fields[7].c_str());
          int vF       = std::atoi(fields[8].c_str());
          if (pi   != 0.0) aggPi[nm]   += pi;
          if (vpi  != 0.0) aggVpi[nm]  += vpi;
          if (vdev != 0.0) aggVdev[nm] += vdev;
          if (wL > aggWorstLoading[nm])                aggWorstLoading[nm] = wL;
          if (wDabs > aggWorstVdev[nm]) { aggWorstVdev[nm] = wDabs; aggWorstVpu[nm] = wV; }
          if (brF) aggBranchViol.insert(nm);
          if (vF)  aggVoltageViol.insert(nm);
        }
      };
      if (world.rank() == 0) {
        absorb(localBlob);
        for (int p = 1; p < world.size(); p++) {
          int len = 0;
          MPI_Recv(&len, 1, MPI_INT, p, 33, mpi_comm, MPI_STATUS_IGNORE);
          std::string remote;
          remote.resize(len);
          if (len > 0) {
            MPI_Recv(&remote[0], len, MPI_CHAR, p, 34, mpi_comm,
                     MPI_STATUS_IGNORE);
          }
          absorb(remote);
        }
      } else {
        int len = static_cast<int>(localBlob.size());
        MPI_Send(&len, 1, MPI_INT, 0, 33, mpi_comm);
        if (len > 0) {
          MPI_Send(const_cast<char*>(localBlob.c_str()), len, MPI_CHAR, 0, 34,
                   mpi_comm);
        }
      }
    }

    if (world.rank() == 0) {
      std::string sumFile = outputFile + "_summary.json";
      std::ofstream sout(sumFile.c_str());
      sout << std::fixed;
      sout << "{\n";
      sout << "  \"total_contingencies\": "     << totalCounters[0] << ",\n";
      sout << "  \"converged\": "               << totalCounters[1] << ",\n";
      sout << "  \"diverged\": "                << (totalCounters[0] - totalCounters[1]) << ",\n";
      // Per-status breakdown of the diverged bucket. Sum equals `diverged`.
      sout << "  \"islanded\": "                << totalCounters[4] << ",\n";
      sout << "  \"no_slack\": "                << totalCounters[5] << ",\n";
      sout << "  \"solver_diverged\": "         << totalCounters[6] << ",\n";
      sout << "  \"slack_overload\": "          << totalCounters[7] << ",\n";
      sout << "  \"with_branch_violation\": "   << totalCounters[2] << ",\n";
      sout << "  \"with_voltage_violation\": "  << totalCounters[3] << ",\n";
      sout << "  \"worst_loading\": ";
      if (wbLo.loading_pct > 0.0) {
        sout << "{\"contingency\": \"" << wbLo.ct_name << "\""
             << ", \"from_bus\": " << wbLo.from
             << ", \"to_bus\": "   << wbLo.to
             << ", \"circuit_id\": \"" << wbLo.ckt << "\""
             << ", \"loading_percent\": " << std::setprecision(2) << wbLo.loading_pct
             << "},\n";
      } else {
        sout << "null,\n";
      }
      sout << "  \"worst_voltage_low\": ";
      if (wvLo.dev_pu < 0.0) {
        sout << "{\"contingency\": \"" << wvLo.ct_name << "\""
             << ", \"bus_id\": " << wvLo.bus_id
             << ", \"v_pu\": "          << std::setprecision(6) << wvLo.v_pu
             << ", \"deviation_pu\": "  << std::setprecision(6) << wvLo.dev_pu
             << "},\n";
      } else {
        sout << "null,\n";
      }
      sout << "  \"worst_voltage_high\": ";
      if (wvHi.dev_pu > 0.0) {
        sout << "{\"contingency\": \"" << wvHi.ct_name << "\""
             << ", \"bus_id\": " << wvHi.bus_id
             << ", \"v_pu\": "         << std::setprecision(6) << wvHi.v_pu
             << ", \"deviation_pu\": " << std::setprecision(6) << wvHi.dev_pu
             << "},\n";
      } else {
        sout << "null,\n";
      }
      auto emitNameArray = [&](const char *field,
                               const std::set<std::string> &names) {
        sout << "  \"" << field << "\": [";
        int emitted = 0;
        for (std::set<std::string>::const_iterator it = names.begin();
             it != names.end(); ++it, ++emitted) {
          if (emitted) sout << ", ";
          sout << "\"" << *it << "\"";
        }
        sout << "],\n";
      };
      emitNameArray("contingencies_with_branch_violation", aggBranchViol);
      emitNameArray("contingencies_with_voltage_violation", aggVoltageViol);
      // top_severe_contingencies: Group A (any violation) first, sorted by
      // worst-single-element severity; Group B (no violations) after,
      // sorted by composite_pi. Combined list capped at topN.
      {
        std::set<std::string> ctSet;
        for (std::map<std::string,double>::const_iterator it = aggPi.begin();
             it != aggPi.end(); ++it) ctSet.insert(it->first);
        for (std::map<std::string,double>::const_iterator it = aggVpi.begin();
             it != aggVpi.end(); ++it) ctSet.insert(it->first);
        for (std::set<std::string>::const_iterator it = aggBranchViol.begin();
             it != aggBranchViol.end(); ++it) ctSet.insert(*it);
        for (std::set<std::string>::const_iterator it = aggVoltageViol.begin();
             it != aggVoltageViol.end(); ++it) ctSet.insert(*it);
        auto agg_get = [](const std::map<std::string,double> &m,
                          const std::string &k) -> double {
          std::map<std::string,double>::const_iterator it = m.find(k);
          return (it == m.end()) ? 0.0 : it->second;
        };
        struct SevRow {
          std::string name;
          bool violated;
          double sortKey;   // Group A: worst-single severity; Group B: composite_pi
          double composite, branchPi, voltagePi;
          double worstLoading;
          double worstVpu, worstVdev;   // worstVdev is unsigned |v - limit|
        };
        std::vector<SevRow> groupA, groupB;
        for (std::set<std::string>::const_iterator it = ctSet.begin();
             it != ctSet.end(); ++it) {
          SevRow r;
          r.name = *it;
          r.branchPi   = agg_get(aggPi, *it);
          r.voltagePi  = agg_get(aggVpi, *it);
          r.composite  = piBranchWeight * r.branchPi + piVoltageWeight * r.voltagePi;
          r.worstLoading = agg_get(aggWorstLoading, *it);
          r.worstVdev  = agg_get(aggWorstVdev, *it);
          r.worstVpu   = agg_get(aggWorstVpu,  *it);
          bool hasBr = aggBranchViol.count(*it) > 0;
          bool hasV  = aggVoltageViol.count(*it) > 0;
          r.violated = hasBr || hasV;
          if (r.violated) {
            double s = 0.0;
            if (r.worstLoading > 100.0) s = std::max(s, r.worstLoading - 100.0);
            if (r.worstVdev    > 0.0)   s = std::max(s, r.worstVdev * 1000.0);
            r.sortKey = s;
            groupA.push_back(r);
          } else {
            r.sortKey = r.composite;
            if (r.sortKey > 0.0) groupB.push_back(r);
          }
        }
        auto sevCmp = [](const SevRow &a, const SevRow &b) {
          if (a.sortKey != b.sortKey) return a.sortKey > b.sortKey;
          return a.name < b.name;
        };
        std::sort(groupA.begin(), groupA.end(), sevCmp);
        std::sort(groupB.begin(), groupB.end(), sevCmp);
        auto emitRow = [&](const SevRow &r, bool first) {
          if (!first) sout << ",\n    ";
          else        sout << "\n    ";
          bool hasBr = aggBranchViol.count(r.name) > 0;
          bool hasV  = r.worstVdev > 0.0;
          sout << "{\"contingency\": \"" << r.name << "\""
               << ", \"composite_pi\": " << std::setprecision(6) << r.composite
               << ", \"worst_branch_loading_percent\": ";
          if (hasBr) sout << std::setprecision(2) << r.worstLoading;
          else       sout << "null";
          sout << ", \"worst_voltage_pu\": ";
          if (hasV) sout << std::setprecision(6) << r.worstVpu;
          else      sout << "null";
          sout << ", \"worst_voltage_deviation_pu\": ";
          if (hasV) sout << std::setprecision(6) << r.worstVdev;
          else      sout << "null";
          sout << ", \"has_branch_violation\": "
               << (hasBr ? "true" : "false")
               << ", \"has_voltage_violation\": "
               << (aggVoltageViol.count(r.name) ? "true" : "false")
               << "}";
        };
        sout << "  \"top_severe_contingencies\": [";
        int emitted = 0;
        for (size_t i = 0; i < groupA.size() && emitted < topN; ++i, ++emitted) {
          emitRow(groupA[i], emitted == 0);
        }
        for (size_t i = 0; i < groupB.size() && emitted < topN; ++i, ++emitted) {
          emitRow(groupB[i], emitted == 0);
        }
        sout << (emitted ? "\n  ]\n" : "]\n");
      }
      sout << "}\n";
      sout.close();
      printf("[summary] wrote %s (%ld contingencies, %ld converged, "
             "%ld with branch violations, %ld with voltage violations)\n",
             sumFile.c_str(),
             totalCounters[0], totalCounters[1],
             totalCounters[2], totalCounters[3]);
    }
  }

  // Print statistics from task manager describing the number of tasks performed
  // per processor
  taskmgr.printStats();

  // Sync GA before MPI collectives to flush any pending one-sided operations
  world.sync();

  // Export CA results to JSON or CSV.
  // Only rank 0 writes output files. Non-zero ranks send their serialized
  // data to rank 0 using point-to-point MPI send/recv.
  if (outputFormat == "json") {
    // Each process serializes its contingency results as JSON text
    std::ostringstream localJsonStream;
    for (size_t ci = 0; ci < localContingencies.size(); ci++) {
      gridpack::utility::ResultsExporter::writeContingencyResultJSON(
          localJsonStream, localContingencies[ci]);
      if (ci + 1 < localContingencies.size()) {
        localJsonStream << ",\n";
      }
    }
    std::string localJson = localJsonStream.str();

    // Gather all JSON fragments on rank 0 using point-to-point send/recv
    MPI_Comm mpi_comm = static_cast<MPI_Comm>(world);
    std::vector<std::string> allFragments(world.size());
    allFragments[0] = localJson;  // rank 0's own data
    if (world.rank() == 0) {
      for (int p = 1; p < world.size(); p++) {
        int len;
        MPI_Recv(&len, 1, MPI_INT, p, 0, mpi_comm, MPI_STATUS_IGNORE);
        allFragments[p].resize(len);
        if (len > 0) {
          MPI_Recv(&allFragments[p][0], len, MPI_CHAR, p, 1, mpi_comm,
                   MPI_STATUS_IGNORE);
        }
      }
    } else {
      int len = static_cast<int>(localJson.size());
      MPI_Send(&len, 1, MPI_INT, 0, 0, mpi_comm);
      if (len > 0) {
        MPI_Send(const_cast<char*>(localJson.c_str()), len, MPI_CHAR, 0, 1,
                 mpi_comm);
      }
    }

    // Rank 0 writes the final JSON file
    if (world.rank() == 0) {
      std::string jsonFile = outputFile + ".json";
      std::ofstream jout(jsonFile.c_str());
      gridpack::utility::ResultsExporter::writeCAJSONHeader(jout,
          baseCaseResults);
      bool firstFragment = true;
      for (int p = 0; p < world.size(); p++) {
        if (!allFragments[p].empty()) {
          if (!firstFragment) jout << ",\n";
          jout << allFragments[p];
          firstFragment = false;
        }
      }
      jout << "\n";
      gridpack::utility::ResultsExporter::writeCAJSONFooter(jout);
      jout.close();
    }
  }

  if (outputFormat == "csv") {
    // Convergence is emitted by the universal sidecar block below.
    std::ostringstream localBus, localBranch, localGen;
    localBus << std::fixed;
    localBranch << std::fixed;
    localGen << std::fixed;
    // Column layout must match the headers written by
    // ResultsExporter::writePFCSV for the base case.
    for (size_t ci = 0; ci < localContingencies.size(); ci++) {
      const gridpack::utility::ContingencyResult& ct = localContingencies[ci];
      const gridpack::utility::PowerFlowResults& r = ct.solution;
      for (size_t bi = 0; bi < r.buses.size(); bi++) {
        const gridpack::utility::BusResult& b = r.buses[bi];
        localBus << ct.name << ","
           << b.busId << "," << b.type << ","
           << b.area << "," << b.zone << ","
           << std::setprecision(2) << b.baseKV << ","
           << std::setprecision(6) << b.voltage << ","
           << std::setprecision(6) << b.angle << ","
           << std::setprecision(4) << b.pInjection << ","
           << std::setprecision(4) << b.qInjection << ","
           << std::setprecision(4) << b.pLoad << ","
           << std::setprecision(4) << b.qLoad << ","
           << std::setprecision(4) << b.pGen << ","
           << std::setprecision(4) << b.qGen << ","
           << std::setprecision(4) << b.shuntMvar << "\n";
      }
      for (size_t bi = 0; bi < r.branches.size(); bi++) {
        const gridpack::utility::BranchResult& br = r.branches[bi];
        localBranch << ct.name << ","
           << br.fromBus << "," << br.toBus << ","
           << br.circuitId << ","
           << std::setprecision(4) << br.pFrom << ","
           << std::setprecision(4) << br.qFrom << ","
           << std::setprecision(4) << br.pTo << ","
           << std::setprecision(4) << br.qTo << ","
           << std::setprecision(4) << br.pLoss << ","
           << std::setprecision(4) << br.qLoss << ","
           << std::setprecision(4) << br.mvaFrom << ","
           << std::setprecision(4) << br.mvaTo << ","
           << std::setprecision(4) << br.rateA << ","
           << std::setprecision(4) << br.rateSelected << ","
           << std::setprecision(2) << br.loadingPercent << "\n";
      }
      for (size_t gi = 0; gi < r.generators.size(); gi++) {
        const gridpack::utility::GeneratorResult& g = r.generators[gi];
        localGen << ct.name << ","
           << g.busId << "," << g.genId << ","
           << std::setprecision(4) << g.pGen << ","
           << std::setprecision(4) << g.qGen << ","
           << std::setprecision(4) << g.qMax << ","
           << std::setprecision(4) << g.qMin << ","
           << std::setprecision(6) << g.voltageSetpoint << ","
           << g.status << "\n";
      }
    }

    // Gather all CSV fragments on rank 0 using point-to-point send/recv
    MPI_Comm mpi_comm = static_cast<MPI_Comm>(world);
    std::vector<std::string> allBus(world.size()), allBranch(world.size());
    std::vector<std::string> allGen(world.size());
    allBus[0] = localBus.str();
    allBranch[0] = localBranch.str();
    allGen[0] = localGen.str();
    if (world.rank() == 0) {
      for (int p = 1; p < world.size(); p++) {
        int lens[3];
        MPI_Recv(lens, 3, MPI_INT, p, 0, mpi_comm, MPI_STATUS_IGNORE);
        allBus[p].resize(lens[0]);
        allBranch[p].resize(lens[1]);
        allGen[p].resize(lens[2]);
        if (lens[0] > 0)
          MPI_Recv(&allBus[p][0], lens[0], MPI_CHAR, p, 1, mpi_comm,
                   MPI_STATUS_IGNORE);
        if (lens[1] > 0)
          MPI_Recv(&allBranch[p][0], lens[1], MPI_CHAR, p, 2, mpi_comm,
                   MPI_STATUS_IGNORE);
        if (lens[2] > 0)
          MPI_Recv(&allGen[p][0], lens[2], MPI_CHAR, p, 3, mpi_comm,
                   MPI_STATUS_IGNORE);
      }
    } else {
      std::string sBus = localBus.str(), sBranch = localBranch.str();
      std::string sGen = localGen.str();
      int lens[3] = {(int)sBus.size(), (int)sBranch.size(), (int)sGen.size()};
      MPI_Send(lens, 3, MPI_INT, 0, 0, mpi_comm);
      if (lens[0] > 0)
        MPI_Send(const_cast<char*>(sBus.c_str()), lens[0], MPI_CHAR, 0, 1,
                 mpi_comm);
      if (lens[1] > 0)
        MPI_Send(const_cast<char*>(sBranch.c_str()), lens[1], MPI_CHAR, 0, 2,
                 mpi_comm);
      if (lens[2] > 0)
        MPI_Send(const_cast<char*>(sGen.c_str()), lens[2], MPI_CHAR, 0, 3,
                 mpi_comm);
    }

    // Rank 0 writes the CSV files
    if (world.rank() == 0) {
      // Write base case first (creates files with headers)
      gridpack::utility::ResultsExporter::writePFCSV(outputFile,
          baseCaseResults, "base_case");
      // Append contingency data from all processes
      {
        std::ofstream out((outputFile + "_buses.csv").c_str(), std::ios::app);
        for (size_t p = 0; p < allBus.size(); p++) out << allBus[p];
      }
      {
        std::ofstream out((outputFile + "_branches.csv").c_str(), std::ios::app);
        for (size_t p = 0; p < allBranch.size(); p++) out << allBranch[p];
      }
      {
        std::ofstream out((outputFile + "_generators.csv").c_str(), std::ios::app);
        for (size_t p = 0; p < allGen.size(); p++) out << allGen[p];
      }
    }
  }

  // Universal convergence sidecar: gather, sort by event_idx, write.
  if (emitConv) {
    auto formatRow = [](std::ostringstream &os, const ConvRow &r) {
      // Derive converged from status so ISLANDED/NO_SLACK cases -- where solve()
      // was never entered and pf_app.getConvergence() returns the previous
      // case's stale value -- read as false, matching _summary.json's
      // diverged=total-converged accounting.
      os << r.event_idx << ","
         << r.name << ","
         << r.type << ","
         << ((r.status == "OK") ? "true" : "false") << ","
         << r.cs.iterations << ","
         << std::scientific << r.cs.finalTolerance << ","
         << std::fixed
         << r.cs.finalMismatch.maxPBus << ","
         << std::setprecision(4) << r.cs.finalMismatch.maxPMismatch << ","
         << r.cs.finalMismatch.maxQBus << ","
         << std::setprecision(4) << r.cs.finalMismatch.maxQMismatch << ","
         << r.status << "\n";
    };

    std::vector<int> idx;
    std::ostringstream localStream;
    localStream << std::fixed;
    std::vector<int> localOffsets;
    localOffsets.reserve(localConvRows.size() + 1);
    for (size_t i = 0; i < localConvRows.size(); i++) {
      localOffsets.push_back(static_cast<int>(localStream.tellp()));
      formatRow(localStream, localConvRows[i]);
      idx.push_back(localConvRows[i].event_idx);
    }
    localOffsets.push_back(static_cast<int>(localStream.tellp()));
    std::string localStr = localStream.str();

    MPI_Comm conv_comm = static_cast<MPI_Comm>(world);
    if (world.rank() == 0) {
      std::vector<std::pair<int, std::string> > all;
      for (size_t i = 0; i < idx.size(); i++) {
        std::string row = localStr.substr(localOffsets[i],
                                          localOffsets[i+1] - localOffsets[i]);
        all.push_back(std::make_pair(idx[i], row));
      }
      for (int p = 1; p < world.size(); p++) {
        int n = 0;
        MPI_Recv(&n, 1, MPI_INT, p, 10, conv_comm, MPI_STATUS_IGNORE);
        if (n <= 0) continue;
        std::vector<int> remIdx(n), remOff(n + 1);
        MPI_Recv(&remIdx[0], n, MPI_INT, p, 11, conv_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&remOff[0], n + 1, MPI_INT, p, 12, conv_comm,
                 MPI_STATUS_IGNORE);
        int total = remOff[n];
        std::string buf(total, '\0');
        if (total > 0) {
          MPI_Recv(&buf[0], total, MPI_CHAR, p, 13, conv_comm,
                   MPI_STATUS_IGNORE);
        }
        for (int i = 0; i < n; i++) {
          all.push_back(std::make_pair(
              remIdx[i],
              buf.substr(remOff[i], remOff[i+1] - remOff[i])));
        }
      }
      std::sort(all.begin(), all.end());
      // With the GPU batch path, check that every case has exactly one
      // outcome (guide 6.5); a missing one is a defect and is reported as
      // MISSING rather than silently left out.
      if (gpuPath.active()) {
        std::vector<int> indices;
        for (size_t i = 0; i < all.size(); i++) {
          indices.push_back(all[i].first);
        }
        const auto coverage = gridpack::batchpf::checkOutcomeCoverage(
            static_cast<int>(events.size()), indices);
        for (int event : coverage.missing) {
          const size_t ei = static_cast<size_t>(event - 1);
          std::ostringstream row;
          std::string nm = events[ei].p_name;
          while (!nm.empty() && nm[nm.size()-1] == ' ') nm.resize(nm.size()-1);
          row << ei + 1 << "," << nm << ","
              << contingencyTypeName(events[ei])
              << ",false,0,0.000000e+00,0,0.0000,0,0.0000,MISSING\n";
          all.push_back(std::make_pair(static_cast<int>(ei) + 1, row.str()));
        }
        if (!coverage.complete()) {
          std::sort(all.begin(), all.end());
          std::cout << "WARNING: study incomplete: " << coverage.missing.size()
                    << " missing outcomes (status MISSING), " << coverage.duplicates.size()
                    << " repeated cases, " << coverage.unexpected.size()
                    << " unexpected indices\n";
        }
      }
      std::string convFile = outputFile + "_convergence.csv";
      std::ofstream cout(convFile.c_str(),
                         std::ios::out | std::ios::trunc);
      cout << "event_idx,contingency,type,converged,iterations,"
              "final_tolerance,max_p_bus,max_p_mismatch,max_q_bus,"
              "max_q_mismatch,status_code\n";
      for (size_t i = 0; i < all.size(); i++) cout << all[i].second;
      cout.close();
      printf("[convergence] wrote %zu rows to %s\n",
             all.size(), convFile.c_str());
    } else {
      int n = static_cast<int>(idx.size());
      MPI_Send(&n, 1, MPI_INT, 0, 10, conv_comm);
      if (n > 0) {
        MPI_Send(&idx[0], n, MPI_INT, 0, 11, conv_comm);
        MPI_Send(&localOffsets[0], n + 1, MPI_INT, 0, 12, conv_comm);
        int total = localOffsets[n];
        if (total > 0) {
          MPI_Send(const_cast<char*>(localStr.c_str()), total, MPI_CHAR, 0, 13,
                   conv_comm);
        }
      }
    }
  }

  timer->stop(t_merge);
  // Print out statistics on contingencies
  if (write_stats) {
    int t_stats = timer->createCategory("Write Statistics");
    timer->start(t_stats);
    if (vmag_stats) {
      vmag_stats->writeMeanAndRMS("vmag.txt",1,false);
      vmag_stats->writeMinAndMax("vmag_mm.txt",1,false);
      if (check_Qlim) vmag_stats->writeMaskValueCount("pq_change_cnt.txt",2,false);
    }
    if (vang_stats) {
      vang_stats->writeMeanAndRMS("vang.txt",1,false);
      vang_stats->writeMinAndMax("vang_mm.txt",1,false);
    }
    if (pgen_stats) {
      pgen_stats->writeMeanAndRMS("pgen.txt",1);
      pgen_stats->writeMinAndMax("pgen_mm.txt",1);
      qgen_stats->writeMeanAndRMS("qgen.txt",1);
      qgen_stats->writeMinAndMax("qgen_mm.txt",1);
    }
    if (pflow_stats) {
      pflow_stats->writeMeanAndRMS("pflow.txt",1);
      pflow_stats->writeMinAndMax("pflow_mm.txt",1);
      pflow_stats->writeMaskValueCount("line_flt_cnt.txt",2);
      qflow_stats->writeMeanAndRMS("qflow.txt",1);
      qflow_stats->writeMinAndMax("qflow_mm.txt",1);
      perf_stats->writeMinAndMax("perf_mm.txt",1);
      perf_stats->sumColumnValues("perf_sum.txt",1);
    }
    if (world.rank() == 0 && (!vmag_stats || !pgen_stats || !pflow_stats)) {
      printf("Note: StatBlock files skipped for element types with no "
             "monitored rows\n");
    }
    timer->stop(t_stats);
  }
  timer->stop(t_total);
  // If all processors executed at least one task, then print out timing
  // statistics (this printout does not work if some processors do not define
  // all timing variables)
  if (events.size()*grp_size >= world.size()) {
    timer->dump();
  }
}
