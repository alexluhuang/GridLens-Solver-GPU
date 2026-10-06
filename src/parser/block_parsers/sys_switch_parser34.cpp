/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 *
 *
 * sys_switch_parser34.cpp
 *       Created on: December December 5, 2022
 *           Author: Bruce Palmer
 */
#include "sys_switch_parser34.hpp"

/**
 * Constructor
 * @param bus_map map indices in RAW file to internal indices
 * @param name_map map name in RAW file to internal indices
 * @param branch_map map bus index pair in RAW file to internal indices
 */
gridpack::parser::SysSwitchParser34::SysSwitchParser34(
    std::map<int,int> *bus_map,
    std::map<std::string,int> *name_map,
    std::map<std::pair<int, int>, int> *branch_map) :
    gridpack::parser::BaseBlockParser(
      bus_map, name_map, branch_map)
{
}


/**
 * Simple Destructor
 */
gridpack::parser::SysSwitchParser34::~SysSwitchParser34(void)
{
}

/**
 * parse system switching device block. Each device (breaker or switch) is
 * added as a branch element between its two buses: zero resistance, the
 * device reactance, its ratings and its status. Devices between the same
 * pair of buses as an existing branch become parallel elements of it
 * @param stream input stream that feeds lines from RAW file
 * @param p_branchData vector of data collection objects for branches
 */
void gridpack::parser::SysSwitchParser34::parse(
    gridpack::stream::InputStream &stream,
    std::vector<boost::shared_ptr<gridpack::component::DataCollection> > &p_branchData)
{
  std::string          line;
  gridpack::utility::StringUtils util;
  int ndevice = 0;
  int nclosed = 0;

  stream.nextLine(line); //this should be the first line of the block

  while(test_end(line)) {
    if (check_comment(line)) {
      stream.nextLine(line);
      continue;
    }
    this->cleanComment(line);
    std::vector<std::string> split_line = this->splitPSSELine(line);
    // I, J, 'CKT', X, RATE1-12, STAT, ...
    int nstr = split_line.size();
    if (nstr < 4) {
      stream.nextLine(line);
      continue;
    }
    int o_idx1 = getBusIndex(split_line[0]);
    int o_idx2 = getBusIndex(split_line[1]);
    if (p_busMap->find(o_idx1) == p_busMap->end() ||
        p_busMap->find(o_idx2) == p_busMap->end()) {
      stream.nextLine(line);
      continue;
    }

    // Find the branch for this pair of buses or create one
    int l_idx = 0;
    int nelems = 0;
    bool switched = false;
    std::pair<int, int> branch_pair(o_idx1, o_idx2);
    std::map<std::pair<int, int>, int>::iterator it;
    it = p_branchMap->find(branch_pair);
    if (it != p_branchMap->end()) {
      l_idx = it->second;
      p_branchData[l_idx]->getValue(BRANCH_NUM_ELEMENTS,&nelems);
    } else {
      it = p_branchMap->find(std::pair<int,int>(o_idx2, o_idx1));
      if (it != p_branchMap->end()) {
        l_idx = it->second;
        p_branchData[l_idx]->getValue(BRANCH_NUM_ELEMENTS,&nelems);
        switched = true;
      } else {
        boost::shared_ptr<gridpack::component::DataCollection>
          data(new gridpack::component::DataCollection);
        l_idx = p_branchData.size();
        p_branchData.push_back(data);
        p_branchMap->insert(std::pair<std::pair<int,int>,int>(branch_pair,
              l_idx));
        p_branchData[l_idx]->addValue(BRANCH_NUM_ELEMENTS,nelems);
        p_branchData[l_idx]->addValue(BRANCH_INDEX,l_idx);
        p_branchData[l_idx]->addValue(BRANCH_FROMBUS,o_idx1);
        p_branchData[l_idx]->addValue(BRANCH_TOBUS,o_idx2);
      }
    }

    p_branchData[l_idx]->addValue(BRANCH_SWITCHED, switched, nelems);
    std::string tag = util.clean2Char(split_line[2]);
    p_branchData[l_idx]->addValue(BRANCH_CKT, tag.c_str(), nelems);
    // A switching device is a near-ideal connection; PSS/E uses a
    // reactance of 0.0001 pu when none is given
    double x = atof(split_line[3].c_str());
    if (x == 0.0) x = 1.0e-4;
    p_branchData[l_idx]->addValue(BRANCH_R, 0.0, nelems);
    p_branchData[l_idx]->addValue(BRANCH_X, x, nelems);
    p_branchData[l_idx]->addValue(BRANCH_B, 0.0, nelems);
    const char *rates[12] = {BRANCH_RATE1, BRANCH_RATE2, BRANCH_RATE3,
      BRANCH_RATE4, BRANCH_RATE5, BRANCH_RATE6, BRANCH_RATE7, BRANCH_RATE8,
      BRANCH_RATE9, BRANCH_RATE10, BRANCH_RATE11, BRANCH_RATE12};
    for (int k = 0; k < 12; k++) {
      double rate = (nstr > 4+k) ? atof(split_line[4+k].c_str()) : 0.0;
      p_branchData[l_idx]->addValue(rates[k], rate, nelems);
      if (k == 0) p_branchData[l_idx]->addValue(BRANCH_RATING_A, rate, nelems);
      if (k == 1) p_branchData[l_idx]->addValue(BRANCH_RATING_B, rate, nelems);
      if (k == 2) p_branchData[l_idx]->addValue(BRANCH_RATING_C, rate, nelems);
    }
    p_branchData[l_idx]->addValue(BRANCH_SHUNT_ADMTTNC_G1, 0.0, nelems);
    p_branchData[l_idx]->addValue(BRANCH_SHUNT_ADMTTNC_B1, 0.0, nelems);
    p_branchData[l_idx]->addValue(BRANCH_SHUNT_ADMTTNC_G2, 0.0, nelems);
    p_branchData[l_idx]->addValue(BRANCH_SHUNT_ADMTTNC_B2, 0.0, nelems);
    int status = (nstr > 16) ? atoi(split_line[16].c_str()) : 1;
    p_branchData[l_idx]->addValue(BRANCH_STATUS, status, nelems);
    p_branchData[l_idx]->addValue(BRANCH_TAP, 0.0, nelems);
    p_branchData[l_idx]->addValue(BRANCH_SHIFT, 0.0, nelems);
    nelems++;
    p_branchData[l_idx]->setValue(BRANCH_NUM_ELEMENTS, nelems);
    ndevice++;
    if (status == 1) nclosed++;

    stream.nextLine(line);
  }
  if (ndevice > 0) {
    printf("System switching devices: %d read, %d closed\n", ndevice, nclosed);
  }
}
