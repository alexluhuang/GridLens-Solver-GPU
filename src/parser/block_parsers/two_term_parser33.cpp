/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 *
 *
 * two_term_parser33.cpp
 *       Created on: November 29, 2022
 *           Author: Bruce Palmer
 */
#include "two_term_parser33.hpp"
#include <cstdio>
#include <cstdlib>

/**
 * Constructor
 * @param bus_map map indices in RAW file to internal indices
 * @param name_map map name in RAW file to internal indices
 * @param branch_map map bus index pair in RAW file to internal indices
 */
gridpack::parser::TwoTermParser33::TwoTermParser33(
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
gridpack::parser::TwoTermParser33::~TwoTermParser33(void)
{
}

/**
 * parse two terminal dc block. Each record (dc line data, rectifier and
 * inverter lines) is stored in the network data collection, indexed by dc
 * line. Lines whose converter buses are not in the network are skipped
 * @param stream input stream that feeds lines from RAW file
 * @param p_network_data data collection object for network-level data
 */
void gridpack::parser::TwoTermParser33::parse(
    gridpack::stream::InputStream &stream,
    boost::shared_ptr<gridpack::component::DataCollection> &p_network_data)
{
  std::string          line;
  gridpack::utility::StringUtils util;
  int ncnt = 0;
  int nactive = 0;

  stream.nextLine(line); //this should be the first line of the block

  while(test_end(line)) {
    if (check_comment(line)) {
      stream.nextLine(line);
      continue;
    }
    // Each record has three lines: dc line data, rectifier, inverter
    std::vector<std::string> rec[3];
    bool complete = true;
    for (int k=0; k<3; k++) {
      if (k > 0) {
        stream.nextLine(line);
        while (test_end(line) && check_comment(line)) stream.nextLine(line);
        if (!test_end(line)) {
          complete = false;
          break;
        }
      }
      this->cleanComment(line);
      rec[k] = this->splitPSSELine(line);
    }
    if (!complete) {
      printf("Two-terminal dc data ends inside a record; record ignored\n");
      break;
    }
    stream.nextLine(line);

    std::vector<std::string> &hdr = rec[0];
    std::vector<std::string> &rct = rec[1];
    std::vector<std::string> &inv = rec[2];
    if (hdr.size() < 8 || rct.size() < 12 || inv.size() < 12) {
      printf("Two-terminal dc record has too few fields; record ignored\n");
      continue;
    }
    std::string name = util.trimQuotes(hdr[0]);
    int ipr = atoi(rct[0].c_str());
    int ipi = atoi(inv[0].c_str());
    if (p_busMap->find(ipr) == p_busMap->end() ||
        p_busMap->find(ipi) == p_busMap->end()) {
      printf("Two-terminal dc line %s: converter bus %d or %d not found;"
          " line ignored\n", name.c_str(), ipr, ipi);
      continue;
    }
    int mdc = atoi(hdr[1].c_str());
    if (mdc != 0) nactive++;

    // Line data: 'NAME', MDC, RDC, SETVL, VSCHD, VCMOD, RCOMP, DELTI, ...
    p_network_data->addValue(HVDC_LINE_NAME, name.c_str(), ncnt);
    p_network_data->addValue(HVDC_LINE_MDC, mdc, ncnt);
    p_network_data->addValue(HVDC_LINE_RDC, atof(hdr[2].c_str()), ncnt);
    p_network_data->addValue(HVDC_LINE_SETVL, atof(hdr[3].c_str()), ncnt);
    p_network_data->addValue(HVDC_LINE_VSCHD, atof(hdr[4].c_str()), ncnt);
    p_network_data->addValue(HVDC_LINE_VCMOD, atof(hdr[5].c_str()), ncnt);
    p_network_data->addValue(HVDC_LINE_RCOMP, atof(hdr[6].c_str()), ncnt);
    p_network_data->addValue(HVDC_LINE_DELTI, atof(hdr[7].c_str()), ncnt);

    // Converter data: IP, NB, ANMX, ANMN, RC, XC, EBAS, TR, TAP, TMX, TMN,
    // STP. The fields that follow STP differ between PSS/E versions and
    // are not used
    p_network_data->addValue(HVDC_RECT_BUS, ipr, ncnt);
    p_network_data->addValue(HVDC_RECT_NB, atof(rct[1].c_str()), ncnt);
    p_network_data->addValue(HVDC_RECT_ANMX, atof(rct[2].c_str()), ncnt);
    p_network_data->addValue(HVDC_RECT_ANMN, atof(rct[3].c_str()), ncnt);
    p_network_data->addValue(HVDC_RECT_RC, atof(rct[4].c_str()), ncnt);
    p_network_data->addValue(HVDC_RECT_XC, atof(rct[5].c_str()), ncnt);
    p_network_data->addValue(HVDC_RECT_EBAS, atof(rct[6].c_str()), ncnt);
    p_network_data->addValue(HVDC_RECT_TR, atof(rct[7].c_str()), ncnt);
    p_network_data->addValue(HVDC_RECT_TAP, atof(rct[8].c_str()), ncnt);
    p_network_data->addValue(HVDC_RECT_TMX, atof(rct[9].c_str()), ncnt);
    p_network_data->addValue(HVDC_RECT_TMN, atof(rct[10].c_str()), ncnt);
    p_network_data->addValue(HVDC_RECT_STP, atof(rct[11].c_str()), ncnt);
    p_network_data->addValue(HVDC_INV_BUS, ipi, ncnt);
    p_network_data->addValue(HVDC_INV_NB, atof(inv[1].c_str()), ncnt);
    p_network_data->addValue(HVDC_INV_ANMX, atof(inv[2].c_str()), ncnt);
    p_network_data->addValue(HVDC_INV_ANMN, atof(inv[3].c_str()), ncnt);
    p_network_data->addValue(HVDC_INV_RC, atof(inv[4].c_str()), ncnt);
    p_network_data->addValue(HVDC_INV_XC, atof(inv[5].c_str()), ncnt);
    p_network_data->addValue(HVDC_INV_EBAS, atof(inv[6].c_str()), ncnt);
    p_network_data->addValue(HVDC_INV_TR, atof(inv[7].c_str()), ncnt);
    p_network_data->addValue(HVDC_INV_TAP, atof(inv[8].c_str()), ncnt);
    p_network_data->addValue(HVDC_INV_TMX, atof(inv[9].c_str()), ncnt);
    p_network_data->addValue(HVDC_INV_TMN, atof(inv[10].c_str()), ncnt);
    p_network_data->addValue(HVDC_INV_STP, atof(inv[11].c_str()), ncnt);
    if (rct.size() > 12 && atoi(rct[12].c_str()) != 0) {
      printf("Two-terminal dc line %s: firing angle measuring bus ICR is not"
          " modeled\n", name.c_str());
    }
    ncnt++;
  }
  p_network_data->addValue(HVDC_LINE_TOTAL, ncnt);
  if (ncnt > 0) {
    printf("Two-terminal dc: %d lines read, %d in service\n", ncnt, nactive);
  }
}
