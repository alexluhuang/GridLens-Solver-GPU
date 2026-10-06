/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 *
 *
 * two_term_parser33.hpp
 *       Created on: November 29, 2022
 *           Author: Bruce Palmer
 */
#ifndef _TWO_TERM_PARSER33_H
#define _TWO_TERM_PARSER33_H

#include "gridpack/parser/block_parsers/base_block_parser.hpp"

namespace gridpack {
namespace parser {

class TwoTermParser33 : public BaseBlockParser {
  public:
  /**
   * Constructor
   * @param bus_map map indices in RAW file to internal indices
   * @param name_map map name in RAW file to internal indices
   * @param branch_map map bus index pair in RAW file to internal indices
   */
  TwoTermParser33(
      std::map<int,int> *bus_map,
      std::map<std::string,int> *name_map,
      std::map<std::pair<int, int>, int> *branch_map);

  /**
   * Simple Destructor
   */
  virtual ~TwoTermParser33(void);

  /**
   * parse two terminal dc block. Each record (dc line data, rectifier and
   * inverter lines) is stored in the network data collection, indexed by dc
   * line. Lines whose converter buses are not in the network are skipped
   * @param stream input stream that feeds lines from RAW file
   * @param p_network_data data collection object for network-level data
   */
  void parse(
      gridpack::stream::InputStream &stream,
      boost::shared_ptr<gridpack::component::DataCollection> &p_network_data);
};

} // parser
} // gridpack
#endif
