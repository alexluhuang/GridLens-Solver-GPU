/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 *
 *
 * base_block_parser.cpp
 *       Created on: November 29, 2022
 *           Author: Bruce Palmer
 */
#include "gridpack/parser/block_parsers/base_block_parser.hpp"

/**
 * Constructor
 * @param bus_map map indices in RAW file to internal indices
 * @param name_map map name in RAW file to internal indices
 * @param branch_map map bus index pair in RAW file to internal indices
 */
gridpack::parser::BaseBlockParser::BaseBlockParser(
    std::map<int,int> *bus_map,
    std::map<std::string,int> *name_map,
    std::map<std::pair<int, int>, int> *branch_map)
{
  p_busMap = bus_map;
  p_nameMap = name_map;
  p_branchMap = branch_map;
}

/**
 * Simple destructor
 */
gridpack::parser::BaseBlockParser::~BaseBlockParser()
{
}

/**
 * Test to see if string is a comment line. Check to see if first
 * non-blank characters are "//"
 * @param str input string
 */
bool gridpack::parser::BaseBlockParser::check_comment(std::string &str) const
{
  int ntok = str.find_first_not_of(' ',0);
  if (ntok != std::string::npos && ntok+1 != std::string::npos &&
      str[ntok] == '/' && str[ntok+1] == '/') {
    return true;
  } else if (ntok != std::string::npos && ntok+1 != std::string::npos &&
      str[ntok] == '@' && str[ntok+1] == '!') {
    return true;
  }
  return false;
}

/**
 * Remove leading and trailing blanks
 * @param str input string
 */
void gridpack::parser::BaseBlockParser::remove_blanks(std::string &str)
{
  int ntok1 = str.find_first_not_of(' ',0);
  int ntok2 = str.length()-1;
  while (str[ntok2] == ' ') {ntok2--;}
  str = str.substr(ntok1, ntok2-ntok1+1);
}

/**
 * Test to see if string terminates a section
 * @param str input string
 * @return: false if first non-blank character is TERM_CHAR
 */
bool gridpack::parser::BaseBlockParser::test_end(std::string &str) const
{
  if (str[0] == TERM_CHAR) {
    return false;
  }
  int len = str.length();
  int i=0;
  while (i<len && str[i] == ' ') {
    i++;
  }
  if (i<len && str[i] != TERM_CHAR) {
    return true;
  } else if (i == len) {
    return true;
  } else if (str[i] == TERM_CHAR) {
    i++;
    if (i>=len || str[i] == ' ' || str[i] == '\\') {
      return false;
    } else { 
      return true;
    }
  } else { 
    return true;
  }
}

/**
 * Check to see if string represents a character string. Look for single
 * quotes. Strip off white space and remove quotes if returning true
 * @param str input string
 * @return true if string is a character string delimited by single quotes
 */
bool gridpack::parser::BaseBlockParser::check_string(std::string &str)
{
  int ntok1 = str.find_first_not_of(' ',0);
  if (ntok1 != std::string::npos && str[ntok1] == '\'') {
    ntok1++;
    int ntok2 = str.find_first_of('\'',ntok1);
    if (ntok2-ntok1 > 1 && ntok2 != std::string::npos) {
      ntok2--;
      str = str.substr(ntok1, ntok2-ntok1+1);
    } else if (ntok2 == std::string::npos && str.length()-1 - ntok1 > 1) {
      ntok2 = str.length()-1;
      str = str.substr(ntok1, ntok2-ntok1+1);
    } else {
      str.clear();
    }
    // strip off any remaining trailing blanks
    if (str.length() > 0) remove_blanks(str);
    return true;
  } else {
    // string may not have single quotes but remove leading and trailing
    // blanks, if any
    remove_blanks(str);
    return false;
  }
}

/**
 * Split bus name into its name and base bus voltage. Assume that incoming
 * string has been stripped of single quotes and leading and trailing
 * white space. The name and the voltage are suppose to be delimited by at
 * least 3 white spaces
 * @param string input string
 * @param name bus name derived from string
 * @param voltage base bus voltage derived from string
 */
void gridpack::parser::BaseBlockParser::parseBusName(
    std::string &string, std::string &name, double &voltage)
{
  // If string contains 12 or less characters, assume it is just a name
  name = string;
  voltage = 0.0;
  int len = string.length();
  if (len > 12) {
    // look for 3 consecutive blank characters
    int ntok1 = string.find("   ",0);
    if (ntok1 != std::string::npos) {
      name = string.substr(0,ntok1);
      int ntok2 = string.find_first_not_of(' ',ntok1);
      if (ntok2 != std::string::npos) {
        voltage = atof(string.substr(ntok2,len-ntok2).c_str());
      }
    }
  }
}

/**
 * Remove comment from string (all text after a single '/' character).
 * A '/' inside any quoted field (delimited by either '' or "") is part
 * of the field, not a comment
 * @param string line of text to be cleaned
 */
void gridpack::parser::BaseBlockParser::cleanComment(std::string &string)
{
  // Track quotes along the whole line: a record can have several quoted
  // fields (e.g. a transformer circuit ID followed by its name) and a '/'
  // in any of them is not a comment
  char quote = 0;
  for (size_t i = 0; i < string.size(); i++) {
    char c = string[i];
    if (quote) {
      if (c == quote) quote = 0;
    } else if (c == '\'' || c == '\"') {
      quote = c;
    } else if (c == '/') {
      string.erase(i);
      return;
    }
  }
}

/**
 * Split PSS/E formatted lines into individual tokens using both blanks and
 * commas as delimiters
 * @param line input string from PSS/E file
 * @return vector of tokens parsed from PSS/E line
 */
std::vector<std::string> gridpack::parser::BaseBlockParser::splitPSSELine (
    std::string line)
{
  std::vector<std::string> ret;
  int i, j;
  std::vector<std::string>  split_line;
  // split line into tokens based on comma delimiters
  boost::algorithm::split(split_line, line, boost::algorithm::is_any_of(","),
      boost::token_compress_off);
  int slen = split_line.size();
  gridpack::utility::StringUtils p_util;
  // parse each token based on blank spaces
  for (i=0; i<slen; i++) {
    if (isBlank(split_line[i])) {
      // If two consecutive commas have nothing in between, replace it with
      // a zero "0" character. This should be converted to 0 and 0.0 by the
      // atoi and atof functions, respectively
      ret.push_back("0");
    } else {
      std::vector<std::string> tokens;
      tokens = p_util.blankTokenizer(split_line[i]);
      int tlen = tokens.size();
      for (j=0; j<tlen; j++) ret.push_back(tokens[j]);
    }
  }
  return ret;
}

/**
 * Check to see if string is blank
 * @param string string that needs to checked for non-blank characters
 * @return true if no non-blank characters are found
 */
bool gridpack::parser::BaseBlockParser::isBlank(std::string string)
{
  int idx = string.find_first_not_of(' ',0);
  if (idx != std::string::npos) return false;
  return true;
}

/**
 * Get bus index from bus name string. If the bus name string does not
 * have quotes, assume it represents an integer index. If it does have
 * quotes, find the corresponding index in the p_nameMap data structure
 */
int gridpack::parser::BaseBlockParser::getBusIndex(std::string str)
{
  if (check_string(str)) {
    std::string name;
    double voltage;
    parseBusName(str,name,voltage);
    std::map<std::string,int>::iterator it;
    it = p_nameMap->find(name);
    if (it != p_nameMap->end()) {
      return it->second;
    } else {
      return -1;
    }
  } else {
    return abs(atoi(str.c_str()));
  }
}

/**
 * Off-nominal turns ratio of a transformer winding in per unit of the base
 * voltage of the winding's bus, from PSS/E winding data
 * @param cw winding data I/O code: 1 ratio in pu of bus base voltage,
 *        2 winding voltage in kV, 3 ratio in pu of nominal winding voltage
 * @param windv winding value (WINDV, RMA or RMI) from the RAW file
 * @param nomv nominal winding voltage in kV; 0 means the bus base voltage
 * @param basekv base voltage of the winding's bus in kV
 * @return ratio in pu of the bus base voltage
 */
double gridpack::parser::BaseBlockParser::windingRatio(int cw, double windv,
    double nomv, double basekv) const
{
  // Without a bus base voltage fall back on the nominal winding voltage
  double base = (basekv > 0.0) ? basekv : nomv;
  if (base <= 0.0) return windv;
  if (cw == 2) return windv/base;
  if (cw == 3) return windv*((nomv > 0.0) ? nomv : base)/base;
  return windv;
}

/**
 * Base voltage of a bus
 * @param o_idx original bus index
 * @param busData data collection objects for buses
 * @return base voltage in kV, 0 if the bus or its base voltage is unknown
 */
double gridpack::parser::BaseBlockParser::busBaseKV(int o_idx,
    std::vector<boost::shared_ptr<gridpack::component::DataCollection> > &busData)
{
  double kv = 0.0;
  std::map<int,int>::iterator it = p_busMap->find(o_idx);
  if (it != p_busMap->end() && it->second < static_cast<int>(busData.size())) {
    busData[it->second]->getValue(BUS_BASEKV, &kv);
  }
  return kv;
}
