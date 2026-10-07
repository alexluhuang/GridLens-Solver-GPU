/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   pf_hvdc.cpp
 *
 * @brief  Host helpers of the two-terminal dc line model. The numeric model
 * is inline in pf_hvdc.hpp, shared with the GPU batch contingency path
 */
// -------------------------------------------------------------

#include "pf_hvdc.hpp"

/**
 * Text label for a control mode
 * @param mode control mode
 * @return label
 */
const char* gridpack::powerflow::hvdcModeName(HVDCMode mode)
{
  switch (mode) {
    case HVDC_NORMAL:
      return "normal";
    case HVDC_INV_GAMMA_MIN:
      return "inverter at gamma min";
    case HVDC_CURRENT_MODE:
      return "current mode (dc voltage below VCMOD)";
    case HVDC_RECT_ALPHA_MIN:
      return "rectifier at alpha min, inverter current control";
    default:
      return "blocked";
  }
}

/**
 * Normalize a dc line name for matching: strip quotes and surrounding
 * white space and collapse internal runs of white space to one blank
 * @param name dc line name
 * @return normalized name
 */
std::string gridpack::powerflow::normalizeHVDCName(const std::string &name)
{
  std::string ret;
  bool blank = false;
  for (size_t i=0; i<name.size(); i++) {
    char c = name[i];
    if (c == '\'' || c == '\"') continue;
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      blank = !ret.empty();
      continue;
    }
    if (blank) ret += ' ';
    blank = false;
    ret += c;
  }
  return ret;
}
