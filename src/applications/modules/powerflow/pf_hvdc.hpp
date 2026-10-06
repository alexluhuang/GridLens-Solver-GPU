/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   pf_hvdc.hpp
 *
 * @brief  Steady-state model of line-commutated two-terminal dc lines for
 * the sequential ac/dc power flow. Each converter is a six-pulse bridge
 * group with commutating reactance and resistance; the rectifier controls
 * the dc current (power or current order) and the inverter holds the
 * compounded dc voltage, with the standard mode changes when a converter
 * angle reaches its limit with the converter transformer tap at its limit.
 */
// -------------------------------------------------------------

#ifndef _pf_hvdc_h_
#define _pf_hvdc_h_

#include <string>

namespace gridpack {
namespace powerflow {

/**
 * Converter data for one end of a two-terminal dc line (PSS/E fields IP, NB,
 * ANMX, ANMN, RC, XC, EBAS, TR, TAP, TMX, TMN, STP)
 */
struct HVDCConverter
{
  int bus;          // ac bus number
  double nb;        // number of bridges in series
  double anmx;      // maximum firing (rectifier) or extinction (inverter) angle, degrees
  double anmn;      // minimum angle, degrees
  double rc;        // commutating resistance per bridge, ohms
  double xc;        // commutating reactance per bridge, ohms
  double ebas;      // primary base ac voltage, kV
  double tr;        // transformer ratio
  double tap;       // tap setting
  double tmx;       // maximum tap
  double tmn;       // minimum tap
  double stp;       // tap step
};

/**
 * Two-terminal dc line (PSS/E fields NAME, MDC, RDC, SETVL, VSCHD, VCMOD,
 * RCOMP, DELTI)
 */
struct HVDCLine
{
  std::string name;
  int mdc;          // control mode: 0 blocked, 1 power, 2 current
  double rdc;       // dc line resistance, ohms
  double setvl;     // power (MW, >0 at rectifier, <0 at inverter) or current (amps)
  double vschd;     // scheduled compounded dc voltage, kV
  double vcmod;     // mode switch dc voltage, kV
  double rcomp;     // compounding resistance, ohms
  double delti;     // current margin, fraction of the order
  HVDCConverter rect;
  HVDCConverter inv;
};

/**
 * Control mode of a dc line at its operating point
 */
enum HVDCMode
{
  HVDC_BLOCKED,        // not operating
  HVDC_NORMAL,         // rectifier holds the order, inverter holds the dc voltage
  HVDC_INV_GAMMA_MIN,  // inverter at minimum extinction angle, dc voltage below schedule
  HVDC_CURRENT_MODE,   // power order switched to current order (dc voltage below VCMOD)
  HVDC_RECT_ALPHA_MIN  // rectifier at minimum firing angle, inverter controls current with margin
};

/**
 * Operating point of one converter
 */
struct HVDCConverterState
{
  double angle;     // firing (rectifier) or extinction (inverter) angle, degrees
  double tap;       // converter transformer tap
  double p;         // ac active power in MW, drawn by the rectifier, sent by the inverter
  double q;         // ac reactive power absorbed, MVar
  bool limited;     // angle is held at a limit
};

/**
 * Operating point of a dc line
 */
struct HVDCSolution
{
  HVDCMode mode;
  double id;        // dc current, kA
  double vdcr;      // dc voltage at the rectifier, kV
  double vdci;      // dc voltage at the inverter, kV
  HVDCConverterState rect;
  HVDCConverterState inv;
  bool limited;     // some converter angle could not be kept inside its limits
};

/**
 * Solve a dc line at the given ac voltage magnitudes of its converter buses
 * @param line dc line data
 * @param vr ac voltage magnitude at the rectifier bus (pu)
 * @param vi ac voltage magnitude at the inverter bus (pu)
 * @return operating point of the line
 */
HVDCSolution solveTwoTerminalDC(const HVDCLine &line, double vr, double vi);

/**
 * Operating point of a line that is not operating
 */
HVDCSolution blockedHVDCSolution();

/**
 * Text label for a control mode
 * @param mode control mode
 * @return label
 */
const char* hvdcModeName(HVDCMode mode);

/**
 * Normalize a dc line name for matching: strip quotes and surrounding
 * white space and collapse internal runs of white space to one blank
 * @param name dc line name
 * @return normalized name
 */
std::string normalizeHVDCName(const std::string &name);

}  // powerflow
}  // gridpack
#endif
