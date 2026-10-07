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
 *
 * The numeric model is written once, as inline functions on plain data, so
 * that GridPACK's power flow and the GPU batch contingency path evaluate the
 * same formulas: compiled by a CUDA compiler the functions are also device
 * functions. The name helpers at the end are host code.
 */
// -------------------------------------------------------------

#ifndef _pf_hvdc_h_
#define _pf_hvdc_h_

#include <cmath>
#include <string>

// Host and device qualifiers when compiled by a CUDA compiler (exception
// EX-CG-01: kernels need CUDA's function qualifiers)
#if defined(__CUDACC__)
#define GRIDPACK_HVDC_HD __host__ __device__
#else
#define GRIDPACK_HVDC_HD
#endif

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
 * Numeric data of a two-terminal dc line (PSS/E fields MDC, RDC, SETVL,
 * VSCHD, VCMOD, RCOMP, DELTI and the two converters)
 */
struct HVDCLineData
{
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
 * Two-terminal dc line (PSS/E field NAME and the numeric data)
 */
struct HVDCLine : HVDCLineData
{
  std::string name;
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

namespace hvdc_detail {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSqrt2 = 1.414213562373095048801688724209698079;
// no-load dc voltage of a six-pulse bridge per kV of ac voltage
constexpr double kKvdo = 3.0*kSqrt2/kPi;
// commutation voltage drop per ohm of commutating reactance and kA
constexpr double kKxc = 3.0/kPi;

/**
 * Converter angle that gives a required dc voltage
 */
struct AngleFit
{
  double c;     // cosine of the angle
  double tap;   // tap used
  int side;     // 0 inside the limits, -1 held at minimum angle, +1 at maximum
};

/**
 * Bridge ac voltage of a converter
 * @param cv converter data
 * @param vac ac voltage magnitude at the converter bus (pu)
 * @param tap converter transformer tap
 * @return bridge ac voltage (kV)
 */
GRIDPACK_HVDC_HD inline double bridgeVoltage(const HVDCConverter &cv,
    double vac, double tap)
{
  return vac*cv.ebas*cv.tr/tap;
}

/**
 * Dc voltage of the bridges behind the commutating resistance
 * @param cv converter data
 * @param rect true for the rectifier end
 * @param vdc dc terminal voltage (kV)
 * @param id dc current (kA)
 * @return dc voltage behind the commutating resistance (kV)
 */
GRIDPACK_HVDC_HD inline double internalVoltage(const HVDCConverter &cv,
    bool rect, double vdc, double id)
{
  return rect ? vdc + 2.0*cv.nb*cv.rc*id : vdc - 2.0*cv.nb*cv.rc*id;
}

/**
 * Angle needed for a dc voltage at a dc current. The tap is moved within its
 * limits only if the angle would otherwise leave its range
 * @param cv converter data
 * @param rect true for the rectifier end
 * @param vac ac voltage magnitude at the converter bus (pu)
 * @param vdc required dc terminal voltage (kV)
 * @param id dc current (kA)
 * @return cosine of the angle, tap and limit indicator
 */
GRIDPACK_HVDC_HD inline AngleFit fitAngle(const HVDCConverter &cv, bool rect,
    double vac, double vdc, double id)
{
  AngleFit fit{};
  const double amin = (cv.anmn < cv.anmx) ? cv.anmn : cv.anmx;
  const double amax = (cv.anmn < cv.anmx) ? cv.anmx : cv.anmn;
  const double cmax = std::cos(amin*kPi/180.0);
  const double cmin = std::cos(amax*kPi/180.0);
  const double need = internalVoltage(cv, rect, vdc, id)/cv.nb + kKxc*cv.xc*id;
  fit.tap = cv.tap;
  fit.side = 0;
  fit.c = need/(kKvdo*bridgeVoltage(cv, vac, fit.tap));
  if (fit.c <= 0.0) {
    fit.c = cmin;
    fit.side = 1;
    return fit;
  }
  if (fit.c > cmax || fit.c < cmin) {
    // cos(angle) scales with the tap since the bridge voltage is inversely
    // proportional to it
    double tap = fit.tap*((fit.c > cmax) ? cmax : cmin)/fit.c;
    if (cv.tmx > 0.0 && tap > cv.tmx) tap = cv.tmx;
    if (cv.tmn > 0.0 && tap < cv.tmn) tap = cv.tmn;
    fit.tap = tap;
    fit.c = need/(kKvdo*bridgeVoltage(cv, vac, tap));
    const double eps = 1.0e-10;
    if (fit.c > cmax*(1.0 + eps)) {
      fit.side = -1;
    } else if (fit.c < cmin*(1.0 - eps)) {
      fit.side = 1;
    }
    if (fit.c > cmax) fit.c = cmax;
    if (fit.c < cmin) fit.c = cmin;
  }
  return fit;
}

/**
 * Dc terminal voltage of a converter held at an angle, as a linear function
 * of the dc current: vdc = a + b*id
 * @param cv converter data
 * @param rect true for the rectifier end
 * @param vac ac voltage magnitude at the converter bus (pu)
 * @param tap converter transformer tap
 * @param c cosine of the angle
 * @param a constant term (kV)
 * @param b slope (kV/kA)
 */
GRIDPACK_HVDC_HD inline void heldVoltage(const HVDCConverter &cv, bool rect,
    double vac, double tap, double c, double *a, double *b)
{
  *a = cv.nb*kKvdo*bridgeVoltage(cv, vac, tap)*c;
  *b = rect ? -cv.nb*kKxc*cv.xc - 2.0*cv.nb*cv.rc
            : -cv.nb*kKxc*cv.xc + 2.0*cv.nb*cv.rc;
}

/**
 * Smallest positive root of a*x^2 + b*x - c = 0 for c > 0
 * @param x root
 * @return false if there is no positive root
 */
GRIDPACK_HVDC_HD inline bool positiveRoot(double a, double b, double c,
    double *x)
{
  const double disc = b*b + 4.0*a*c;
  if (disc < 0.0) return false;
  const double den = b + std::sqrt(disc);
  if (den <= 0.0) return false;
  *x = 2.0*c/den;
  return *x > 0.0;
}

/**
 * Ac power of a converter at its operating point
 * @param cv converter data
 * @param rect true for the rectifier end
 * @param vac ac voltage magnitude at the converter bus (pu)
 * @param vdc dc terminal voltage (kV)
 * @param id dc current (kA)
 * @param fit converter angle and tap
 * @return converter state
 */
GRIDPACK_HVDC_HD inline HVDCConverterState converterState(
    const HVDCConverter &cv, bool rect, double vac, double vdc, double id,
    const AngleFit &fit)
{
  HVDCConverterState s{};
  const double e = bridgeVoltage(cv, vac, fit.tap);
  const double a = std::acos(fit.c);
  // overlap angle mu from cos(a) - cos(a+mu) = sqrt(2)*xc*id/e
  double c2 = fit.c - kSqrt2*cv.xc*id/e;
  if (c2 < -1.0) c2 = -1.0;
  double mu = std::acos(c2) - a;
  if (mu < 0.0) mu = 0.0;
  // power factor angle including commutation overlap
  const double num = 2.0*mu + std::sin(2.0*a) - std::sin(2.0*(a+mu));
  const double den = std::cos(2.0*a) - std::cos(2.0*(a+mu));
  s.angle = a*180.0/kPi;
  s.tap = fit.tap;
  s.p = internalVoltage(cv, rect, vdc, id)*id;
  s.q = (den > 0.0) ? s.p*num/den : 0.0;
  s.limited = (fit.side != 0);
  return s;
}

} // namespace hvdc_detail

/**
 * Operating point of a line that is not operating
 */
GRIDPACK_HVDC_HD inline HVDCSolution blockedHVDCSolution()
{
  HVDCSolution sol{};
  sol.mode = HVDC_BLOCKED;
  sol.id = 0.0;
  sol.vdcr = 0.0;
  sol.vdci = 0.0;
  sol.limited = false;
  HVDCConverterState zero{};
  zero.angle = 0.0;
  zero.tap = 0.0;
  zero.p = 0.0;
  zero.q = 0.0;
  zero.limited = false;
  sol.rect = zero;
  sol.inv = zero;
  return sol;
}

/**
 * Solve a dc line at the given ac voltage magnitudes of its converter buses.
 * In normal operation the rectifier holds the power or current order and the
 * inverter holds the compounded dc voltage vdci + id*rcomp = vschd. If the
 * inverter cannot hold the voltage with its tap at a limit it runs at minimum
 * extinction angle and sets the dc voltage; a power order is then met with
 * more current, switching to a current order if the dc voltage drops below
 * VCMOD. If the rectifier cannot supply the dc voltage with its tap at a
 * limit it runs at minimum firing angle and the inverter controls the
 * current, reduced by the margin DELTI
 * @param line dc line data
 * @param vr ac voltage magnitude at the rectifier bus (pu)
 * @param vi ac voltage magnitude at the inverter bus (pu)
 * @return operating point of the line
 */
GRIDPACK_HVDC_HD inline HVDCSolution solveTwoTerminalDC(
    const HVDCLineData &line, double vr, double vi)
{
  using hvdc_detail::AngleFit;
  using hvdc_detail::fitAngle;
  using hvdc_detail::heldVoltage;
  using hvdc_detail::positiveRoot;
  HVDCSolution sol = blockedHVDCSolution();
  if (line.mdc != 1 && line.mdc != 2) return sol;
  if (vr <= 0.0 || vi <= 0.0 || line.vschd <= 0.0) return sol;
  if (line.rect.ebas <= 0.0 || line.rect.tr <= 0.0 || line.rect.nb <= 0.0 ||
      line.inv.ebas <= 0.0 || line.inv.tr <= 0.0 || line.inv.nb <= 0.0) {
    return sol;
  }
  const bool power = (line.mdc == 1);
  // a positive power demand is metered at the rectifier
  const bool rect_metered = (line.setvl > 0.0);
  const double pord = std::fabs(line.setvl);
  bool limited = false;

  // Normal operation
  double id = 0.0;
  if (power) {
    const double a = rect_metered ? line.rdc - line.rcomp : -line.rcomp;
    if (!positiveRoot(a, line.vschd, pord, &id)) return sol;
  } else {
    id = line.setvl/1000.0;
  }
  if (id <= 0.0) return sol;
  double vdci = line.vschd - id*line.rcomp;
  double vdcr = vdci + id*line.rdc;
  HVDCMode mode = HVDC_NORMAL;

  // Inverter holds the dc voltage unless it reaches minimum extinction angle
  AngleFit ifit = fitAngle(line.inv, false, vi, vdci, id);
  if (ifit.side < 0) {
    double ai = 0.0;
    double bi = 0.0;
    heldVoltage(line.inv, false, vi, ifit.tap, ifit.c, &ai, &bi);
    mode = HVDC_INV_GAMMA_MIN;
    if (power) {
      // meet the power order at the lower dc voltage
      const double a = rect_metered ? bi + line.rdc : bi;
      double idp = 0.0;
      if (positiveRoot(a, ai, pord, &idp)) {
        id = idp;
      } else {
        // order exceeds the maximum transfer; run at the maximum power current
        limited = true;
        if (a < 0.0) id = -ai/(2.0*a);
      }
      vdci = ai + bi*id;
      if (line.vcmod > 0.0 && vdci < line.vcmod) {
        // switch to the current that gives the order at scheduled voltage
        mode = HVDC_CURRENT_MODE;
        id = pord/line.vschd;
        vdci = ai + bi*id;
      }
    } else {
      vdci = ai + bi*id;
    }
    vdcr = vdci + id*line.rdc;
  } else if (ifit.side > 0) {
    limited = true;
  }
  const double iorder = id;

  // Rectifier holds the current unless it reaches minimum firing angle
  const AngleFit rfit = fitAngle(line.rect, true, vr, vdcr, id);
  if (rfit.side < 0) {
    double ar = 0.0;
    double br = 0.0;
    heldVoltage(line.rect, true, vr, rfit.tap, rfit.c, &ar, &br);
    mode = HVDC_RECT_ALPHA_MIN;
    id = (1.0 - line.delti)*iorder;
    vdcr = ar + br*id;
    vdci = vdcr - id*line.rdc;
    ifit = fitAngle(line.inv, false, vi, vdci, id);
    if (ifit.side != 0) limited = true;
  } else if (rfit.side > 0) {
    limited = true;
  }
  if (id <= 0.0 || vdcr <= 0.0 || vdci <= 0.0) return sol;

  sol.mode = mode;
  sol.id = id;
  sol.vdcr = vdcr;
  sol.vdci = vdci;
  sol.rect = hvdc_detail::converterState(line.rect, true, vr, vdcr, id, rfit);
  sol.inv = hvdc_detail::converterState(line.inv, false, vi, vdci, id, ifit);
  sol.limited = limited;
  return sol;
}

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

#undef GRIDPACK_HVDC_HD

#endif
