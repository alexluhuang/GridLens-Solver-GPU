/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   pf_hvdc.cpp
 *
 * @brief  Steady-state model of line-commutated two-terminal dc lines for
 * the sequential ac/dc power flow
 */
// -------------------------------------------------------------

#include "pf_hvdc.hpp"
#include <cmath>

namespace {

const double PI = std::acos(-1.0);
// no-load dc voltage of a six-pulse bridge per kV of ac voltage
const double KVDO = 3.0*std::sqrt(2.0)/PI;
// commutation voltage drop per ohm of commutating reactance and kA
const double KXC = 3.0/PI;

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
double bridgeVoltage(const gridpack::powerflow::HVDCConverter &cv, double vac,
    double tap)
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
double internalVoltage(const gridpack::powerflow::HVDCConverter &cv, bool rect,
    double vdc, double id)
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
AngleFit fitAngle(const gridpack::powerflow::HVDCConverter &cv, bool rect,
    double vac, double vdc, double id)
{
  AngleFit fit;
  double amin = (cv.anmn < cv.anmx) ? cv.anmn : cv.anmx;
  double amax = (cv.anmn < cv.anmx) ? cv.anmx : cv.anmn;
  double cmax = std::cos(amin*PI/180.0);
  double cmin = std::cos(amax*PI/180.0);
  double need = internalVoltage(cv, rect, vdc, id)/cv.nb + KXC*cv.xc*id;
  fit.tap = cv.tap;
  fit.side = 0;
  fit.c = need/(KVDO*bridgeVoltage(cv, vac, fit.tap));
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
    fit.c = need/(KVDO*bridgeVoltage(cv, vac, tap));
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
void heldVoltage(const gridpack::powerflow::HVDCConverter &cv, bool rect,
    double vac, double tap, double c, double *a, double *b)
{
  *a = cv.nb*KVDO*bridgeVoltage(cv, vac, tap)*c;
  *b = rect ? -cv.nb*KXC*cv.xc - 2.0*cv.nb*cv.rc
            : -cv.nb*KXC*cv.xc + 2.0*cv.nb*cv.rc;
}

/**
 * Smallest positive root of a*x^2 + b*x - c = 0 for c > 0
 * @param x root
 * @return false if there is no positive root
 */
bool positiveRoot(double a, double b, double c, double *x)
{
  double disc = b*b + 4.0*a*c;
  if (disc < 0.0) return false;
  double den = b + std::sqrt(disc);
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
gridpack::powerflow::HVDCConverterState converterState(
    const gridpack::powerflow::HVDCConverter &cv, bool rect, double vac,
    double vdc, double id, const AngleFit &fit)
{
  gridpack::powerflow::HVDCConverterState s;
  double e = bridgeVoltage(cv, vac, fit.tap);
  double a = std::acos(fit.c);
  // overlap angle mu from cos(a) - cos(a+mu) = sqrt(2)*xc*id/e
  double c2 = fit.c - std::sqrt(2.0)*cv.xc*id/e;
  if (c2 < -1.0) c2 = -1.0;
  double mu = std::acos(c2) - a;
  if (mu < 0.0) mu = 0.0;
  // power factor angle including commutation overlap
  double num = 2.0*mu + std::sin(2.0*a) - std::sin(2.0*(a+mu));
  double den = std::cos(2.0*a) - std::cos(2.0*(a+mu));
  s.angle = a*180.0/PI;
  s.tap = fit.tap;
  s.p = internalVoltage(cv, rect, vdc, id)*id;
  s.q = (den > 0.0) ? s.p*num/den : 0.0;
  s.limited = (fit.side != 0);
  return s;
}

}  // namespace

/**
 * Operating point of a line that is not operating
 */
gridpack::powerflow::HVDCSolution gridpack::powerflow::blockedHVDCSolution()
{
  HVDCSolution sol;
  sol.mode = HVDC_BLOCKED;
  sol.id = 0.0;
  sol.vdcr = 0.0;
  sol.vdci = 0.0;
  sol.limited = false;
  HVDCConverterState zero;
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
gridpack::powerflow::HVDCSolution gridpack::powerflow::solveTwoTerminalDC(
    const HVDCLine &line, double vr, double vi)
{
  HVDCSolution sol = blockedHVDCSolution();
  if (line.mdc != 1 && line.mdc != 2) return sol;
  if (vr <= 0.0 || vi <= 0.0 || line.vschd <= 0.0) return sol;
  if (line.rect.ebas <= 0.0 || line.rect.tr <= 0.0 || line.rect.nb <= 0.0 ||
      line.inv.ebas <= 0.0 || line.inv.tr <= 0.0 || line.inv.nb <= 0.0) {
    return sol;
  }
  bool power = (line.mdc == 1);
  // a positive power demand is metered at the rectifier
  bool rect_metered = (line.setvl > 0.0);
  double pord = std::abs(line.setvl);
  bool limited = false;

  // Normal operation
  double id = 0.0;
  if (power) {
    double a = rect_metered ? line.rdc - line.rcomp : -line.rcomp;
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
    double ai, bi;
    heldVoltage(line.inv, false, vi, ifit.tap, ifit.c, &ai, &bi);
    mode = HVDC_INV_GAMMA_MIN;
    if (power) {
      // meet the power order at the lower dc voltage
      double a = rect_metered ? bi + line.rdc : bi;
      double idp;
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
  double iorder = id;

  // Rectifier holds the current unless it reaches minimum firing angle
  AngleFit rfit = fitAngle(line.rect, true, vr, vdcr, id);
  if (rfit.side < 0) {
    double ar, br;
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
  sol.rect = converterState(line.rect, true, vr, vdcr, id, rfit);
  sol.inv = converterState(line.inv, false, vi, vdci, id, ifit);
  sol.limited = limited;
  return sol;
}

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
