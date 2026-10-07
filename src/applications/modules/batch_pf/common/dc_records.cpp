/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   dc_records.cpp
 * @date   2026-10-07
 *
 * @brief Conversion of dc line records, see dc_records.hpp
 */

#include "dc_records.hpp"

namespace gridpack {
namespace batchpf {

namespace {

HVDCConverter converter(const batchpf_dc_converter &r)
{
  HVDCConverter c{};
  c.bus = r.bus;
  c.nb = r.nb;
  c.anmx = r.anmx;
  c.anmn = r.anmn;
  c.rc = r.rc;
  c.xc = r.xc;
  c.ebas = r.ebas;
  c.tr = r.tr;
  c.tap = r.tap;
  c.tmx = r.tmx;
  c.tmn = r.tmn;
  c.stp = r.stp;
  return c;
}

batchpf_dc_converter converterRecord(const HVDCConverter &c, int bus)
{
  batchpf_dc_converter r{};
  r.bus = bus;
  r.nb = c.nb;
  r.anmx = c.anmx;
  r.anmn = c.anmn;
  r.rc = c.rc;
  r.xc = c.xc;
  r.ebas = c.ebas;
  r.tr = c.tr;
  r.tap = c.tap;
  r.tmx = c.tmx;
  r.tmn = c.tmn;
  r.stp = c.stp;
  return r;
}

HVDCConverterState converterState(const batchpf_dc_converter_state &r)
{
  HVDCConverterState c{};
  c.angle = r.angle;
  c.tap = r.tap;
  c.p = r.p;
  c.q = r.q;
  c.limited = r.limited != 0;
  return c;
}

batchpf_dc_converter_state converterStateRecord(const HVDCConverterState &c)
{
  batchpf_dc_converter_state r{};
  r.angle = c.angle;
  r.tap = c.tap;
  r.p = c.p;
  r.q = c.q;
  r.limited = c.limited ? 1 : 0;
  return r;
}

}  // namespace

HVDCLineData dcLineData(const batchpf_dc_line &r)
{
  HVDCLineData l{};
  l.mdc = r.mdc;
  l.rdc = r.rdc;
  l.setvl = r.setvl;
  l.vschd = r.vschd;
  l.vcmod = r.vcmod;
  l.rcomp = r.rcomp;
  l.delti = r.delti;
  l.rect = converter(r.rect);
  l.inv = converter(r.inv);
  return l;
}

batchpf_dc_line dcLineRecord(const HVDCLineData &line, int rect_bus, int inv_bus,
                             const HVDCSolution &reference)
{
  batchpf_dc_line r{};
  r.mdc = line.mdc;
  r.rdc = line.rdc;
  r.setvl = line.setvl;
  r.vschd = line.vschd;
  r.vcmod = line.vcmod;
  r.rcomp = line.rcomp;
  r.delti = line.delti;
  r.rect = converterRecord(line.rect, rect_bus);
  r.inv = converterRecord(line.inv, inv_bus);
  r.reference = dcState(reference);
  return r;
}

HVDCSolution dcSolution(const batchpf_dc_state &r)
{
  HVDCSolution s = gridpack::powerflow::blockedHVDCSolution();
  if (r.mode >= BATCHPF_DC_BLOCKED && r.mode <= BATCHPF_DC_RECT_ALPHA_MIN) {
    s.mode = static_cast<HVDCMode>(r.mode);
  }
  s.id = r.id;
  s.vdcr = r.vdcr;
  s.vdci = r.vdci;
  s.rect = converterState(r.rect);
  s.inv = converterState(r.inv);
  s.limited = r.limited != 0;
  return s;
}

batchpf_dc_state dcState(const HVDCSolution &s)
{
  batchpf_dc_state r{};
  r.mode = static_cast<int32_t>(s.mode);
  r.limited = s.limited ? 1 : 0;
  r.id = s.id;
  r.vdcr = s.vdcr;
  r.vdci = s.vdci;
  r.rect = converterStateRecord(s.rect);
  r.inv = converterStateRecord(s.inv);
  return r;
}

}  // namespace batchpf
}  // namespace gridpack
