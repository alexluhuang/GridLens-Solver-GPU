#!/usr/bin/env python3
"""Build a test network with two-terminal dc lines and distributed generation.

Starts from the public 240-bus WECC case (PSS/E v34) shipped with GridPACK
and adds:
  - three two-terminal dc lines: a power order between two 500 kV buses, a
    current order from a 345 kV to a 230 kV bus, and a blocked line
    (MDC = 0);
  - distributed generation (DGENP, DGENQ) on every fifth load, plus one load
    whose distributed generation is out of service (DGENF = 0).
Each converter transformer ratio is chosen so that the base operating point
sits at a firing angle of 15 degrees (rectifier) or an extinction angle of
19 degrees (inverter) at the bus voltage of the case, inside the limits.

usage: make_dc_case.py SOURCE.raw OUTPUT.raw
"""
import math
import sys

KVDO = 3.0 * math.sqrt(2.0) / math.pi   # no-load dc voltage per kV of ac voltage
KXC = 3.0 / math.pi                    # commutation drop per ohm and kA

# name, mdc, rectifier bus, inverter bus, setvl, vschd (kV), vcmod (kV), rdc
# (ohm), bridges, commutating reactance (ohm)
LINES = [
    ("DC_POWER", 1, 4005, 2401, 1000.0, 500.0, 400.0, 10.0, 2, 8.0),
    ("DC_CURRENT", 2, 6504, 2612, 1600.0, 500.0, 0.0, 12.0, 2, 7.0),
    ("DC_BLOCKED", 0, 4001, 3803, 500.0, 500.0, 0.0, 10.0, 2, 8.0),
]


def fields(line):
    return [f.strip() for f in line.split(",")]


def dc_current(mdc, setvl, vschd, rdc):
    if mdc == 2:
        return setvl / 1000.0
    # power metered at the rectifier: setvl = (vschd + id*rdc) * id
    return (-vschd + math.sqrt(vschd * vschd + 4.0 * rdc * setvl)) / (2.0 * rdc)


def ratio(vdc, rect, nb, xc, idc, vac, ebas, angle):
    """Transformer ratio TR giving the angle at tap 1 (pf_hvdc.hpp fitAngle)"""
    need = vdc / nb + KXC * xc * idc
    return need / (KVDO * vac * ebas * math.cos(math.radians(angle)))


def main():
    src, out = sys.argv[1], sys.argv[2]
    lines = open(src, encoding="latin-1").read().split("\n")
    start = next(i for i, l in enumerate(lines) if "END OF SYSTEM-WIDE DATA" in l.upper())
    end = next(i for i, l in enumerate(lines) if "END OF BUS DATA" in l.upper())
    buses = {}
    for l in lines[start + 1:end]:
        if l.startswith("@!") or not l.strip():
            continue
        f = fields(l)
        buses[int(f[0])] = (float(f[2]), float(f[7]))   # base kV, voltage (pu)

    # Distributed generation on loads
    lstart = end
    lend = next(i for i, l in enumerate(lines) if "END OF LOAD DATA" in l.upper())
    count = 0
    for i in range(lstart + 1, lend):
        if lines[i].startswith("@!") or not lines[i].strip():
            continue
        f = lines[i].split(",")
        if len(f) < 17:
            continue
        count += 1
        if count % 5 == 0:
            f[14], f[15], f[16] = "    60.000", "    15.000", "   1"
        elif count == 7:
            f[14], f[15], f[16] = "    80.000", "    20.000", "   0"
        lines[i] = ",".join(f)

    # Two-terminal dc lines
    records = []
    for name, mdc, ir, ii, setvl, vschd, vcmod, rdc, nb, xc in LINES:
        idc = dc_current(1 if mdc == 0 else mdc, setvl, vschd, rdc)
        vdci = vschd
        vdcr = vdci + idc * rdc
        kvr, vr = buses[ir]
        kvi, vi = buses[ii]
        trr = ratio(vdcr, True, nb, xc, idc, vr, kvr, 15.0)
        tri = ratio(vdci, False, nb, xc, idc, vi, kvi, 19.0)
        records.append(f"'{name}',{mdc},{rdc:.4f},{setvl:.2f},{vschd:.2f},{vcmod:.2f},"
                       f"0.0000,0.10000,'I',0.00,20,1.00000")
        records.append(f"{ir},{nb},20.00,5.00,0.0000,{xc:.4f},{kvr:.2f},{trr:.5f},1.00000,"
                       f"1.10000,0.90000,0.00625,0,0,0,'1',0.0000,0")
        records.append(f"{ii},{nb},25.00,18.00,0.0000,{xc:.4f},{kvi:.2f},{tri:.5f},1.00000,"
                       f"1.10000,0.90000,0.00625,0,0,0,'1',0.0000,0")
    at = next(i for i, l in enumerate(lines) if "END OF TWO-TERMINAL DC DATA" in l.upper())
    lines[at:at] = records
    open(out, "w", encoding="latin-1").write("\n".join(lines))
    print(f"wrote {out}: {len(LINES)} dc lines, distributed generation on "
          f"{count // 5} loads (one more out of service)")


if __name__ == "__main__":
    main()
