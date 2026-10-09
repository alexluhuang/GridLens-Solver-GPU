#!/usr/bin/env python3
#
#     Copyright (c) 2013 Battelle Memorial Institute
#     Licensed under modified BSD License. A copy of this license can be
#     found in the LICENSE file in the top level directory of this
#     distribution.
#
"""Trends in the matrix results: speedups, where the time goes, how each
path scales with ranks and with network size, what the GPU engine did, how
many solver steps each path takes, how much memory the runs used, and
whether the paths flag the same limit violations.

  python3 tools/ca_matrix/analyze_matrix.py [--out matrix/analysis]

Reads matrix/results.jsonl, the logs and small tables in matrix/out/, and
matrix/accuracy.csv if compare_runs.py has written it. Writes, to --out:

  wall.csv          wall time per network, ranks and path, speedups
  steps.csv         every step at every rank count, seconds and share of wall
  savings.csv       seconds each pipeline part saves, stock to optimized CPU
                    and optimized CPU to each GPU path
  scaling.csv       wall time relative to 8 ranks
  size.csv          network size, cases, and time per case
  gpu.csv           GPU engine telemetry of every GPU run
  iterations.csv    solver steps and case outcomes per path (16 ranks)
  violations.csv    agreement of the violation tables with optimized CPU
  memory.csv        peak memory of every run
  backends.csv      Alg 2 against cuDSS, with the fill of the factors
  summary.md        the same, as Markdown tables
  fig_*.svg         figures

Uses only the Python standard library. Times are the middle value
(median) over repeats, as in matrix_report.py.
"""

import argparse
import csv
import math
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
MATRIX = os.path.join(REPO, "matrix")
PATHS = ["stock", "cpu", "alg2", "cudss"]
LABEL = {"stock": "Stock GridPACK", "cpu": "Optimized CPU", "alg2": "Alg 2", "cudss": "cuDSS"}
RANKS = [8, 16, 20]
# Validated with the dataviz skill's palette checker (light surface): slots
# 1-4 of its reference palette. The light aqua and yellow fall below 3:1, so
# every figure has its numbers in summary.md.
COLOR = {"stock": "#2a78d6", "cpu": "#eb6834", "alg2": "#1baf7a", "cudss": "#eda100"}
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# Pipeline parts of the step chart, in pipeline order. Case-loop parts are
# per-rank averages; "waiting" is the rest of the loop; "start-up" is the
# part of the wall time outside every timed phase (MPI and GPU start-up,
# exit).
PARTS = ["start-up and shutdown", "read, base case and case list", "apply and restore outage",
         "solve", "write table rows", "other checks and reporting", "waiting", "merge files"]


def path_of(r):
    return "stock" if r["program"] == "stock" else r["path"]


def local(folder):
    return os.path.join(REPO, os.path.relpath(folder, "/src"))


def step(r, name, kind):
    return r["steps"].get(name, {}).get(kind, 0.0)


def parts_of(r):
    """Seconds of each pipeline part of one run (see PARTS)."""
    loop = step(r, "CA: Solve and Report Cases", "max")
    apply_ = step(r, "CA case: Apply Outage", "avg") + step(r, "CA case: Restore Network", "avg") \
        + step(r, "CA case: Inject GPU Result", "avg")
    solve = step(r, "CA case: CPU Solve", "avg")
    rows = step(r, "CA case: Write Table Rows", "avg")
    report = step(r, "CA case: Check and Report", "avg") - rows
    waiting = max(0.0, loop - apply_ - solve - rows - report)
    setup = sum(step(r, n, "max") for n in ("CA: Read Network", "CA: Base Case",
                                             "CA: Case List and Output Setup"))
    merge = step(r, "CA: Merge Output Files", "max")
    startup = max(0.0, r["wall_s"] - setup - loop - merge)
    return dict(zip(PARTS, [startup, setup, apply_, solve, rows, report, waiting, merge]))


def median_parts(rs):
    ps = [parts_of(r) for r in rs]
    return {k: statistics.median(p[k] for p in ps) for k in PARTS}


def load_runs(path):
    runs = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["returncode"] == 0:
                runs[(r["network"], r["ranks"], path_of(r))].append(r)
    return runs


def first_folder(rs):
    rs = sorted(rs, key=lambda r: r.get("repeat", 1))
    return local(rs[0]["folder"])


def count_lines(path):
    with open(path, errors="replace") as f:
        return sum(1 for _ in f) - 1


def gpu_telemetry(log):
    """The batch path's summary lines of one log."""
    t = {}
    with open(log, errors="replace") as f:
        text = f.read()
    m = re.search(r"model: (\d+) buses, (\d+) directed branch entries", text)
    if m:
        t["buses"], t["branch_entries"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"classified (\d+) cases in [\d.e-]+ s: (\d+) GPU, (\d+) CPU", text)
    if m:
        t["cases"], t["gpu_cases"], t["cpu_cases"] = map(int, m.groups())
    m = re.search(r"plan: (\d+) rows, (\d+) Jacobian entries \(standard layout (\d+), superset "
                  r"overhead ([\d.]+)%\), (\d+) factor entries, levels (\d+)/", text)
    if m:
        t["jacobian_rows"], t["jacobian_entries"], t["standard_entries"] = map(int, m.groups()[:3])
        t["superset_overhead_pct"] = float(m.group(4))
        t["factor_entries"], t["levels"] = int(m.group(5)), int(m.group(6))
    m = re.search(r"(\d+) multiply-adds per factorization", text)
    if m:
        t["multiply_adds"] = int(m.group(1))
    m = re.search(r"GPU: backend (\w+), batch size (\d+), (\d+) cases \((\d+) converged, (\d+) "
                  r"diverged, (\d+) flagged\) in (\d+) submissions, ([\d.]+) s, slot occupancy "
                  r"([\d.]+)%", text)
    if m:
        t["backend"] = m.group(1)
        t["batch"], _, t["gpu_converged"], t["gpu_diverged"], t["gpu_flagged"], \
            t["submissions"] = map(int, m.groups()[1:7])
        t["engine_s"], t["occupancy_pct"] = float(m.group(8)), float(m.group(9))
    m = re.search(r"phases \(s\): (.*)", text)
    if m:
        for name, sec in re.findall(r"([A-Za-z ]+?) ([\d.]+)(?: \([\d.]+ GB/s\))?(?:,|$)",
                                    m.group(1)):
            t["phase " + name.strip()] = float(sec)
    m = re.search(r"paths: .*?(\d+) solved again on the CPU after the GPU \(([^)]*)\)", text)
    if m:
        t["retried_on_cpu"] = int(m.group(1))
        t["retry_reasons"] = m.group(2)
    m = re.search(r"times \(s\): classify ([\d.]+), plan ([\d.]+)", text)
    if m:
        t["classify_s"], t["plan_s"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"GPU Newton iterations per case: (.*)", text)
    if m:
        t["iteration_histogram"] = m.group(1).strip()
    return t


def outcomes(folder):
    """Case outcomes and solver steps from the convergence table."""
    files = glob.glob(os.path.join(folder, "*_convergence.csv"))
    out = {"cases": 0, "status": defaultdict(int), "iterations": []}
    if not files:
        return out
    with open(files[0], errors="replace") as f:
        for row in csv.DictReader(f):
            if row["event_idx"] == "0":
                continue
            out["cases"] += 1
            out["status"][row["status_code"]] += 1
            if row["converged"] == "true" and row["iterations"]:
                out["iterations"].append(int(row["iterations"]))
    return out


def violation_keys(folder):
    files = glob.glob(os.path.join(folder, "*_violations.csv"))
    keys = {}
    if not files:
        return None
    with open(files[0], errors="replace") as f:
        for row in csv.DictReader(f):
            keys[(row["event_idx"], row["type"], row["element"])] = row["severity"]
    return keys


def fmt(v, digits=4):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return "%.*f" % (digits, v)
    return str(v)


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in fields})


def md_table(rows, cols=None):
    if not rows:
        return "(no data)\n"
    cols = cols or list(rows[0])
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c), 2) for c in cols) + " |")
    return "\n".join(lines) + "\n"


# ---- SVG figures -----------------------------------------------------------

class Svg:
    def __init__(self, width, height, title):
        self.w, self.h = width, height
        self.items = ['<rect width="%d" height="%d" fill="%s"/>' % (width, height, SURFACE),
                      self.text(16, 26, title, size=15, weight=600)]

    @staticmethod
    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def text(self, x, y, s, size=12, anchor="start", color=INK, weight=400):
        return ('<text x="%.1f" y="%.1f" font-size="%d" text-anchor="%s" fill="%s" '
                'font-weight="%d">%s</text>' % (x, y, size, anchor, color, weight, self.esc(s)))

    def add(self, item):
        self.items.append(item)

    def label(self, *a, **k):
        self.add(self.text(*a, **k))

    def line(self, x1, y1, x2, y2, color=GRID, width=1):
        self.add('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%g"/>'
                 % (x1, y1, x2, y2, color, width))

    def bar(self, x, y, w, h, color, tip, round_end=None):
        """A bar; round_end 'right' or 'top' rounds that end by 4 px."""
        if w <= 0 or h <= 0:
            return
        r = min(4.0, w / 2, h / 2) if round_end else 0
        if round_end == "right":
            d = ("M%.1f,%.1f h%.1f a%.1f,%.1f 0 0 1 %.1f,%.1f v%.1f a%.1f,%.1f 0 0 1 %.1f,%.1f "
                 "h%.1f z" % (x, y, w - r, r, r, r, r, h - 2 * r, r, r, -r, r, -(w - r)))
        elif round_end == "top":
            d = ("M%.1f,%.1f v%.1f a%.1f,%.1f 0 0 1 %.1f,%.1f h%.1f a%.1f,%.1f 0 0 1 %.1f,%.1f "
                 "v%.1f z" % (x, y + h, -(h - r), r, r, r, -r, w - 2 * r, r, r, r, r, h - r))
        else:
            d = "M%.1f,%.1f h%.1f v%.1f h%.1f z" % (x, y, w, h, -w)
        self.add('<path d="%s" fill="%s"><title>%s</title></path>' % (d, color, self.esc(tip)))

    def legend(self, x, y, entries):
        for name, color in entries:
            self.add('<rect x="%.1f" y="%.1f" width="10" height="10" rx="2" fill="%s"/>'
                     % (x, y - 9, color))
            self.label(x + 14, y, name, size=11, color=INK2)
            x += 24 + 6.5 * len(name)

    def save(self, path):
        with open(path, "w") as f:
            f.write('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
                    'viewBox="0 0 %d %d" font-family="system-ui, -apple-system, Segoe UI, '
                    'Helvetica, Arial, sans-serif">\n' % (self.w, self.h, self.w, self.h))
            f.write("\n".join(self.items))
            f.write("\n</svg>\n")


def nice_max(v):
    for e in range(-3, 7):
        for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8):
            if m * 10 ** e >= v:
                return m * 10 ** e
    return v


def fig_speedup(networks, wall, path):
    """Grouped columns: speedup over stock at 16 ranks, per network."""
    series = ["cpu", "alg2", "cudss"]
    vals = {(n, p): wall.get((n, 16, "stock"), 0) / wall[(n, 16, p)]
            for n in networks for p in series if wall.get((n, 16, p)) and wall.get((n, 16, "stock"))}
    if not vals:
        return
    networks = [n for n in networks if any((n, p) in vals for p in series)]
    top = nice_max(max(vals.values()))
    left, plot_w, plot_h, top_y = 60, 110 * len(networks), 260, 60
    s = Svg(left + plot_w + 30, top_y + plot_h + 70,
            "Speedup over stock GridPACK at 16 processes (times faster)")
    s.legend(left, 46, [(LABEL[p], COLOR[p]) for p in series])
    for k in range(6):
        v = top * k / 5
        y = top_y + plot_h - plot_h * v / top
        s.line(left, y, left + plot_w, y)
        s.label(left - 6, y + 4, ("%g" % v), size=11, anchor="end", color=INK2)
    y1 = top_y + plot_h - plot_h / top
    s.line(left, y1, left + plot_w, y1, color=INK2)
    for i, n in enumerate(networks):
        x0 = left + 110 * i + 15
        for j, p in enumerate(series):
            v = vals.get((n, p))
            if v is None:
                continue
            h = plot_h * v / top
            x = x0 + j * 26
            s.bar(x, top_y + plot_h - h, 24, h, COLOR[p], "%s, %s: %.2fx" % (n, LABEL[p], v),
                  round_end="top")
            if p != "cpu":
                s.label(x + 12, top_y + plot_h - h - 4, "%.1f" % v, size=10, anchor="middle",
                        color=INK2)
        s.label(x0 + 38, top_y + plot_h + 16, short(n), size=11, anchor="middle", color=INK2)
    s.label(left, top_y + plot_h + 50, "The darker rule marks 1x (as fast as stock).", size=11,
            color=INK2)
    s.save(path)


# Networks that may be named in the output. Any other network (for example
# a CEII planning case) is reported only as "private network".
PUBLIC = {"MemphisCase2026_Mar7": "Memphis", "Wisconsin_1664": "Wisconsin",
          "Base_Florida_42GW": "Florida", "Texas7k_20210804": "Texas7k",
          "ACTIVSg10k": "ACTIVSg10k", "Base_MIOHIN_76GW": "MIOHIN"}


def short(n):
    return PUBLIC.get(n, "private network")


def fig_steps(networks, parts, path):
    """Horizontal 100% stacked bars at 16 ranks: share of wall per part."""
    rows = [(n, p) for n in networks for p in PATHS if (n, 16, p) in parts]
    if not rows:
        return
    left, bar_w, top_y = 230, 560, 84
    s = Svg(left + bar_w + 90, top_y + 30 * len(rows) + 16 * len(networks) + 30,
            "Where the time goes at 16 processes (share of wall time; seconds at right)")
    s.legend(16, 46, [(p, SLOTS[i]) for i, p in enumerate(PARTS[:4])])
    s.legend(16, 64, [(p, SLOTS[i + 4]) for i, p in enumerate(PARTS[4:])])
    y, last = top_y, None
    for n, p in rows:
        if n != last:
            y += 16 if last else 0
            s.label(16, y + 12, short(n), size=12, weight=600)
            last = n
        s.label(left - 8, y + 13, LABEL[p], size=11, anchor="end", color=INK2)
        ps = parts[(n, 16, p)]
        total = sum(ps.values())
        x = left
        nonzero = [k for k in PARTS if ps[k] > 0]
        for k in PARTS:
            w = bar_w * ps[k] / total if total else 0
            if w <= 0:
                continue
            color = SLOTS[PARTS.index(k)]
            s.bar(x, y, max(0.0, w - 2), 18, color,
                  "%s, %s, %s: %.2f s (%.1f%%)" % (short(n), LABEL[p], k, ps[k], 100 * ps[k] / total),
                  round_end="right" if k == nonzero[-1] else None)
            x += w
        s.label(left + bar_w + 8, y + 13, "%.1f s" % total, size=11, color=INK2)
        y += 30
    s.save(path)


def fig_scaling(networks, wall, path):
    """Small multiples: wall time at 8, 16 and 20 ranks relative to 8 ranks."""
    nets = [n for n in networks if any((n, 8, p) in wall for p in PATHS)]
    if not nets:
        return
    cols = 4
    cw, ch = 200, 170
    s = Svg(30 + cw * min(cols, len(nets)), 70 + ch * ((len(nets) + cols - 1) // cols),
            "Wall time relative to 8 processes (lower is faster)")
    s.legend(16, 46, [(LABEL[p], COLOR[p]) for p in PATHS])
    for i, n in enumerate(nets):
        ox, oy = 50 + cw * (i % cols), 70 + ch * (i // cols)
        pw, ph = cw - 60, ch - 60
        s.label(ox, oy + 12, short(n), size=12, weight=600)
        lo, hi = 0.4, 1.2
        for v in (0.4, 0.6, 0.8, 1.0, 1.2):
            y = oy + 20 + ph - ph * (v - lo) / (hi - lo)
            s.line(ox, y, ox + pw, y, color=INK2 if v == 1.0 else GRID)
            s.label(ox - 4, y + 4, "%.1f" % v, size=10, anchor="end", color=INK2)
        xs = {8: ox, 16: ox + pw * 8 / 12, 20: ox + pw}
        for r in RANKS:
            s.label(xs[r], oy + 20 + ph + 14, str(r), size=10, anchor="middle", color=INK2)
        for p in PATHS:
            if (n, 8, p) not in wall:
                continue
            pts = [(xs[r], oy + 20 + ph - ph * (min(hi, max(lo, wall[(n, r, p)] / wall[(n, 8, p)]))
                                                - lo) / (hi - lo))
                   for r in RANKS if (n, r, p) in wall]
            s.add('<polyline points="%s" fill="none" stroke="%s" stroke-width="2" '
                  'stroke-linejoin="round" stroke-linecap="round"/>'
                  % (" ".join("%.1f,%.1f" % q for q in pts), COLOR[p]))
            for (x, y), r in zip(pts, [r for r in RANKS if (n, r, p) in wall]):
                s.add('<circle cx="%.1f" cy="%.1f" r="4" fill="%s" stroke="%s" stroke-width="2">'
                      '<title>%s, %s, %d ranks: %.1f s (%.2f of 8 ranks)</title></circle>'
                      % (x, y, COLOR[p], SURFACE, short(n), LABEL[p], r, wall[(n, r, p)],
                         wall[(n, r, p)] / wall[(n, 8, p)]))
    s.label(16, s.h - 8, "x axis: processes (MPI ranks)", size=10, color=INK2)
    s.save(path)


def fig_size(sizes, path):
    """Milliseconds per case against buses, one line per path (log scale)."""
    pts = {p: sorted((d["buses"], d["ms_" + p]) for d in sizes if d.get("ms_" + p))
           for p in PATHS}
    allv = [v for p in pts for _, v in pts[p]]
    allb = [b for p in pts for b, _ in pts[p]]
    if not allv:
        return
    left, top_y, pw, ph = 70, 66, 560, 300
    s = Svg(left + pw + 120, top_y + ph + 60,
            "Time per contingency case at 16 processes (ms, log scale) against network size")
    s.legend(left, 46, [(LABEL[p], COLOR[p]) for p in PATHS])
    lo = 10 ** math.floor(math.log10(min(allv)))
    hi = 10 ** math.ceil(math.log10(max(allv)))
    blo, bhi = 0, nice_max(max(allb))

    def y_of(v):
        return top_y + ph - ph * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))

    def x_of(b):
        return left + pw * (b - blo) / (bhi - blo)
    e = math.log10(lo)
    while e <= math.log10(hi) + 1e-9:
        y = y_of(10 ** e)
        s.line(left, y, left + pw, y)
        s.label(left - 6, y + 4, "%g" % 10 ** e, size=10, anchor="end", color=INK2)
        e += 1
    for k in range(6):
        b = bhi * k / 5
        s.label(x_of(b), top_y + ph + 16, "{:,}".format(int(b)), size=10, anchor="middle",
                color=INK2)
    s.label(left + pw / 2, top_y + ph + 36, "buses", size=11, anchor="middle", color=INK2)
    for p in PATHS:
        if not pts[p]:
            continue
        xy = [(x_of(b), y_of(v)) for b, v in pts[p]]
        s.add('<polyline points="%s" fill="none" stroke="%s" stroke-width="2" '
              'stroke-linejoin="round" stroke-linecap="round"/>'
              % (" ".join("%.1f,%.1f" % q for q in xy), COLOR[p]))
        for (x, y), (b, v) in zip(xy, pts[p]):
            s.add('<circle cx="%.1f" cy="%.1f" r="4" fill="%s" stroke="%s" stroke-width="2">'
                  '<title>%s, %d buses: %.3f ms per case</title></circle>'
                  % (x, y, COLOR[p], SURFACE, LABEL[p], b, v))
    s.save(path)


def fig_backend(rows, path):
    """cuDSS time over Alg 2 time against fill, for factorization and solve."""
    pts = [(r["fill (factor entries per Jacobian entry)"], r["cuDSS / Alg 2 factor"],
            r["cuDSS / Alg 2 solve"], r["network"]) for r in rows
           if r.get("cuDSS / Alg 2 factor") and r.get("cuDSS / Alg 2 solve")]
    if not pts:
        return
    left, top_y, pw, ph = 70, 66, 520, 280
    s = Svg(left + pw + 150, top_y + ph + 60,
            "cuDSS time divided by Alg 2 time, against fill of the factors (16 processes)")
    series = [("factorization", 1, "#2a78d6"), ("triangular solves", 2, "#eb6834")]
    s.legend(left, 46, [(name, color) for name, _, color in series])
    xlo = math.floor(2 * min(p[0] for p in pts) - 0.2) / 2
    xhi = math.ceil(2 * max(p[0] for p in pts) + 0.2) / 2
    ylo, yhi = 0.0, nice_max(max(max(p[1], p[2]) for p in pts))

    def x_of(v):
        return left + pw * (v - xlo) / (xhi - xlo)

    def y_of(v):
        return top_y + ph - ph * (v - ylo) / (yhi - ylo)
    for k in range(6):
        v = ylo + (yhi - ylo) * k / 5
        s.line(left, y_of(v), left + pw, y_of(v))
        s.label(left - 6, y_of(v) + 4, "%g" % v, size=10, anchor="end", color=INK2)
    s.line(left, y_of(1), left + pw, y_of(1), color=INK2)
    s.label(left + pw + 6, y_of(1) + 4, "equal time", size=10, color=INK2)
    v = xlo
    while v <= xhi + 1e-9:
        s.label(x_of(v), top_y + ph + 16, "%.1f" % v, size=10, anchor="middle", color=INK2)
        v += 0.5
    s.label(left + pw / 2, top_y + ph + 36,
            "factor entries per Jacobian entry (more fill to the right)", size=11,
            anchor="middle", color=INK2)
    for name, idx, color in series:
        for p in pts:
            s.add('<circle cx="%.1f" cy="%.1f" r="5" fill="%s" stroke="%s" stroke-width="2">'
                  '<title>%s, %s: cuDSS/Alg 2 = %.2f at fill %.2f</title></circle>'
                  % (x_of(p[0]), y_of(p[idx]), color, SURFACE, short(p[3]), name, p[idx], p[0]))
    # labels alternate above and below the factorization points, left to
    # right, so that neighbours do not collide
    for i, p in enumerate(sorted(pts)):
        dy = -10 if i % 2 == 0 else 18
        s.label(x_of(p[0]), y_of(p[1]) + dy, short(p[3]), size=10, anchor="middle",
                color=INK2)
    s.label(left, top_y + ph + 54, "Below the rule cuDSS is faster; above it Alg 2 is faster.",
            size=11, color=INK2)
    s.save(path)


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(MATRIX, "results.jsonl"))
    ap.add_argument("--accuracy", default=os.path.join(MATRIX, "accuracy.csv"))
    ap.add_argument("--out", default=os.path.join(MATRIX, "analysis"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load_runs(args.results)
    order = {}
    for (n, r, p), rs in runs.items():
        order.setdefault(n, len(order))
    networks = sorted(order, key=order.get)   # the campaign runs smallest first

    def name(n):
        return n if n in PUBLIC else "private network"

    wall = {k: statistics.median(r["wall_s"] for r in rs) for k, rs in runs.items()}
    parts = {k: median_parts(rs) for k, rs in runs.items()}
    busy = {k: sum(1 for r in rs if r.get("machine_busy")) for k, rs in runs.items()}

    # wall.csv: speedups
    wall_rows = []
    for n in networks:
        for r in RANKS:
            for p in PATHS:
                k = (n, r, p)
                if k not in wall:
                    continue
                st, cpu = wall.get((n, r, "stock")), wall.get((n, r, "cpu"))
                wall_rows.append({"network": name(n), "ranks": r, "path": LABEL[p],
                                  "runs": len(runs[k]), "busy runs": busy[k],
                                  "wall (s)": wall[k],
                                  "spread between runs (%)": 100 * (max(x["wall_s"] for x in runs[k])
                                                                    - min(x["wall_s"] for x in runs[k]))
                                  / wall[k] if len(runs[k]) > 1 else None,
                                  "speedup vs stock": st / wall[k] if st else None,
                                  "speedup vs optimized CPU": cpu / wall[k] if cpu else None})
    write_csv(os.path.join(args.out, "wall.csv"), wall_rows)

    # steps.csv
    step_rows = []
    for n in networks:
        for r in RANKS:
            for p in PATHS:
                k = (n, r, p)
                if k not in parts:
                    continue
                total = sum(parts[k].values())
                row = {"network": name(n), "ranks": r, "path": LABEL[p], "wall (s)": wall[k]}
                for part in PARTS:
                    row[part + " (s)"] = parts[k][part]
                for part in PARTS:
                    row[part + " (%)"] = 100 * parts[k][part] / total if total else None
                step_rows.append(row)
    write_csv(os.path.join(args.out, "steps.csv"), step_rows)

    # savings.csv: which parts of the pipeline account for each step's saving
    save_rows = []
    for n in networks:
        for r in RANKS:
            for a, b in (("stock", "cpu"), ("cpu", "alg2"), ("cpu", "cudss")):
                if (n, r, a) not in parts or (n, r, b) not in parts:
                    continue
                pa, pb = parts[(n, r, a)], parts[(n, r, b)]
                total = sum(pa.values()) - sum(pb.values())
                row = {"network": name(n), "ranks": r, "from": LABEL[a], "to": LABEL[b],
                       "saving (s)": total}
                for part in PARTS:
                    row[part + " (s)"] = pa[part] - pb[part]
                for part in PARTS:
                    row[part + " (% of saving)"] = 100 * (pa[part] - pb[part]) / total if total else None
                save_rows.append(row)
    write_csv(os.path.join(args.out, "savings.csv"), save_rows)

    # scaling.csv
    scale_rows = []
    for n in networks:
        for p in PATHS:
            if (n, 8, p) not in wall:
                continue
            row = {"network": name(n), "path": LABEL[p]}
            for r in RANKS:
                if (n, r, p) in wall:
                    row["%d ranks (s)" % r] = wall[(n, r, p)]
                    row["%d ranks / 8 ranks" % r] = wall[(n, r, p)] / wall[(n, 8, p)]
            # parallel efficiency of the case loop from 8 to 16 ranks
            if (n, 16, p) in parts:
                l8 = sum(parts[(n, 8, p)][x] for x in PARTS[2:7])
                l16 = sum(parts[(n, 16, p)][x] for x in PARTS[2:7])
                row["case loop 8->16 efficiency (%)"] = 100 * l8 / (2 * l16) if l16 else None
            scale_rows.append(row)
    write_csv(os.path.join(args.out, "scaling.csv"), scale_rows)

    # gpu.csv and network sizes
    gpu_rows, info = [], {}
    for n in networks:
        for r in RANKS:
            for p in ("alg2", "cudss"):
                k = (n, r, p)
                if k not in runs:
                    continue
                t = gpu_telemetry(os.path.join(first_folder(runs[k]), "log.txt"))
                info.setdefault(n, {}).update({x: t[x] for x in t if x in (
                    "buses", "branch_entries", "cases", "jacobian_rows", "jacobian_entries",
                    "standard_entries", "superset_overhead_pct", "factor_entries", "levels",
                    "multiply_adds")})
                row = {"network": name(n), "ranks": r, "path": LABEL[p]}
                for x in ("cases", "gpu_cases", "cpu_cases", "retried_on_cpu", "retry_reasons",
                          "batch", "submissions", "engine_s", "occupancy_pct", "classify_s",
                          "plan_s"):
                    row[x] = t.get(x)
                for x in sorted(t):
                    if x.startswith("phase "):
                        row[x + " (s)"] = t[x]
                loop = statistics.median(step(x, "CA: Solve and Report Cases", "max")
                                         for x in runs[k])
                if t.get("engine_s") and loop:
                    row["GPU busy share of case loop (%)"] = 100 * t["engine_s"] / loop
                if t.get("engine_s") and t.get("gpu_cases"):
                    row["engine ms per GPU case"] = 1000 * t["engine_s"] / t["gpu_cases"]
                if t.get("phase factor") and t.get("multiply_adds") and t.get("gpu_cases"):
                    row["factor time per multiply-add per case (ns)"] = \
                        1e9 * t["phase factor"] / (t["multiply_adds"] * t["gpu_cases"])
                row["iteration histogram"] = t.get("iteration_histogram")
                gpu_rows.append(row)
    write_csv(os.path.join(args.out, "gpu.csv"), gpu_rows)

    # iterations.csv: outcomes and solver steps at 16 ranks
    it_rows, stat = [], {}
    for n in networks:
        for p in PATHS:
            k = (n, 16, p)
            if k not in runs:
                continue
            o = outcomes(first_folder(runs[k]))
            stat[k] = o
            info.setdefault(n, {}).setdefault("cases", o["cases"])
            its = o["iterations"]
            row = {"network": name(n), "path": LABEL[p], "cases": o["cases"]}
            for code in sorted(o["status"]):
                row[code] = o["status"][code]
            row["mean solver steps (converged)"] = statistics.mean(its) if its else None
            row["median solver steps"] = statistics.median(its) if its else None
            row["max solver steps"] = max(its) if its else None
            it_rows.append(row)
    write_csv(os.path.join(args.out, "iterations.csv"), it_rows)

    # violations.csv: agreement with optimized CPU at 16 ranks
    vio_rows = []
    for n in networks:
        if (n, 16, "cpu") not in runs:
            continue
        ref = violation_keys(first_folder(runs[(n, 16, "cpu")]))
        if ref is None:
            continue
        for p in ("stock", "alg2", "cudss"):
            if (n, 16, p) not in runs:
                continue
            other = violation_keys(first_folder(runs[(n, 16, p)]))
            if other is None:
                continue
            both = ref.keys() & other.keys()
            same_sev = sum(1 for x in both if ref[x] == other[x])
            vio_rows.append({"network": name(n), "path": LABEL[p],
                             "optimized CPU violations": len(ref), "path violations": len(other),
                             "in both": len(both), "only optimized CPU": len(ref) - len(both),
                             "only this path": len(other) - len(both),
                             "same severity (% of both)": 100 * same_sev / len(both) if both else None})
    write_csv(os.path.join(args.out, "violations.csv"), vio_rows)

    # memory.csv
    mem_rows = []
    for n in networks:
        for r in RANKS:
            for p in PATHS:
                k = (n, r, p)
                if k not in runs:
                    continue
                mem_rows.append({"network": name(n), "ranks": r, "path": LABEL[p],
                                 "peak container memory (GB)": max(x.get("peak_memory_gb") or 0
                                                                   for x in runs[k]),
                                 "peak program memory (GB)": max(x.get("peak_program_memory_gb") or 0
                                                                 for x in runs[k])})
    write_csv(os.path.join(args.out, "memory.csv"), mem_rows)

    # size.csv: time per case
    size_rows = []
    for n in networks:
        d = info.get(n, {})
        row = {"network": name(n), "buses": d.get("buses"), "cases": d.get("cases"),
               "branch entries": d.get("branch_entries"),
               "Jacobian rows": d.get("jacobian_rows"), "Jacobian entries": d.get("jacobian_entries"),
               "superset overhead (%)": d.get("superset_overhead_pct"),
               "factor entries": d.get("factor_entries"), "levels": d.get("levels"),
               "multiply-adds per factorization": d.get("multiply_adds")}
        for p in PATHS:
            if (n, 16, p) in wall and d.get("cases"):
                row["ms_" + p] = 1000 * wall[(n, 16, p)] / d["cases"]
        size_rows.append(row)
    write_csv(os.path.join(args.out, "size.csv"), size_rows)

    # backends.csv: Alg 2 against cuDSS at 16 ranks, with the plan's fill
    tele = {(r["network"], r["path"]): r for r in gpu_rows if r["ranks"] == 16}
    back_rows = []
    for n in networks:
        d = info.get(n, {})
        a, c = tele.get((name(n), LABEL["alg2"])), tele.get((name(n), LABEL["cudss"]))
        if not (a and c and d.get("factor_entries") and d.get("jacobian_entries")):
            continue
        row = {"network": name(n),
               "fill (factor entries per Jacobian entry)": d["factor_entries"] / d["jacobian_entries"],
               "multiply-adds per level": d["multiply_adds"] / d["levels"] if d.get("levels") else None,
               "levels": d.get("levels")}
        for what in ("factor", "solve"):
            fa, fc = a.get("phase " + what + " (s)"), c.get("phase " + what + " (s)")
            row["Alg 2 %s (s)" % what], row["cuDSS %s (s)" % what] = fa, fc
            row["cuDSS / Alg 2 %s" % what] = fc / fa if fa and fc else None
        row["Alg 2 wall (s)"], row["cuDSS wall (s)"] = wall.get((n, 16, "alg2")), wall.get((n, 16, "cudss"))
        back_rows.append(row)
    write_csv(os.path.join(args.out, "backends.csv"), back_rows)

    # figures
    fig_backend(back_rows, os.path.join(args.out, "fig_backends.svg"))
    fig_speedup(networks, wall, os.path.join(args.out, "fig_speedup.svg"))
    fig_steps(networks, parts, os.path.join(args.out, "fig_steps.svg"))
    fig_scaling(networks, wall, os.path.join(args.out, "fig_scaling.svg"))
    fig_size([r for r in size_rows if r.get("buses")], os.path.join(args.out, "fig_size.svg"))

    # summary.md
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("# Matrix analysis\n\n")
        f.write("## Wall time and speedup\n\n" + md_table(wall_rows) + "\n")
        f.write("## Pipeline parts (seconds)\n\n" + md_table(
            step_rows, ["network", "ranks", "path", "wall (s)"] + [x + " (s)" for x in PARTS]) + "\n")
        f.write("## Where each saving comes from (seconds saved per part)\n\n" + md_table(
            save_rows, ["network", "ranks", "from", "to", "saving (s)"] + [x + " (s)" for x in PARTS])
            + "\n")
        f.write("## Rank scaling\n\n" + md_table(scale_rows) + "\n")
        f.write("## Network size and time per case (ms, 16 ranks)\n\n" + md_table(size_rows) + "\n")
        f.write("## GPU engine\n\n" + md_table(gpu_rows) + "\n")
        f.write("## Alg 2 against cuDSS (16 ranks)\n\n" + md_table(back_rows) + "\n")
        f.write("## Outcomes and solver steps (16 ranks)\n\n" + md_table(it_rows) + "\n")
        f.write("## Violation agreement with optimized CPU (16 ranks)\n\n" + md_table(vio_rows) + "\n")
        f.write("## Peak memory\n\n" + md_table(mem_rows) + "\n")
        if os.path.exists(args.accuracy):
            with open(args.accuracy) as a:
                acc = [r for r in csv.DictReader(a)]
            for r in acc:
                r["network"] = name(r.get("network", ""))
            f.write("## Accuracy against optimized CPU (16 ranks)\n\n" + md_table(acc) + "\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
