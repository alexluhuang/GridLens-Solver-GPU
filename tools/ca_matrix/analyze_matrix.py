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
  waits.csv         GPU paths' start-up and waiting, from runs with the idle
                    timers (--timing-check)
  backends.csv      Alg 2 against cuDSS, with the fill of the factors
  summary.md        the same, as Markdown tables
  fig_*.svg         figures; fig_wall_cpu_gpu and fig_accuracy compare
                    stock ("CPU") with the cuDSS path ("GPU") at 16 ranks

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
IDLE = "#a3a19b"   # waiting is idle time, shown in a neutral grey

# Pipeline parts of the step chart, in pipeline order. Case-loop parts are
# per-rank averages. "launch and exit" is wall time outside GridPACK's
# "Total Application" timer (starting the MPI processes, MPI set-up, exit).
# "GPU start-up and preparation" is loading the plugin and starting the
# device ("CA: GPU Start-up", or before that timer existed, the part of
# "Total Application" outside every stage) plus "GPU batch: prepare".
# "waiting" is case-loop time in no per-case step: on the GPU paths mostly
# waiting for GPU results, on all paths the wait for the last rank.
PARTS = ["launch and exit", "GPU start-up and preparation", "read, base case and case list",
         "apply and restore outage", "solve", "write table rows", "other checks and reporting",
         "waiting", "merge files"]
LOOP_PARTS = PARTS[3:8]
STAGES = ["CA: Read Network", "CA: Base Case", "CA: Case List and Output Setup",
          "CA: Solve and Report Cases", "CA: Merge Output Files"]


def path_of(r):
    return "stock" if r["program"] == "stock" else r["path"]


def local(folder):
    return os.path.join(REPO, os.path.relpath(folder, "/src"))


def step(r, name, kind):
    return r["steps"].get(name, {}).get(kind, 0.0)


def log_timers(folder):
    """Every GridPACK timer category of a run's log: {name: {avg, max}}."""
    found = {}
    try:
        with open(os.path.join(folder, "log.txt"), errors="replace") as f:
            text = f.read()
    except OSError:
        return found
    for m in re.finditer(r"Timing statistics for: (.*)\n\s+Average time:\s+([\d.]+)\n"
                         r"\s+Maximum time:\s+([\d.]+)", text):
        found[m.group(1).strip()] = {"avg": float(m.group(2)), "max": float(m.group(3))}
    return found


def parts_of(r):
    """Seconds of each pipeline part of one run (see PARTS)."""
    t = r.get("log_timers") or {}

    def tmax(name):
        return t.get(name, {}).get("max", step(r, name, "max"))
    loop = step(r, "CA: Solve and Report Cases", "max")
    apply_ = step(r, "CA case: Apply Outage", "avg") + step(r, "CA case: Restore Network", "avg") \
        + step(r, "CA case: Inject GPU Result", "avg")
    solve = step(r, "CA case: CPU Solve", "avg")
    rows = step(r, "CA case: Write Table Rows", "avg")
    report = step(r, "CA case: Check and Report", "avg") - rows
    waiting = max(0.0, loop - apply_ - solve - rows - report)
    setup = sum(step(r, n, "max") for n in STAGES[:3])
    merge = step(r, "CA: Merge Output Files", "max")
    total = t.get("Total Application", {}).get("max")
    prep = tmax("GPU batch: prepare (export, classify, plan)")
    if "CA: GPU Start-up" in t:
        gpu = tmax("CA: GPU Start-up") + prep
    elif total is not None and r["path"] != "cpu":
        # per-rank averages, since different stages peak on different ranks
        tavg = t.get("Total Application", {}).get("avg", total)
        gpu = max(0.0, tavg - sum(t.get(n, {}).get("avg", 0.0) for n in STAGES))
    else:
        gpu = 0.0
    if total is not None:
        launch = max(0.0, r["wall_s"] - total)
    else:
        launch = max(0.0, r["wall_s"] - setup - loop - merge - gpu)
    return dict(zip(PARTS, [launch, gpu, setup, apply_, solve, rows, report, waiting, merge]))


def median_parts(rs):
    ps = [parts_of(r) for r in rs]
    return {k: statistics.median(p[k] for p in ps) for k in PARTS}


def load_runs(path):
    runs = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["returncode"] == 0:
                r["log_timers"] = log_timers(local(r["folder"]))
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
        """A bar; round_end 'right', 'left' or 'top' rounds that end by 4 px."""
        if w <= 0 or h <= 0:
            return
        r = min(4.0, w / 2, h / 2) if round_end else 0
        if round_end == "right":
            d = ("M%.1f,%.1f h%.1f a%.1f,%.1f 0 0 1 %.1f,%.1f v%.1f a%.1f,%.1f 0 0 1 %.1f,%.1f "
                 "h%.1f z" % (x, y, w - r, r, r, r, r, h - 2 * r, r, r, -r, r, -(w - r)))
        elif round_end == "left":
            d = ("M%.1f,%.1f h%.1f a%.1f,%.1f 0 0 0 %.1f,%.1f v%.1f a%.1f,%.1f 0 0 0 %.1f,%.1f "
                 "h%.1f z" % (x + w, y, -(w - r), r, r, -r, r, h - 2 * r, r, r, r, r, w - r))
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


def fig_columns(networks, vals, series, labels, title, notes, path, unit="", extra=None,
                group=130):
    """Grouped columns, one group per network; vals[(network, path)].

    series is [(path, legend label)]; labels names each network under its
    group, one line per "\n". extra[(network, path)] is bold text after
    that column's value, which then starts at the column's left edge.
    """
    extra = extra or {}
    networks = [n for n in networks if any((n, p) in vals for p, _ in series)]
    if not networks:
        return
    group = max(30 * len(series) + 40, group)
    lines = max(len(labels.get(n, short(n)).split("\n")) for n in networks)
    foot = 14 * (lines - 1)
    left, plot_w, plot_h, top_y = 60, group * len(networks), 260, 60
    s = Svg(max(left + plot_w + 50, 560), top_y + plot_h + 54 + foot + 16 * len(notes), title)
    if len(series) > 1:   # one series needs no legend; the title names it
        s.legend(left, 46, [(lab, COLOR[p]) for p, lab in series])
    hi = nice_max(max(vals.values()))

    def y_of(v):
        return top_y + plot_h - plot_h * v / hi
    for k in range(6):
        v = hi * k / 5
        s.line(left, y_of(v), left + plot_w, y_of(v))
        s.label(left - 6, y_of(v) + 4, "%g" % v, size=11, anchor="end", color=INK2)
    for i, n in enumerate(networks):
        x0 = left + group * i + (group - 30 * len(series)) / 2
        for j, (p, lab) in enumerate(series):
            v = vals.get((n, p))
            if v is None:
                continue
            x = x0 + j * 30
            s.bar(x, y_of(v), 28, y_of(0) - y_of(v), COLOR[p],
                  "%s, %s: %.2f%s" % (short(n), lab, v, unit), round_end="top")
            value = ("%.0f" if v >= 100 else "%.1f") % v
            if (n, p) in extra:
                s.add('<text x="%.1f" y="%.1f" font-size="10" fill="%s">%s <tspan fill="%s" '
                      'font-weight="600">%s</tspan></text>'
                      % (x + 2, y_of(v) - 4, INK2, value, INK, Svg.esc(extra[(n, p)])))
            else:
                s.label(x + 14, y_of(v) - 4, value, size=10, anchor="middle", color=INK2)
        for k, line in enumerate(labels.get(n, short(n)).split("\n")):
            s.label(x0 + 15 * len(series) - 1, top_y + plot_h + 16 + 14 * k, line, size=11,
                    anchor="middle", color=INK2)
    for k, note in enumerate(notes):
        s.label(left, top_y + plot_h + 50 + foot + 16 * k, note, size=11, color=INK2)
    s.save(path)


def fig_cpu_gpu(networks, wall, buses, out):
    """Stock GridPACK ("CPU") against the cuDSS path ("GPU") at 16 ranks:
    wall seconds, with the GPU's speedup. Stock cannot solve the private
    case's base case, so there the optimized CPU path is the CPU run; the
    report's caption says so, the figure does not."""
    has = [n for n in networks if (n, 16, "cudss") in wall
           and ((n, 16, "stock") in wall or (n, 16, "cpu") in wall)]
    base = {n: "stock" if (n, 16, "stock") in wall else "cpu" for n in has}
    speed = {(n, "cudss"): wall[(n, 16, base[n])] / wall[(n, 16, "cudss")] for n in has}
    # the bus count goes on a second line, to keep the groups narrow
    names = {n: short(n) + ("\n(%s)" % format(buses[n], ",") if buses.get(n) else "")
             for n in has}
    secs = {(n, "stock"): wall[(n, 16, base[n])] for n in has}
    secs.update({(n, "cudss"): wall[(n, 16, "cudss")] for n in has})
    fig_columns(has, secs, [("stock", "CPU"), ("cudss", "GPU")],
                names, "Wall time at 16 processes (seconds)",
                ["CPU = stock GridPACK, GPU = the cuDSS path, with how many times faster it is. "
                 "Buses in parentheses. Lower is faster."],
                os.path.join(out, "fig_wall_cpu_gpu.svg"), unit=" s",
                extra={(n, "cudss"): "(%.1f\u00d7)" % speed[(n, "cudss")] for n in has}, group=100)


ACC_VARS = [("utilization (%)", "Loading (% pts)"), ("complex power (MVA)", "S (MVA)"),
            ("real power (MW)", "P (MW)"), ("reactive power (MVAr)", "Q (MVAr)"),
            ("bus voltage (pu)", "V (pu)"), ("bus angle (deg)", "Angle (deg)")]
# Bins of the largest difference. csv_flat prints four decimals, so 0.0001
# is one printed digit. Steps 100/250/300/450/600 of the dataviz blue ramp.
ACC_BINS = [(0.0001, "#cde2fb", INK, "at most 1 printed digit (0.0001)"),
            (0.01, "#86b6ef", INK, "up to 0.01"),
            (1.0, "#6da7ec", INK, "up to 1"),
            (2.0, "#2a78d6", "#ffffff", "1 to 2"),
            (math.inf, "#184f95", "#ffffff", "over 2")]


def fig_accuracy(networks, acc_path, path):
    """Heat table: largest difference from the optimized CPU path at 16
    ranks, per network, variable and version (stock = CPU, cuDSS = GPU)."""
    if not os.path.exists(acc_path):
        return
    acc = {}
    with open(acc_path) as f:
        for r in csv.DictReader(f):
            p = "stock" if r["program"] == "stock" else r["path"]
            acc[(r["network"], p, r["variable"])] = r
    series = [("stock", "CPU"), ("cudss", "GPU")]
    nets = [n for n in networks if any((n, p, ACC_VARS[0][0]) in acc for p, _ in series)]
    if not nets:
        return
    left, cw, rh, top_y = 170, 92, 24, 104
    width = left + cw * (len(ACC_VARS) + 1) + 20
    s = Svg(width, top_y + (2 * rh + 12) * len(nets) + 64,
            "Largest absolute difference from the optimized CPU path, 16 processes (column units)")
    s.legend(16, 46, [(lab, c) for _, c, _, lab in ACC_BINS])
    for j, (_, head) in enumerate(ACC_VARS + [(None, "Rows > 1% off")]):
        s.label(left + cw * j + cw / 2, top_y - 10, head, size=11, anchor="middle", color=INK2,
                weight=600)
    y = top_y
    for n in nets:
        s.label(16, y + rh + 5, short(n), size=12, weight=600)
        for p, lab in series:
            s.label(left - 8, y + 16, lab, size=11, anchor="end", color=INK2)
            for j, (var, head) in enumerate(ACC_VARS):
                x = left + cw * j
                r = acc.get((n, p, var))
                if r is None:
                    s.add('<rect x="%.1f" y="%.1f" width="%d" height="%d" rx="3" fill="none" '
                          'stroke="%s"/>' % (x + 1, y + 1, cw - 2, rh - 2, GRID))
                    s.label(x + cw / 2, y + 16, "not run", size=10, anchor="middle", color=INK2)
                    continue
                v = float(r["max abs diff"])
                _, fill, ink, _ = next(b for b in ACC_BINS if v <= b[0])
                s.add('<rect x="%.1f" y="%.1f" width="%d" height="%d" rx="3" fill="%s">'
                      '<title>%s, %s, %s: largest %.4f, mean %.4f</title></rect>'
                      % (x + 1, y + 1, cw - 2, rh - 2, fill, short(n), lab, head, v,
                         float(r["mean abs diff"])))
                s.label(x + cw / 2, y + 16, "%.4f" % v, size=10, anchor="middle", color=ink)
            # share of rows more than 1% apart: the largest over the variables
            shares = [float(acc[(n, p, v)]["% rows over 1%"]) for v, _ in ACC_VARS
                      if (n, p, v) in acc]
            x = left + cw * len(ACC_VARS)
            s.label(x + cw / 2, y + 16, "%.4f%%" % max(shares) if shares else "not run",
                    size=10, anchor="middle", color=INK2)
            y += rh
        y += 12
    mean = max(float(acc[(n, p, v)]["mean abs diff"]) for n in nets for p, _ in series
               for v, _ in ACC_VARS if (n, p, v) in acc)
    s.label(16, y + 14, "CPU = stock GridPACK, GPU = the cuDSS path. Stock cannot run the "
            "private case. Mean differences are at most %.4f." % mean, size=11, color=INK2)
    s.label(16, y + 30, "Memphis on GPU: one ill-conditioned case (1068); stock on ACTIVSg10k: "
            "voltage setting carried between cases.", size=11, color=INK2)
    s.save(path)


# Networks that may be named in the output. Any other network (for example
# a CEII planning case) is reported only as "private network".
PUBLIC = {"MemphisCase2026_Mar7": "Memphis", "Wisconsin_1664": "Wisconsin",
          "Base_Florida_42GW": "Florida", "Texas7k_20210804": "Texas7k",
          "ACTIVSg10k": "ACTIVSg10k", "Base_MIOHIN_76GW": "MIOHIN",
          "ACTIVSg25k": "ACTIVSg25k", "training": "training",
          "Midwest24k_20220923": "Midwest24k"}


def short(n):
    return PUBLIC.get(n, "private network")


def part_color(part):
    if part == "waiting":
        return IDLE
    i = PARTS.index(part)
    return SLOTS[i if i < PARTS.index("waiting") else i - 1]


def fig_steps(networks, parts, path):
    """Horizontal 100% stacked bars at 16 ranks: share of wall per part."""
    rows = [(n, p) for n in networks for p in PATHS if (n, 16, p) in parts]
    if not rows:
        return
    left, bar_w, top_y = 230, 560, 84
    s = Svg(left + bar_w + 90, top_y + 30 * len(rows) + 16 * len(networks) + 30,
            "Where the time goes at 16 processes (share of wall time; seconds at right)")
    s.legend(16, 46, [(p, part_color(p)) for p in PARTS[:5]])
    s.legend(16, 64, [(p, part_color(p)) for p in PARTS[5:]])
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
            color = part_color(k)
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
    """Which method factorizes faster, network by network, sorted by fill.

    One horizontal bar per network from a centre line: to the left when
    cuDSS is faster, to the right when Alg 2 is faster, its length the
    factor by which it is faster (logarithmic, so 2x left and 2x right are
    the same length)."""
    pts = sorted((r["fill (factor entries per Jacobian entry)"], r["cuDSS / Alg 2 factor"],
                  r["network"]) for r in rows if r.get("cuDSS / Alg 2 factor"))
    if not pts:
        return
    label_w, half, top_y, row_h = 230, 230, 92, 34
    mid = label_w + half
    s = Svg(label_w + 2 * half + 40, top_y + row_h * len(pts) + 70,
            "Which method factorizes faster, by fill of the factors (16 processes)")
    s.legend(16, 46, [("cuDSS faster", COLOR["cudss"]), ("Alg 2 faster", COLOR["alg2"])])
    s.label(mid - 8, top_y - 10, "\u2190 cuDSS faster", size=11, anchor="end", color=INK2)
    s.label(mid + 8, top_y - 10, "Alg 2 faster \u2192", size=11, color=INK2)
    top = max(4.0, max(max(r, 1 / r) for _, r, _ in pts))
    scale = half / math.log(top)
    for t in (1.5, 2, 3, 4):
        if t > top:
            continue
        for sign in (-1, 1):
            x = mid + sign * scale * math.log(t)
            s.line(x, top_y - 4, x, top_y + row_h * len(pts) - 6)
            s.label(x, top_y + row_h * len(pts) + 10, "%gx" % t, size=10, anchor="middle",
                    color=INK2)
    s.line(mid, top_y - 4, mid, top_y + row_h * len(pts) - 6, color=INK2)
    for i, (fill, r, net) in enumerate(pts):
        y = top_y + i * row_h
        s.label(label_w - 10, y + 15, "%s (fill %.2f)" % (short(net), fill), size=11,
                anchor="end", color=INK2)
        faster = 1 / r if r < 1 else r
        w = scale * math.log(faster)
        if r < 1:
            s.bar(mid - w, y + 3, w, 18, COLOR["cudss"],
                  "%s: cuDSS factorizes %.2f times as fast" % (short(net), faster),
                  round_end="left")
            s.label(mid - w - 6, y + 16, "%.2fx" % faster, size=10, anchor="end", color=INK2)
        else:
            s.bar(mid, y + 3, w, 18, COLOR["alg2"],
                  "%s: Alg 2 factorizes %.2f times as fast" % (short(net), faster),
                  round_end="right")
            s.label(mid + w + 6, y + 16, "%.2fx" % faster, size=10, color=INK2)
    s.label(16, top_y + row_h * len(pts) + 34,
            "Fill = entries in the LU factors per entry of the Jacobian.", size=11, color=INK2)
    s.label(16, top_y + row_h * len(pts) + 50,
            "Sparser factors (top) favour Alg 2; denser factors (bottom) favour cuDSS.",
            size=11, color=INK2)
    s.save(path)


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(MATRIX, "results.jsonl"))
    ap.add_argument("--accuracy", default=os.path.join(MATRIX, "accuracy.csv"))
    ap.add_argument("--out", default=os.path.join(MATRIX, "analysis"))
    ap.add_argument("--timing-check", default=os.path.join(MATRIX, "timing_check", "results.jsonl"),
                    help="GPU-path runs made with the idle-time timers (waits.csv)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load_runs(args.results)
    order = {}
    for (n, r, p), rs in runs.items():
        order.setdefault(n, len(order))
    # smallest network file first, as the campaign runs them
    def file_size(n):
        for f in glob.glob(os.path.join(MATRIX, "networks", n + ".*")):
            return os.path.getsize(f)
        return order[n]
    networks = sorted(order, key=lambda n: (file_size(n), order[n]))

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
                l8 = sum(parts[(n, 8, p)][x] for x in LOOP_PARTS)
                l16 = sum(parts[(n, 16, p)][x] for x in LOOP_PARTS)
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
        # the same split into time outside the case loop (start-up, reading,
        # base case, merge: paid once per study) and the case loop per case
        for p in PATHS:
            k = (n, 16, p)
            if k in runs and d.get("cases"):
                loop = statistics.median(step(x, "CA: Solve and Report Cases", "max")
                                         for x in runs[k])
                row["fixed s_" + p] = wall[k] - loop
                row["loop ms per case_" + p] = 1000 * loop / d["cases"]
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

    # waits.csv: what the GPU paths' start-up and waiting consist of, from
    # runs made with the "CA: GPU Start-up" and "GPU batch: wait for ..."
    # timers (one run per setting)
    wait_rows = []
    if os.path.exists(args.timing_check):
        with open(args.timing_check) as f:
            for line in f:
                c = json.loads(line)
                t = c["timers"]
                if c["returncode"] != 0 or "GPU batch: wait for GPU results" not in t:
                    continue

                def avg(k):
                    return t.get(k, {}).get("avg", 0.0)

                def mx(k):
                    return t.get(k, {}).get("max", 0.0)
                loop = mx("CA: Solve and Report Cases")
                work = sum(avg(k) for k in ("CA case: Apply Outage", "CA case: CPU Solve",
                                            "CA case: Inject GPU Result",
                                            "CA case: Check and Report",
                                            "CA case: Restore Network"))
                idle = avg("GPU batch: wait for GPU results")
                last = avg("GPU batch: wait for other ranks")
                wait_rows.append({
                    "network": name(c["network"]), "ranks": c["ranks"], "path": LABEL[c["path"]],
                    "wall (s)": c["wall_s"],
                    "launch and exit (s)": c["wall_s"] - mx("Total Application"),
                    "GPU start-up (s)": mx("CA: GPU Start-up"),
                    "GPU preparation (s)": mx("GPU batch: prepare (export, classify, plan)"),
                    "case loop (s)": loop, "per-case work (s)": work,
                    "waiting for GPU results (s)": idle,
                    "waiting for other ranks (s)": last,
                    "other loop time (s)": loop - work - idle - last,
                    "waiting for GPU results (% of loop)": 100 * idle / loop if loop else None})
    write_csv(os.path.join(args.out, "waits.csv"), wait_rows)

    # figures
    fig_backend(back_rows, os.path.join(args.out, "fig_backends.svg"))
    fig_cpu_gpu(networks, wall, {n: info.get(n, {}).get("buses") for n in networks},
                args.out)
    fig_accuracy(networks, args.accuracy, os.path.join(args.out, "fig_accuracy.svg"))
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
        if wait_rows:
            f.write("## GPU paths: start-up and waiting, measured\n\n" + md_table(wait_rows)
                    + "\n")
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
