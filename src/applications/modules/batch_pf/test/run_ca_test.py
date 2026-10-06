#!/usr/bin/env python3
#
#     Copyright (c) 2013 Battelle Memorial Institute
#     Licensed under modified BSD License. A copy of this license can be
#     found in the LICENSE file in the top level directory of this
#     distribution.
#
"""End-to-end tests of ca.x with and without the GPU batch path.

Each mode runs ca.x on a RAW case with full N-1 and compares against a run
of the same ca.x without a GPUBatch block (stock GridPACK behavior):

  parity       GPUBatch with the given backend and warmStart=raw must give
               the same statuses and iterations, and the same flows,
               voltages and violations within a numeric tolerance (FID-1,
               FID-2). The only allowed difference is the convergence
               record GridPACK writes for cases it never solves (ISLANDED,
               NO_SLACK), which repeats the previous case on that rank.
  no_block     no GPUBatch block: output identical to stock (CONF-2)
  disabled     GPUBatch enabled=off: identical to stock (RT-4)
  no_device    GPUBatch with no visible GPU: the run completes on the CPU
               path with a logged reason and stock output (CONF-3)
  no_plugin    pluginPath pointing to an empty directory: same (CONF-3)
  invalid      an invalid GPUBatch value stops the run at start-up with a
               message naming the key (CONF-4, RT-5)
  required     enabled=on and onUnavailable=error without a usable plugin
               stops the run (guide 6.0)

Prints "No errors detected" on success (GridPACK's test convention).
"""

import argparse
import contextlib
import csv
from decimal import Decimal
import functools
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<Configuration>
  <Contingency_analysis>
    <printCalcFiles>false</printCalcFiles>
    <writeStats>false</writeStats>
{contingencies}
    <qlim>true</qlim>
    <outputFormat>{output_format}</outputFormat>
{gpu}  </Contingency_analysis>
  <Powerflow>
    <networkConfiguration>{raw}</networkConfiguration>
    <maxIteration>50</maxIteration>
    <tolerance>1.0e-6</tolerance>
    <qlim>true</qlim>
    <LinearSolver>
      <PETScOptions>-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type {solver}</PETScOptions>
    </LinearSolver>
  </Powerflow>
</Configuration>
"""

def default_ranks(cores=None):
    """MPI ranks for a study, the same rule as contingency_analysis/ca_run.sh:
    cores - 4 with 8 or more cores, otherwise cores - 2 (at least 1). The
    GPU run and its CPU comparison always use the same count."""
    if cores is None:
        cores = len(os.sched_getaffinity(0))
    return cores - 4 if cores >= 8 else max(1, cores - 2)


FLAT = "ca_results_flat.csv"
FILES = ["ca_results_convergence.csv", "ca_results_delta.csv",
         "ca_results_violations.csv", "ca_results_summary.json",
         "ca_results_contingencies.csv"]


def run(cax, workdir, raw, gpu_lines, solver, env=None, launcher=(),
        execution=None, output_format="csv_delta", contingency_list=None):
    os.makedirs(workdir, exist_ok=True)
    shutil.copy(raw, workdir)
    gpu = ""
    if gpu_lines is not None:
        gpu = "    <GPUBatch>\n" + "".join("      %s\n" % g for g in gpu_lines) + \
              "    </GPUBatch>\n"
        if execution:
            gpu += "    <Execution><acceleratorRanks>%s</acceleratorRanks>" \
                   "<cpuBinding>none</cpuBinding></Execution>\n" % execution
    with open(os.path.join(workdir, "input.xml"), "w") as f:
        contingencies = "    <FullBranchN1>true</FullBranchN1>\n" \
                        "    <FullGeneratorN1>true</FullGeneratorN1>"
        if contingency_list:
            shutil.copy(contingency_list, os.path.join(workdir, "contingencies.xml"))
            contingencies = "    <contingencyList>contingencies.xml</contingencyList>"
        f.write(TEMPLATE.format(gpu=gpu, raw=os.path.basename(raw), solver=solver,
                                output_format=output_format, contingencies=contingencies))
    e = dict(os.environ)
    if env:
        e.update(env)
    start = time.monotonic()
    p = subprocess.run(list(launcher) + [cax, "input.xml"], cwd=workdir, env=e,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=3000)
    with open(os.path.join(workdir, "log.txt"), "w") as f:
        f.write(p.stdout)
    with open(os.path.join(workdir, "timing.json"), "w") as f:
        json.dump({"seconds": time.monotonic() - start, "ranks": launcher,
                   "returncode": p.returncode}, f)
    return p.returncode, p.stdout


def rows(path):
    with open(path) as f:
        return list(csv.reader(f))


@contextlib.contextmanager
def event_rows(path, presorted=False):
    # Stock files arrive in rank order. External sorting bounds memory even
    # for a full study; Python holds only the rows of one event at a time.
    if presorted:
        with open(path) as f:
            reader = csv.reader(f)
            yield next(reader), itertools.groupby(reader, lambda r: int(r[0]))
        return
    process = subprocess.Popen(["sort", "--stable", "--buffer-size=256M", "--parallel=1",
                                "--field-separator=,", "--key=1,1n", path],
                               stdout=subprocess.PIPE, universal_newlines=True,
                               env=dict(os.environ, LC_ALL="C"))
    try:
        reader = csv.reader(process.stdout)
        header = next(reader)
        yield header, itertools.groupby(reader, lambda r: int(r[0]))
        if process.wait() != 0:
            raise RuntimeError("could not sort " + path)
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()


def compare_table(a, b, name, nkey, tol, errors):
    with event_rows(os.path.join(a, name)) as (ha, ga), \
            event_rows(os.path.join(b, name), presorted=True) as (hb, gb):
        if ha != hb:
            errors.append(name + ": header differs")
            return
        worst = 0.0
        for left, right in itertools.zip_longest(ga, gb):
            if left is None or right is None or left[0] != right[0]:
                errors.append(name + ": event rows differ")
                return
            ra, rb = list(left[1]), list(right[1])
            ka = {tuple(r[:nkey]): r for r in ra}
            kb = {tuple(r[:nkey]): r for r in rb}
            if len(ka) != len(ra) or len(kb) != len(rb):
                errors.append("%s: duplicate element rows in event %s" % (name, left[0]))
            if set(ka) != set(kb):
                errors.append("%s: different elements in event %s" % (name, left[0]))
                continue
            for key in ka:
                left_row, right_row = ka[key], kb[key]
                if len(left_row) != len(right_row) or len(left_row) != len(ha):
                    errors.append(name + ": column count differs")
                if left_row == right_row:
                    continue
                for x, y in zip(left_row, right_row):
                    if x == y:
                        continue
                    try:
                        difference = abs(float(x) - float(y))
                        if not math.isfinite(difference):
                            difference = math.inf
                        if not difference <= tol:
                            worst = max(worst, difference)
                    except ValueError:
                        errors.append("%s: text differs at %s" % (name, key))
        if worst > tol:
            errors.append("%s: numbers differ by up to %g" % (name, worst))


def equivalent_voltage_tie(a, b, key, left, right, tol):
    """Both named buses must attain the global extreme in both output tables."""
    candidates = {str(left["bus_id"]), str(right["bus_id"])}
    voltage_tol = min(tol, 1e-6)
    for directory in (a, b):
        found, values = set(), []
        with open(os.path.join(directory, FILES[2])) as table:
            for row in csv.DictReader(table):
                if row["type"] != "voltage":
                    continue
                value = float(row["mva_or_vpu"])
                values.append(value)
                if (row["contingency"] == left["contingency"]
                        and row["element"] in candidates
                        and all(abs(value - item["v_pu"]) <= voltage_tol
                                for item in (left, right))):
                    found.add(row["element"])
        if found != candidates or not values:
            return False
        extreme = min(values) if key == "worst_voltage_low" else max(values)
        if any(abs(extreme - item["v_pu"]) > voltage_tol for item in (left, right)):
            return False
    return True


def compare(a, b, tol, errors, allow_unsolved=True, output_format="csv_delta",
            allow_iteration_differences=False):
    """Compare the outputs of two run directories"""
    # convergence: status and iterations equal (except unsolved cases)
    ca = {r[0]: r for r in rows(os.path.join(a, FILES[0]))[1:]}
    cb = {r[0]: r for r in rows(os.path.join(b, FILES[0]))[1:]}
    if set(ca) != set(cb):
        errors.append("convergence tables list different cases")
    for k in ca:
        if k not in cb:
            continue
        x, y = ca[k], cb[k]
        if x[10] != y[10]:
            errors.append("case %s status %s vs %s" % (k, x[10], y[10]))
        elif x[10] in ("ISLANDED", "NO_SLACK") and allow_unsolved:
            continue
        elif x[4] != y[4] and x[10] != "DIVERGED" and not allow_iteration_differences:
            errors.append("case %s iterations %s vs %s" % (k, x[4], y[4]))
    # tables: same rows, numbers within tol
    tables = [(FILES[2], 4), (FILES[4], 1)]
    if output_format == "csv_delta":
        tables.append((FILES[1], 6))
    if output_format == "csv_flat":
        tables.append((FLAT, 5))
    for name, nkey in tables:
        compare_table(a, b, name, nkey, tol, errors)
    with open(os.path.join(a, FILES[3])) as f, open(os.path.join(b, FILES[3])) as g:
        def same(x, y):
            if isinstance(x, dict) and isinstance(y, dict):
                return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
            if isinstance(x, list) and isinstance(y, list):
                return len(x) == len(y) and all(same(u, v) for u, v in zip(x, y))
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                return abs(x - y) <= tol
            return x == y
        left, right = json.load(f), json.load(g)
        for key in ("worst_voltage_low", "worst_voltage_high"):
            va, vb = left.get(key) or {}, right.get(key) or {}
            if (va.get("bus_id") != vb.get("bus_id")
                    and same({k: v for k, v in va.items() if k != "bus_id"},
                             {k: v for k, v in vb.items() if k != "bus_id"})
                    and equivalent_voltage_tie(a, b, key, va, vb, tol)):
                print("Equivalent worst-voltage tie:", key, va["bus_id"], vb["bus_id"])
                vb["bus_id"] = va["bus_id"]
        worst_a, worst_b = left.get("worst_loading") or {}, right.get("worst_loading") or {}
        if (worst_a.get("contingency") != worst_b.get("contingency")
                and same({k: v for k, v in worst_a.items() if k != "contingency"},
                         {k: v for k, v in worst_b.items() if k != "contingency"})):
            # GridPACK retains the first strict maximum in completion
            # order. Accept a different name only if BOTH selected cases
            # attain that same reported maximum in BOTH violation tables.
            candidates = {worst_a["contingency"], worst_b["contingency"]}
            element = "%s-%s-%s" % (worst_a["from_bus"], worst_a["to_bus"],
                                     worst_a["circuit_id"])
            tied = True
            for directory in (a, b):
                found = set()
                with open(os.path.join(directory, FILES[2])) as table:
                    for row in csv.DictReader(table):
                        if (row["type"] == "branch" and row["element"] == element
                                and row["contingency"] in candidates
                                and abs(float(row["loading_percent"]) -
                                        worst_a["loading_percent"]) <= tol):
                            found.add(row["contingency"])
                tied = tied and found == candidates
            if tied:
                print("Equivalent worst-loading tie:", sorted(candidates), element)
                worst_b["contingency"] = worst_a["contingency"]
        if not same(left, right):
            errors.append("summary JSON differs")


def identical(a, b, errors, output_format="csv_delta"):
    names = [n for n in FILES if n != FILES[1] or output_format == "csv_delta"]
    if output_format == "csv_flat":
        names.append(FLAT)
    for name in names:
        with open(os.path.join(a, name)) as f, open(os.path.join(b, name)) as g:
            if f.read() != g.read():
                errors.append("%s differs from the stock run" % name)


def ordered(workdir, errors, output_format="csv_delta"):
    for name in (FILES[0], FILES[1], FILES[2], FILES[4], FLAT):
        if name == FILES[1] and output_format != "csv_delta":
            continue
        if name == FLAT and output_format != "csv_flat":
            continue
        previous = -1
        with open(os.path.join(workdir, name)) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                event = int(row[0])
                if event < previous:
                    errors.append("%s: rows are not in event order" % name)
                    break
                previous = event


def complete(workdir, expected, errors):
    for name in (FILES[0], FILES[4], "ca_results_gpu_outcomes.csv"):
        indices = [int(r[0]) for r in rows(os.path.join(workdir, name))[1:]]
        start = 0 if name == FILES[4] else 1
        if sorted(indices) != list(range(start, expected + 1)):
            errors.append("%s: a case is missing or repeated" % name)
    for row in rows(os.path.join(workdir, FILES[0]))[1:]:
        if row[10] == "MISSING":
            errors.append("case %s has no outcome" % row[0])


def shadow(workdir, errors, require_sets=False):
    for row in csv.DictReader(open(os.path.join(workdir, "ca_results_gpu_shadow.csv"))):
        if row["cpu_converged"] != row["gpu_converged"]:
            errors.append("case %s: shadow convergence differs" % row["event_idx"])
        for key in ("max_dv_pu", "max_dtheta_rad"):
            if not abs(float(row[key])) <= 1e-6:
                errors.append("case %s: %s exceeds 1e-6" % (row["event_idx"], key))
        if row["pv_buses_cpu"] != row["pv_buses_gpu"] or row["classification_match"] != "1":
            errors.append("case %s: shadow bus types or classification differ" % row["event_idx"])
        if row.get("pv_set_match") == "0" or (require_sets and "pv_set_match" not in row):
            errors.append("case %s: exact PV/PQ set comparison failed or absent" % row["event_idx"])
        if "pq_buses_cpu" in row and row["pq_buses_cpu"] != row["pq_buses_gpu"]:
            errors.append("case %s: shadow PQ counts differ" % row["event_idx"])


def reported_state(workdir, errors):
    with open(os.path.join(workdir, FILES[0])) as stream:
        convergence = {r["event_idx"]: r for r in csv.DictReader(stream)}
    with open(os.path.join(workdir, "ca_results_gpu_outcomes.csv")) as stream:
        outcomes = {r["event_idx"]: r for r in csv.DictReader(stream)}
    fields = {"reported_status", "reported_iterations", "reported_tolerance",
              "final_pv_buses", "final_pq_buses"}
    for event, row in outcomes.items():
        if not fields.issubset(row):
            errors.append("reported case state is absent")
            return
        conv = convergence[event]
        status = row["reported_status"]
        numerical = (status == "NUMERICAL_FAILURE" and conv["status_code"] == "DIVERGED"
                     and row["path"] == "cpu_fallback" and int(row["health_events"]) & 7
                     and row["reported_iterations"] and row["reported_tolerance"]
                     and not math.isfinite(float(row["reported_tolerance"])))
        if status != conv["status_code"] and not numerical:
            errors.append("case %s: reported status differs" % event)
        if min(int(row["final_pv_buses"]), int(row["final_pq_buses"])) < 0:
            errors.append("case %s: negative final bus count" % event)
        if status in ("ISLANDED", "NO_SLACK"):
            if row["reported_iterations"] or row["reported_tolerance"]:
                errors.append("case %s: unsolved case inherited a solve record" % event)
            continue
        if row["reported_iterations"]:
            if row["reported_iterations"] != conv["iterations"]:
                errors.append("case %s: reported iterations differ" % event)
            x, y = row["reported_tolerance"], conv["final_tolerance"]
            # GridPACK reuses a stream whose precision falls from six to four
            # decimals after its first row. Compare the published rounding bins.
            rounding = (0.5 * sum(10.0 ** Decimal(v).as_tuple().exponent for v in (x, y))
                        if x and x != y and all(math.isfinite(float(v)) for v in (x, y)) else 0)
            if not x or (x != y and not abs(float(x) - float(y)) <= rounding):
                errors.append("case %s: reported tolerance differs" % event)
        elif status in ("OK", "SLACK_OVERLOAD"):
            errors.append("case %s: solved case has no reported solve record" % event)
    path = os.path.join(workdir, "ca_results_gpu_shadow.csv")
    if os.path.exists(path):
        with open(path) as stream:
            for row in csv.DictReader(stream):
                outcome = outcomes[row["event_idx"]]
                for kind in ("pv", "pq"):
                    count = outcome["final_%s_buses" % kind]
                    if any(count != row.get("%s_buses_%s" % (kind, side))
                           for side in ("cpu", "gpu")):
                        errors.append("case %s: reported %s count differs from shadow" %
                                      (row["event_idx"], kind))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cax", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--backend", default="alg2")
    ap.add_argument("--solver", default="klu")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--ranks", default="auto",
                    help="MPI ranks for both runs, or auto (see default_ranks)")
    ap.add_argument("--mpiexec", default="mpiexec")
    ap.add_argument("--accelerator-ranks")
    ap.add_argument("--gpu-setting", action="append", default=[])
    ap.add_argument("--stock-cax")
    ap.add_argument("--expect-batch-at-most", type=int)
    ap.add_argument("--require-shadow-sets", action="store_true")
    ap.add_argument("--require-reported-state", action="store_true")
    ap.add_argument("--output-format", choices=("csv_flat", "csv_delta", "text"),
                    default="csv_flat")
    ap.add_argument("--warm-start", choices=("raw", "base_case"))
    ap.add_argument("--shadow-fraction", type=float)
    ap.add_argument("--contingency-list", help="Optional existing XML list for a repeatable sample")
    args = ap.parse_args()
    errors = []
    args.ranks = default_ranks() if args.ranks == "auto" else int(args.ranks)
    if args.ranks < 1:
        ap.error("--ranks must be positive")
    launcher = [args.mpiexec, "--bind-to", "none", "-n", str(args.ranks)] if args.ranks > 1 else []
    if args.output_format == "text" and args.mode == "parity":
        ap.error("--output-format=text omits the branch tables; parity tests compare them")
    if args.shadow_fraction is not None and not 0 <= args.shadow_fraction <= 1:
        ap.error("--shadow-fraction must be between zero and one")
    invoke = functools.partial(run, launcher=launcher, execution=args.accelerator_ranks,
                               output_format=args.output_format, contingency_list=args.contingency_list)
    stock = os.path.join(args.workdir, "stock")
    code, _ = invoke(args.stock_cax or args.cax, stock, args.raw, None, args.solver)
    if code != 0:
        print("stock run failed")
        print("failure detected")
        return 1
    test = os.path.join(args.workdir, args.mode)
    if args.mode in ("parity", "benchmark"):
        # The same initial state makes stock iteration records comparable.
        # A production warm start can be benchmarked explicitly.
        warm_start = args.warm_start or "raw"
        fraction = args.shadow_fraction if args.shadow_fraction is not None else \
                   (1.0 if args.mode == "parity" else 0.0)
        code, out = invoke(args.cax, test, args.raw,
                        ["<enabled>on</enabled>", "<onUnavailable>error</onUnavailable>",
                         "<backend>%s</backend>" % args.backend,
                         "<warmStart>%s</warmStart>" % warm_start,
                         "<shadowFraction>%g</shadowFraction>" % fraction] + args.gpu_setting, args.solver)
        if code != 0:
            errors.append("GPU run failed with exit code %d" % code)
        else:
            compare(stock, test, args.tol, errors, output_format=args.output_format,
                    allow_iteration_differences=warm_start == "base_case")
            ordered(test, errors, args.output_format)
            expected = len(rows(os.path.join(stock, FILES[4]))) - 2
            complete(test, expected, errors)
            if args.require_reported_state:
                reported_state(test, errors)
            if fraction > 0:
                shadow(test, errors, args.require_shadow_sets)
            if args.expect_batch_at_most is not None:
                capacities = re.findall(r"backend \w+ \([^\n]+\), batch size (\d+)", out)
                if not capacities or any(int(b) > args.expect_batch_at_most for b in capacities):
                    errors.append("effective batch exceeds the expected admission limit")
            if fraction > 0 and "shadow validation" not in out:
                errors.append("no shadow validation summary in the log")
            if args.mode == "benchmark":
                with open(os.path.join(stock, "timing.json")) as f:
                    cpu_seconds = json.load(f)["seconds"]
                with open(os.path.join(test, "timing.json")) as f:
                    gpu_seconds = json.load(f)["seconds"]
                result = {"cases": expected, "ranks": args.ranks, "raw": os.path.basename(args.raw),
                          "output_format": args.output_format, "warm_start": warm_start,
                          "shadow_fraction": fraction, "cpu_seconds": cpu_seconds,
                          "gpu_seconds": gpu_seconds, "cases_per_second": expected / gpu_seconds,
                          "speedup": cpu_seconds / gpu_seconds, "fidelity_passed": not errors}
                with open(os.path.join(args.workdir, "benchmark.json"), "w") as f:
                    json.dump(result, f, indent=2)
                print(json.dumps(result, sort_keys=True))
    elif args.mode in ("no_block", "disabled"):
        lines = None if args.mode == "no_block" else ["<enabled>off</enabled>"]
        code, out = invoke(args.cax, test, args.raw, lines, args.solver)
        if code != 0:
            errors.append("run failed")
        else:
            identical(stock, test, errors, args.output_format)
            if "[gpu-batch" in out:
                errors.append("batch path printed messages although not requested")
    elif args.mode in ("no_device", "no_plugin"):
        lines = ["<backend>%s</backend>" % args.backend]
        env = None
        if args.mode == "no_device":
            env = {"CUDA_VISIBLE_DEVICES": ""}
        else:
            empty = os.path.join(args.workdir, "empty_plugins")
            os.makedirs(empty, exist_ok=True)
            lines.append("<pluginPath>%s</pluginPath>" % empty)
        code, out = invoke(args.cax, test, args.raw, lines, args.solver, env)
        if code != 0:
            errors.append("run did not complete without the accelerator")
        else:
            identical(stock, test, errors, args.output_format)
            if "running GridPACK's CPU contingency loop" not in out:
                errors.append("no fallback reason in the log")
    elif args.mode == "invalid":
        code, out = invoke(args.cax, test, args.raw, ["<batchSize>lots</batchSize>"], args.solver)
        if code == 0:
            errors.append("invalid setting did not stop the run")
        if "GPUBatch/batchSize" not in out:
            errors.append("error message does not name the key")
        if os.path.exists(os.path.join(test, FILES[0])):
            errors.append("a partial run wrote results")
    elif args.mode == "required":
        empty = os.path.join(args.workdir, "empty_plugins")
        os.makedirs(empty, exist_ok=True)
        code, out = invoke(args.cax, test, args.raw,
                        ["<enabled>on</enabled>", "<onUnavailable>error</onUnavailable>",
                         "<pluginPath>%s</pluginPath>" % empty], args.solver)
        if code == 0:
            errors.append("required accelerator missing but the run continued")
    else:
        errors.append("unknown mode " + args.mode)
    for e in errors[:20]:
        print("FAILED:", e)
    if errors:
        print("%d failure detected" % len(errors))
        return 1
    print("No errors detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
