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
    <FullBranchN1>true</FullBranchN1>
    <FullGeneratorN1>true</FullGeneratorN1>
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

FILES = ["ca_results_convergence.csv", "ca_results_delta.csv",
         "ca_results_violations.csv", "ca_results_summary.json",
         "ca_results_contingencies.csv"]


def run(cax, workdir, raw, gpu_lines, solver, env=None, launcher=(),
        execution=None, output_format="csv_delta"):
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
        f.write(TEMPLATE.format(gpu=gpu, raw=os.path.basename(raw), solver=solver,
                                output_format=output_format))
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
def event_rows(path):
    # Stock files arrive in rank order. External sorting bounds memory even
    # for a full study; Python holds only the rows of one event at a time.
    process = subprocess.Popen(["sort", "--stable", "--field-separator=,", "--key=1,1n", path],
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
            event_rows(os.path.join(b, name)) as (hb, gb):
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
                if len(ka[key]) != len(kb[key]) or len(ka[key]) != len(ha):
                    errors.append(name + ": column count differs")
                for x, y in zip(ka[key], kb[key]):
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


def compare(a, b, tol, errors, allow_unsolved=True):
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
        elif x[4] != y[4] and x[10] != "DIVERGED":
            errors.append("case %s iterations %s vs %s" % (k, x[4], y[4]))
    # tables: same rows, numbers within tol
    for name, nkey in ((FILES[1], 6), (FILES[2], 4), (FILES[4], 1)):
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
        if not same(json.load(f), json.load(g)):
            errors.append("summary JSON differs")


def identical(a, b, errors):
    for name in FILES:
        with open(os.path.join(a, name)) as f, open(os.path.join(b, name)) as g:
            if f.read() != g.read():
                errors.append("%s differs from the stock run" % name)


def ordered(workdir, errors):
    for name in (FILES[0], FILES[1], FILES[2], FILES[4]):
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


def shadow(workdir, errors):
    for row in csv.DictReader(open(os.path.join(workdir, "ca_results_gpu_shadow.csv"))):
        if row["cpu_converged"] != row["gpu_converged"]:
            errors.append("case %s: shadow convergence differs" % row["event_idx"])
        for key in ("max_dv_pu", "max_dtheta_rad"):
            if not abs(float(row[key])) <= 1e-6:
                errors.append("case %s: %s exceeds 1e-6" % (row["event_idx"], key))
        if row["pv_buses_cpu"] != row["pv_buses_gpu"] or row["classification_match"] != "1":
            errors.append("case %s: shadow bus types or classification differ" % row["event_idx"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cax", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--backend", default="alg2")
    ap.add_argument("--solver", default="klu")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--ranks", type=int, default=1)
    ap.add_argument("--mpiexec", default="mpiexec")
    ap.add_argument("--accelerator-ranks")
    ap.add_argument("--gpu-setting", action="append", default=[])
    ap.add_argument("--stock-cax")
    ap.add_argument("--expect-batch-at-most", type=int)
    args = ap.parse_args()
    errors = []
    if args.ranks < 1:
        ap.error("--ranks must be positive")
    launcher = [args.mpiexec, "--bind-to", "none", "-n", str(args.ranks)] if args.ranks > 1 else []
    invoke = functools.partial(run, launcher=launcher, execution=args.accelerator_ranks)
    stock = os.path.join(args.workdir, "stock")
    code, _ = invoke(args.stock_cax or args.cax, stock, args.raw, None, args.solver)
    if code != 0:
        print("stock run failed")
        print("failure detected")
        return 1
    test = os.path.join(args.workdir, args.mode)
    if args.mode == "parity":
        code, out = invoke(args.cax, test, args.raw,
                        ["<enabled>on</enabled>", "<onUnavailable>error</onUnavailable>",
                         "<backend>%s</backend>" % args.backend, "<warmStart>raw</warmStart>",
                         "<shadowFraction>1.0</shadowFraction>"] + args.gpu_setting, args.solver)
        if code != 0:
            errors.append("GPU run failed with exit code %d" % code)
        else:
            compare(stock, test, args.tol, errors)
            ordered(test, errors)
            expected = len(rows(os.path.join(stock, FILES[4]))) - 2
            complete(test, expected, errors)
            shadow(test, errors)
            if args.expect_batch_at_most is not None:
                capacities = re.findall(r"backend \w+ \([^\n]+\), batch size (\d+)", out)
                if not capacities or any(int(b) > args.expect_batch_at_most for b in capacities):
                    errors.append("effective batch exceeds the expected admission limit")
            if "shadow validation" not in out:
                errors.append("no shadow validation summary in the log")
    elif args.mode in ("no_block", "disabled"):
        lines = None if args.mode == "no_block" else ["<enabled>off</enabled>"]
        code, out = invoke(args.cax, test, args.raw, lines, args.solver)
        if code != 0:
            errors.append("run failed")
        else:
            identical(stock, test, errors)
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
            identical(stock, test, errors)
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
