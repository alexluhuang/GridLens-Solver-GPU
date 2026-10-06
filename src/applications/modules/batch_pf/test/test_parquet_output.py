#!/usr/bin/env python3
"""outputFormat=parquet must hold exactly csv_delta's rows.

Runs the same study twice, with csv_delta and with parquet, on one rank (so
both runs process cases in the same order). The branches and flows tables,
joined on branch_id and with the contingency table for names and types, must
print exactly csv_delta's text with csv_delta's precisions; the delta columns
are differences of stored columns. Every other output file must be identical
between the two runs. Reading uses pyarrow, a separate implementation from
the writer. Exit code 77 (skipped) when pyarrow is not installed or ca.x
was built without Parquet support.
"""

import argparse
import csv
import os
import sys

from run_ca_test import run

try:
    import pyarrow.parquet as pq
except ImportError:
    print("pyarrow is not installed; skipped")
    sys.exit(77)

# csv_delta column -> (source, digits); None = integer or text
DELTA = [("event_idx", None), ("contingency", None), ("type", None), ("from_bus", None),
         ("to_bus", None), ("ckt", None), ("base_kv_from", 2), ("base_kv_to", 2),
         ("area_from", None), ("area_to", None), ("base_rate_mva", 4), ("cont_rate_mva", 4),
         ("base_p_mw", 4), ("cont_p_mw", 4), ("base_q_mvar", 4), ("cont_q_mvar", 4),
         ("base_mva", 4), ("cont_mva", 4), ("base_loading_pct", 2), ("cont_loading_pct", 2),
         ("v_from_base", 6), ("v_from_cont", 6), ("v_to_base", 6), ("v_to_cont", 6),
         ("ang_from_base", 4), ("ang_from_cont", 4), ("ang_to_base", 4), ("ang_to_cont", 4),
         ("d_v_base", 6), ("d_v_cont", 6), ("d_angle_base", 4), ("d_angle_cont", 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cax", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--solver", default="klu")
    ap.add_argument("--gpu-setting", action="append", default=None,
                    help="GPUBatch element; omit for the CPU path")
    args = ap.parse_args()
    errors = []
    dirs = {}
    for fmt in ("csv_delta", "parquet"):
        d = os.path.join(args.workdir, fmt)
        code, out = run(args.cax, d, args.raw, args.gpu_setting, args.solver, output_format=fmt)
        if code and "needs Parquet support" in out:
            print("ca.x was built without Parquet support; skipped")
            return 77
        if code:
            print("FAILED: %s run exited with %d" % (fmt, code))
            print("1 failure detected")
            return 1
        dirs[fmt] = d
    delta, par = dirs["csv_delta"], dirs["parquet"]

    # Every file both modes write must be identical
    for name in sorted(os.listdir(delta)):
        if not name.startswith("ca_results_") or name == "ca_results_delta.csv":
            continue
        if os.path.isdir(os.path.join(delta, name)):
            continue
        with open(os.path.join(delta, name), "rb") as a, open(os.path.join(par, name), "rb") as b:
            if a.read() != b.read():
                errors.append("%s differs between csv_delta and parquet" % name)

    # Rebuild csv_delta's text from the two tables
    branches = pq.read_table(os.path.join(par, "ca_results_branches.parquet")).to_pylist()
    flows_dir = os.path.join(par, "ca_results_flows")
    flows = []
    for name in sorted(os.listdir(flows_dir)):   # name order is event order
        flows.extend(pq.read_table(os.path.join(flows_dir, name)).to_pylist())
    by_id = {b["branch_id"]: b for b in branches}
    names = {}
    with open(os.path.join(par, "ca_results_contingencies.csv")) as f:
        for r in csv.DictReader(f):
            names[int(r["event_idx"])] = (r["contingency"].rstrip(), r["type"])
    previous = -1
    rebuilt = []
    for row in flows:
        if row["event_idx"] < previous:
            errors.append("flows rows are not in event order")
            break
        previous = row["event_idx"]
        b = by_id[row["branch_id"]]
        v = dict(b)
        v.update(row)
        v["contingency"], v["type"] = names[row["event_idx"]]
        v["d_v_base"] = b["v_from_base"] - b["v_to_base"]
        v["d_v_cont"] = row["v_from_cont"] - row["v_to_cont"]
        v["d_angle_base"] = b["ang_from_base"] - b["ang_to_base"]
        v["d_angle_cont"] = row["ang_from_cont"] - row["ang_to_cont"]
        rebuilt.append(",".join(str(v[c]) if digits is None else "%.*f" % (digits, v[c])
                                for c, digits in DELTA) + "\n")
    with open(os.path.join(delta, "ca_results_delta.csv")) as f:
        header = f.readline()
        expected = f.readlines()
    if header.rstrip("\n").split(",") != [c for c, _ in DELTA]:
        errors.append("csv_delta header changed; update this test")
    if len(rebuilt) != len(expected):
        errors.append("parquet has %d flows rows, csv_delta %d" % (len(rebuilt), len(expected)))
    mismatched = sum(1 for x, y in zip(rebuilt, expected) if x != y)
    if mismatched:
        first = next(i for i, (x, y) in enumerate(zip(rebuilt, expected)) if x != y)
        errors.append("%d rows differ, first:\n  parquet   %s  csv_delta %s"
                      % (mismatched, rebuilt[first], expected[first]))
    print("parquet: %d branches, %d flows rows; csv_delta: %d rows"
          % (len(branches), len(flows), len(expected)))
    for e in errors:
        print("FAILED:", e)
    print("%d failure detected" % len(errors) if errors else "No errors detected")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
