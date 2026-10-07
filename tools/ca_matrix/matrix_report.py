#!/usr/bin/env python3
#
#     Copyright (c) 2013 Battelle Memorial Institute
#     Licensed under modified BSD License. A copy of this license can be
#     found in the LICENSE file in the top level directory of this
#     distribution.
#
"""Step times of every run in matrix/results.jsonl, each also as a multiple
of the optimized CPU run of the same network and rank count.

  python3 tools/ca_matrix/matrix_report.py [--csv report.csv]

Values are the middle value (median) over repeats. Study steps are the
slowest rank's time; steps inside the case loop are the average per rank of
the time summed over that rank's cases.
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

MATRIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "matrix")
STEPS = [("total", None, None),
         ("read network", "CA: Read Network", "max"),
         ("base case", "CA: Base Case", "max"),
         ("case setup", "CA: Case List and Output Setup", "max"),
         ("all cases", "CA: Solve and Report Cases", "max"),
         ("merge files", "CA: Merge Output Files", "max"),
         ("apply outage", "CA case: Apply Outage", "avg"),
         ("solve", "CA case: CPU Solve", "avg"),
         ("inject GPU result", "CA case: Inject GPU Result", "avg"),
         ("check and report", "CA case: Check and Report", "avg"),
         ("write rows", "CA case: Write Table Rows", "avg"),
         ("restore", "CA case: Restore Network", "avg")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(MATRIX, "results.jsonl"))
    ap.add_argument("--csv", help="also write the table to this CSV file")
    args = ap.parse_args()
    runs = defaultdict(list)
    with open(args.results) as f:
        for line in f:
            r = json.loads(line)
            if r["returncode"] == 0:
                path = r["path"]
                if r.get("start") == "file" and path != "cpu":
                    path += " (file start)"
                runs[(r["network"], r["ranks"], r["program"], path)].append(r)

    def value(rs, step):
        name, cat, kind = step
        if cat is None:
            return statistics.median(r["wall_s"] for r in rs)
        return statistics.median(r["steps"].get(cat, {}).get(kind, 0.0) for r in rs)

    table = []
    for key in sorted(runs):
        network, ranks, program, path = key
        rs = runs[key]
        base = runs.get((network, ranks, "ours", "cpu"))
        row = {"network": network, "ranks": ranks, "program": program, "path": path,
               "runs": len(rs)}
        for step in STEPS:
            v = value(rs, step)
            b = value(base, step) if base else 0.0
            row[step[0] + " s"] = round(v, 2)
            row[step[0] + " vs CPU"] = round(v / b, 3) if b > 0 else ""
        table.append(row)
        print("%s, %d ranks, %s %s (%d runs)" % (network, ranks, program, path, len(rs)))
        for step in STEPS:
            ratio = row[step[0] + " vs CPU"]
            print("    %-18s %9.2f s%s" % (step[0], row[step[0] + " s"],
                                         "   x%.2f of optimized CPU" % ratio if ratio != "" else ""))
    if args.csv and table:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
