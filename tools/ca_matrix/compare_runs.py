#!/usr/bin/env python3
#
#     Copyright (c) 2013 Battelle Memorial Institute
#     Licensed under modified BSD License. A copy of this license can be
#     found in the LICENSE file in the top level directory of this
#     distribution.
#
"""Accuracy of every kept run against the optimized CPU run of the same
network and rank count (inside the container, through run.sh).

  python3 /src/tools/ca_matrix/compare_runs.py

For each run folder kept with run_matrix.py --keep, the result tables are
compared with those of matrix/out/ours-r<ranks>/<network>_cpu: the same
cases, the same convergence status and iterations, and every number within
0.001. Iteration counts are skipped for GPU runs that started from the
solved base case (run_matrix.py --start base_case, the default), since they
start from a different point. Prints PASS or FAIL with the first
differences.
"""

import os
import sys

MATRIX = "/src/matrix"
sys.path.insert(0, "/src/src/applications/modules/batch_pf/test")
from run_ca_test import compare  # noqa: E402


def main():
    out = os.path.join(MATRIX, "out")
    failed = 0
    for group in sorted(os.listdir(out)):
        program, _, ranks = group.rpartition("-r")
        for name in sorted(os.listdir(os.path.join(out, group))):
            run = os.path.join(out, group, name)
            if "-run" in name or not os.path.exists(os.path.join(run, "ca_results_flat.csv")):
                continue
            network, _, label = name.rpartition("_")
            path = label.split("-")[0]   # "alg2-filestart" is the alg2 path
            reference = os.path.join(out, "ours-r" + ranks, network + "_cpu")
            if program == "ours" and path == "cpu":
                continue
            if not os.path.exists(os.path.join(reference, "ca_results_flat.csv")):
                print("%s, %s ranks, %s %s: no optimized CPU run to compare with" % (
                    network, ranks, program, path))
                continue
            # A GPU run that started from the solved base case takes a
            # different number of Newton steps than the CPU run, which starts
            # from the network file; its results are still compared
            with open(os.path.join(run, "input.xml")) as f:
                base_start = path != "cpu" and "<warmStart>raw</warmStart>" not in f.read()
            errors = []
            # CPU-path tables are in the order cases finished; sort them too
            compare(reference, run, 1e-3, errors, output_format="csv_flat",
                    allow_iteration_differences=base_start,
                    b_in_event_order=(path != "cpu"))
            failed += 1 if errors else 0
            print("%s, %s ranks, %s %s: %s%s" % (
                network, ranks, program, label,
                "PASS" if not errors else "FAIL (%d differences)" % len(errors),
                " (iteration counts not compared: started from the base case)"
                if base_start else ""))
            for e in errors[:5]:
                print("    " + e)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
