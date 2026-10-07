#!/usr/bin/env python3
#
#     Copyright (c) 2013 Battelle Memorial Institute
#     Licensed under modified BSD License. A copy of this license can be
#     found in the LICENSE file in the top level directory of this
#     distribution.
#
"""Run full N-1 contingency studies for the test matrix (inside the
container, through run.sh).

Each run goes from the network file to the final csv_flat tables, with the
same settings for every program and path, and appends one line with its
wall time and step times to matrix/results.jsonl.

  python3 /src/tools/ca_matrix/run_matrix.py --program ours \\
      --networks Texas7k_20210804.RAW,ACTIVSg10k.RAW --paths cpu,alg2,cudss \\
      --ranks 8 --repeat 2 --keep

--program stock runs matrix/builds/stock (paths: cpu only); --program ours
runs matrix/builds/optimized. Networks are read from matrix/networks/.
Tables of the first repeat are kept with --keep (for compare_runs.py);
otherwise they are deleted after the run to save disk space. GPU paths
start each case from the solved base case (production default); --start
file makes them start from the network file like the CPU paths.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

MATRIX = "/src/matrix"
sys.path.insert(0, "/src/src/applications/modules/batch_pf/test")
from run_ca_test import TEMPLATE, default_ranks  # noqa: E402

PROGRAMS = {
    "stock": MATRIX + "/builds/stock/applications/contingency_analysis/ca.x",
    "ours": MATRIX + "/builds/optimized/applications/contingency_analysis/ca.x",
}
PATHS = {"cpu": None, "alg2": "alg2", "cudss": "cudss"}
STEPS = ["CA: Read Network", "CA: Base Case", "CA: Case List and Output Setup",
         "CA: Solve and Report Cases", "CA: Merge Output Files", "CA case: Apply Outage",
         "CA case: CPU Solve", "CA case: Inject GPU Result", "CA case: Check and Report",
         "CA case: Write Table Rows", "CA case: Restore Network"]


def timers(log):
    found = {}
    for step in STEPS:
        m = re.search(re.escape("Timing statistics for: " + step) +
                      r"\s+Average time:\s+([\d.]+)\s+Maximum time:\s+([\d.]+)", log)
        if m:
            found[step] = {"avg": float(m.group(1)), "max": float(m.group(2))}
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", choices=sorted(PROGRAMS), required=True)
    ap.add_argument("--networks", required=True, help="comma-separated file names")
    ap.add_argument("--paths", default="cpu", help="comma-separated: cpu, alg2, cudss")
    ap.add_argument("--ranks", type=int, default=0, help="MPI ranks (default: cores rule)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--batch", type=int, default=512, help="GPU batch size")
    ap.add_argument("--keep", action="store_true", help="keep the first repeat's tables")
    ap.add_argument("--start", choices=("base_case", "file"), default="base_case",
                    help="GPU paths: start each case from the solved base case (production "
                         "default) or, like the CPU paths, from the network file's voltages "
                         "(then iteration counts can be compared too)")
    args = ap.parse_args()
    ranks = args.ranks or default_ranks()
    cax = PROGRAMS[args.program]
    paths = args.paths.split(",")
    if args.program == "stock" and paths != ["cpu"]:
        ap.error("stock GridPACK has only the cpu path")
    out_root = "%s/out/%s-r%d" % (MATRIX, args.program, ranks)
    for repeat in range(1, args.repeat + 1):
        for net in args.networks.split(","):
            for path in paths:
                name = "%s_%s" % (os.path.splitext(net)[0], path)
                if PATHS[path] and args.start == "file":
                    name += "-filestart"   # kept apart from base-case-start runs
                work = os.path.join(out_root, name if repeat == 1 else "%s-run%d" % (name, repeat))
                shutil.rmtree(work, ignore_errors=True)
                os.makedirs(work)
                shutil.copy(os.path.join(MATRIX, "networks", net), work)
                block = ""
                if PATHS[path]:
                    gpu = ["<enabled>on</enabled>", "<onUnavailable>error</onUnavailable>",
                           "<backend>%s</backend>" % PATHS[path],
                           "<batchSize>%d</batchSize>" % args.batch,
                           "<shadowFraction>0</shadowFraction>",
                           "<warmStart>%s</warmStart>" % ("raw" if args.start == "file"
                                                          else "base_case")]
                    block = ("    <GPUBatch>\n" + "".join("      %s\n" % g for g in gpu) +
                             "    </GPUBatch>\n")
                with open(os.path.join(work, "input.xml"), "w") as f:
                    f.write(TEMPLATE.format(
                        gpu=block, raw=net, solver="klu", output_format="csv_flat",
                        contingencies="    <FullBranchN1>true</FullBranchN1>\n"
                                      "    <FullGeneratorN1>true</FullGeneratorN1>"))
                start = time.monotonic()
                run = subprocess.run(["mpiexec", "--bind-to", "none", "-n", str(ranks), cax,
                                      "input.xml"], cwd=work, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, universal_newlines=True)
                wall = time.monotonic() - start
                with open(os.path.join(work, "log.txt"), "w") as f:
                    f.write(run.stdout)
                os.remove(os.path.join(work, net))
                if not (args.keep and repeat == 1):
                    for f in os.listdir(work):
                        if f.startswith("ca_results") and f.endswith(".csv"):
                            os.remove(os.path.join(work, f))
                record = {"program": args.program, "network": os.path.splitext(net)[0],
                          "path": path, "ranks": ranks, "repeat": repeat,
                          "batch": args.batch if PATHS[path] else None,
                          "start": (args.start if PATHS[path] else "file"),
                          "returncode": run.returncode, "wall_s": round(wall, 2),
                          "steps": timers(run.stdout), "folder": work}
                with open(MATRIX + "/results.jsonl", "a") as f:
                    f.write(json.dumps(record, sort_keys=True) + "\n")
                print("%s %s %s ranks %d run %d: %s, %.1f s" % (
                    args.program, net, path, ranks, repeat,
                    "ok" if run.returncode == 0 else "FAILED", wall), flush=True)


if __name__ == "__main__":
    main()
