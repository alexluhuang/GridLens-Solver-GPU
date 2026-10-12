#!/usr/bin/env python3
#
#     Copyright (c) 2013 Battelle Memorial Institute
#     Licensed under modified BSD License. A copy of this license can be
#     found in the LICENSE file in the top level directory of this
#     distribution.
#
"""GPU-path studies of the public networks at 8, 16 and 20 ranks, once each,
recording every GridPACK timer category, including the GPU start-up and
idle-time timers. Results go to matrix/timing_check/results.jsonl only, so
the matrix's results.jsonl is not touched (delete the old file before a new
check); analyze_matrix.py turns them into waits.csv.

  tools/ca_matrix/run.sh "python3 /src/tools/ca_matrix/timing_check.py [NET.RAW,...]"
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
sys.path.insert(0, "/src/tools/ca_matrix")
import run_matrix as rm  # noqa: E402

OUT = "/src/matrix/timing_check"
os.makedirs(OUT, exist_ok=True)
CAX = rm.PROGRAMS["ours"]
NETS = (sys.argv[1].split(",") if len(sys.argv) > 1 else
        ["MemphisCase2026_Mar7.RAW", "Wisconsin_1664.RAW", "Base_Florida_42GW.RAW",
         "Texas7k_20210804.RAW", "ACTIVSg10k.RAW", "Base_MIOHIN_76GW.RAW"])


def all_timers(log):
    t = {}
    for m in re.finditer(r"Timing statistics for: (.*)\n\s+Average time:\s+([\d.]+)\n"
                         r"\s+Maximum time:\s+([\d.]+)\n\s+Minimum time:\s+([\d.]+)", log):
        t[m.group(1).strip()] = {"avg": float(m.group(2)), "max": float(m.group(3)),
                                 "min": float(m.group(4))}
    return t

def main():
    with open("/src/test_runs/input.xml") as f:
        template = f.read()
    for net in NETS:
        for ranks in (8, 16, 20):
            for path in ("alg2", "cudss"):
                work = "%s/%s_%s_r%d" % (OUT, os.path.splitext(net)[0], path, ranks)
                shutil.rmtree(work, ignore_errors=True)
                os.makedirs(work)
                shutil.copy("/src/matrix/networks/" + net, work)
                with open(work + "/input.xml", "w") as f:
                    f.write(rm.from_template(template, path, net))
                quiet = rm.wait_until_quiet()
                start = time.monotonic()
                run = subprocess.run(["mpiexec", "--bind-to", "none", "-n", str(ranks), CAX,
                                      "input.xml"], cwd=work, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, universal_newlines=True)
                wall = time.monotonic() - start
                with open(work + "/log.txt", "w") as f:
                    f.write(run.stdout)
                os.remove(work + "/" + net)
                for name in os.listdir(work):
                    if name.endswith("_flat.csv") or name.endswith("_delta.csv"):
                        os.remove(os.path.join(work, name))
                rec = {"network": os.path.splitext(net)[0], "path": path, "ranks": ranks,
                       "returncode": run.returncode, "wall_s": round(wall, 2), "busy": not quiet,
                       "timers": all_timers(run.stdout)}
                with open(OUT + "/results.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(net, path, ranks, run.returncode, round(wall, 1), flush=True)


if __name__ == "__main__":
    main()
