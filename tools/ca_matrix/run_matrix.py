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
import threading
import time

MATRIX = "/src/matrix"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


def from_template(template, path, net):
    """The template input with only the solve path, outputFile and
    networkConfiguration changed. The CPU paths switch the GPU block off
    (stock GridPACK ignores it); the GPU paths switch it on with their backend."""
    text = template
    def put(tag, value, within=None):
        nonlocal text
        pattern = re.compile(r"(<%s>)[^<]*(</%s>)" % (tag, tag))
        region = text
        if within:
            m = re.search(r"<%s>.*?</%s>" % (within, within), text, re.S)
            if not m:
                raise SystemExit("template has no <%s> block" % within)
            region = m.group(0)
        if len(pattern.findall(region)) != 1:
            raise SystemExit("template must have exactly one <%s>%s" % (
                tag, " in <%s>" % within if within else ""))
        changed = pattern.sub(lambda m: m.group(1) + value + m.group(2), region)
        text = text.replace(region, changed) if within else changed
    put("enabled", "on" if PATHS[path] else "off", "GPUBatch")
    if PATHS[path]:
        put("backend", PATHS[path], "GPUBatch")
    put("outputFile", os.path.splitext(net)[0])
    put("networkConfiguration", net)
    return text


class MemoryWatch:
    """Peak memory of this container while a run is going (cgroup counters:
    all memory including file cache, and program memory alone)"""
    def __init__(self):
        self.peak_total = self.peak_anon = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    @staticmethod
    def _read():
        try:
            with open("/sys/fs/cgroup/memory.current") as f:
                total = int(f.read())
            anon = 0
            with open("/sys/fs/cgroup/memory.stat") as f:
                for line in f:
                    if line.startswith("anon "):
                        anon = int(line.split()[1])
            return total, anon
        except OSError:
            return 0, 0

    def _loop(self):
        while not self._stop.is_set():
            total, anon = self._read()
            self.peak_total = max(self.peak_total, total)
            self.peak_anon = max(self.peak_anon, anon)
            self._stop.wait(0.5)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def runnable_tasks():
    """Tasks ready to run on the whole machine, the least of a few quick
    samples (/proc/loadavg is not limited to the container)"""
    counts = []
    for _ in range(5):
        with open("/proc/loadavg") as f:
            counts.append(int(f.read().split()[3].split("/")[0]))
        time.sleep(0.4)
    return min(counts)


def wait_until_quiet(limit_s=600, quiet=3):
    """Wait for other work on the machine to finish; True if it did"""
    start = time.monotonic()
    busy = runnable_tasks()
    if busy <= quiet:
        return True
    print("machine busy (%d tasks running); waiting up to %d s" % (busy, limit_s), flush=True)
    while time.monotonic() - start < limit_s:
        time.sleep(10)
        if runnable_tasks() <= quiet:
            return True
    print("still busy; running anyway, the run is marked busy", flush=True)
    return False


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
    ap.add_argument("--resume", action="store_true",
                    help="skip runs already in results.jsonl that finished, with their "
                         "compact table when --keep")
    ap.add_argument("--template", help="input file to start from (e.g. "
                    "/src/test_runs/input.xml); only the solve path, outputFile and "
                    "networkConfiguration are changed, and --batch and --start are unused")
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
    template = open(args.template).read() if args.template else None
    done = set()
    if args.resume and os.path.exists(MATRIX + "/results.jsonl"):
        with open(MATRIX + "/results.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r["returncode"] == 0:
                    done.add((r["program"], r["network"], r["path"], r["ranks"], r["repeat"]))
    for repeat in range(1, args.repeat + 1):
        for net in args.networks.split(","):
            for path in paths:
                name = "%s_%s" % (os.path.splitext(net)[0], path)
                if PATHS[path] and args.start == "file":
                    name += "-filestart"   # kept apart from base-case-start runs
                work = os.path.join(out_root, name if repeat == 1 else "%s-run%d" % (name, repeat))
                key = (args.program, os.path.splitext(net)[0], path, ranks, repeat)
                if key in done and not (args.keep and repeat == 1 and
                                        not os.path.isdir(os.path.join(work, "flat.parquet"))):
                    print("%s %s %s ranks %d run %d: already done" % (
                        args.program, net, path, ranks, repeat), flush=True)
                    continue
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
                    if template:
                        f.write(from_template(template, path, net))
                    else:
                        f.write(TEMPLATE.format(
                            gpu=block, raw=net, solver="klu", output_format="csv_flat",
                            qlim="true",
                            contingencies="    <FullBranchN1>true</FullBranchN1>\n"
                                          "    <FullGeneratorN1>true</FullGeneratorN1>"))
                quiet = wait_until_quiet()
                start = time.monotonic()
                with MemoryWatch() as memory:
                    run = subprocess.run(["mpiexec", "--bind-to", "none", "-n", str(ranks),
                                          cax, "input.xml"], cwd=work, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, universal_newlines=True)
                wall = time.monotonic() - start
                with open(os.path.join(work, "log.txt"), "w") as f:
                    f.write(run.stdout)
                os.remove(os.path.join(work, net))
                prefix = os.path.splitext(net)[0] if template else "ca_results"
                tables = [f for f in os.listdir(work)
                          if f.startswith(prefix + "_") and f.endswith(".csv")]
                kept = args.keep and repeat == 1 and run.returncode == 0
                if kept:
                    # Keep a compact copy of the large table for compare_runs.py and
                    # delete the text table, which can be over 100 GB
                    import compare_runs
                    compare_runs.parquet_copy(work)
                    os.remove(compare_runs.output_file(work, "_flat.csv"))
                else:
                    for f in tables:
                        if f.endswith("_flat.csv") or f.endswith("_delta.csv"):
                            os.remove(os.path.join(work, f))
                record = {"program": args.program, "network": os.path.splitext(net)[0],
                          "path": path, "ranks": ranks, "repeat": repeat,
                          "batch": args.batch if PATHS[path] else None,
                          "start": (args.start if PATHS[path] else "file"),
                          "returncode": run.returncode, "wall_s": round(wall, 2),
                          "machine_busy": not quiet,
                          "peak_memory_gb": round(memory.peak_total / 1e9, 2),
                          "peak_program_memory_gb": round(memory.peak_anon / 1e9, 2),
                          "template": args.template,
                          "steps": timers(run.stdout), "folder": work}
                with open(MATRIX + "/results.jsonl", "a") as f:
                    f.write(json.dumps(record, sort_keys=True) + "\n")
                print("%s %s %s ranks %d run %d: %s, %.1f s%s" % (
                    args.program, net, path, ranks, repeat,
                    "ok" if run.returncode == 0 else "FAILED", wall,
                    "" if quiet else " (machine was busy)"), flush=True)


if __name__ == "__main__":
    main()
