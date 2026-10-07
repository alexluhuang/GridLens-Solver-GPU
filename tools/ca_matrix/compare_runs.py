#!/usr/bin/env python3
#
#     Copyright (c) 2013 Battelle Memorial Institute
#     Licensed under modified BSD License. A copy of this license can be
#     found in the LICENSE file in the top level directory of this
#     distribution.
#
"""Accuracy of every kept run against the optimized CPU run of the same
network and rank count (inside the container, through run.sh).

  python3 /src/tools/ca_matrix/compare_runs.py [--engine gpu|cpu] [--rows]

The csv_flat tables of the two runs are matched row by row on (case,
from bus, to bus, circuit). For each variable below, over all matched rows:

  utilization      loading_percent        complex power   mva_from
  real power       p_from_mw              reactive power  q_from_mvar
  bus voltage      v_from_pu and v_to_pu  bus angle       ang_from_deg and ang_to_deg

it reports the mean and largest absolute difference, the mean and largest
percentage difference (|run - CPU| / |CPU| x 100), and the percentage of
rows that differ from the CPU value by more than 1%. Rows whose CPU value is
exactly zero have no percentage difference; they are counted separately,
and counted as more than 1% off unless the run's value is zero too. Rows
present in only one of the two runs (a case solved by one path only) are
counted. Solver step counts are not compared: the GPU paths take different
steps by design.

Values are printed to four decimal places and written to
matrix/accuracy.csv. With --rows, every matched row's values and
differences are also written to matrix/accuracy_rows/<run>/.

Memory stays bounded however large the tables are. Each table is first
converted to Parquet (ca_results_flat.parquet/ next to it, reused later,
and kept when the CSV tables are deleted), split into groups of cases of a
few million rows. The groups are then compared one at a time, with Polars'
multi-threaded CPU engine (default) or on the GPU with Polars' cuDF engine,
its memory capped at --gpu-memory-gb. On a GPU that shares memory with the
system, as on the DGX Spark, running out of GPU memory takes the whole
machine down, so the cap matters. On the DGX Spark, comparing two
149-million-row ACTIVSg10k tables took 9.2 s on the CPU engine and 12.3 s
on the GPU engine once converted; converting a 17 GB table takes about 21 s.
"""

import argparse
import csv
import glob
import os
import shutil
import sys
import time

import polars as pl

MATRIX = "/src/matrix"
FLAT = "ca_results_flat.csv"
CACHE = "ca_results_flat.parquet"
ROWS_PER_GROUP = 8_000_000
KEYS = ["event_idx", "from_bus", "to_bus", "circuit_id"]
SCHEMA = {"event_idx": pl.Int32, "contingency": pl.Utf8, "from_bus": pl.Int32,
          "to_bus": pl.Int32, "circuit_id": pl.Utf8, "p_from_mw": pl.Float64,
          "q_from_mvar": pl.Float64, "mva_from": pl.Float64, "rate_mva": pl.Float64,
          "loading_percent": pl.Float64, "viol": pl.Int32, "v_from_pu": pl.Float64,
          "v_to_pu": pl.Float64, "ang_from_deg": pl.Float64, "ang_to_deg": pl.Float64}
# (name, csv_flat columns): two columns are pooled (both ends of a branch)
VARIABLES = [("utilization (%)", ["loading_percent"]),
             ("complex power (MVA)", ["mva_from"]),
             ("real power (MW)", ["p_from_mw"]),
             ("reactive power (MVAr)", ["q_from_mvar"]),
             ("bus voltage (pu)", ["v_from_pu", "v_to_pu"]),
             ("bus angle (deg)", ["ang_from_deg", "ang_to_deg"])]
COLUMNS = sorted({c for _, cols in VARIABLES for c in cols})


def cases_per_group(reference):
    """Group size in cases, from the reference table's rows per case"""
    path = os.path.join(reference, FLAT)
    cached = glob.glob(os.path.join(reference, CACHE, "group_size=*"))
    if cached:
        return int(cached[0].rsplit("=", 1)[1])
    with open(os.path.join(reference, "ca_results_contingencies.csv")) as f:
        cases = max(1, sum(1 for _ in f) - 1)
    with open(path) as f:
        f.readline()
        sample = [len(f.readline()) for _ in range(1000)]
    rows = os.path.getsize(path) / max(1.0, sum(sample) / max(1, len(sample)))
    return max(1, int(ROWS_PER_GROUP / max(1.0, rows / cases)))


def parquet_copy(run, group_size):
    """Parquet copy of a run's table, split into groups of cases"""
    cache = os.path.join(run, CACHE)
    marker = os.path.join(cache, "group_size=%d" % group_size)
    if os.path.exists(marker):
        return cache
    shutil.rmtree(cache, ignore_errors=True)
    (pl.scan_csv(os.path.join(run, FLAT), schema=SCHEMA, quote_char='"')
     .select(KEYS + COLUMNS)
     .with_columns((pl.col("event_idx") // group_size).alias("group"))
     .sink_parquet(pl.PartitionBy(cache, key="group", include_key=False),
                   mkdir=True, engine="streaming"))
    open(marker, "w").close()
    return cache


def groups(cache):
    return {int(d.rsplit("=", 1)[1]): d for d in glob.glob(os.path.join(cache, "group=*"))}


def gpu_engine(limit_gb):
    import rmm
    mr = rmm.mr.LimitingResourceAdaptor(rmm.mr.CudaAsyncMemoryResource(),
                                        allocation_limit=int(limit_gb * 1e9))
    return pl.GPUEngine(memory_resource=mr, raise_on_fail=True)


def group_stats(ref_dir, run_dir, engine, rows_out):
    """Sums, maxima and counts of one group of cases"""
    empty = pl.LazyFrame(schema={k: SCHEMA[k] for k in KEYS + COLUMNS})
    def scan(d, suffix):
        lf = pl.scan_parquet(os.path.join(d, "*.parquet")) if d else empty
        return lf.rename({c: c + suffix for c in COLUMNS})
    ref, new = scan(ref_dir, "_cpu"), scan(run_dir, "_run")
    # One full join finds matched rows and the rows of either run alone
    both = ref.join(new, on=KEYS, how="full", coalesce=True)
    in_cpu = pl.col(COLUMNS[0] + "_cpu").is_not_null()
    in_run = pl.col(COLUMNS[0] + "_run").is_not_null()
    matched = in_cpu & in_run
    joined = both.filter(matched)
    # A row of one run alone has empty values on the other side; sums and
    # maxima skip empty values, and the counts below are limited to matches
    aggs = [matched.cast(pl.Int64).sum().alias("rows"),
            (in_cpu & ~in_run).cast(pl.Int64).sum().alias("only_cpu"),
            (in_run & ~in_cpu).cast(pl.Int64).sum().alias("only_run")]
    for c in COLUMNS:
        r, t = pl.col(c + "_cpu"), pl.col(c + "_run")
        absd = (t - r).abs()
        nonzero = r != 0.0
        pct = pl.when(nonzero).then(absd / r.abs() * 100.0)
        over = pl.when(matched).then(pl.when(nonzero).then(pct > 1.0).otherwise(t != 0.0))
        aggs += [absd.sum().alias(c + ":abs_sum"), absd.max().alias(c + ":abs_max"),
                 pct.sum().alias(c + ":pct_sum"), pct.max().alias(c + ":pct_max"),
                 over.cast(pl.Int64).sum().alias(c + ":over"),
                 (~nonzero).cast(pl.Int64).sum().alias(c + ":zero")]
    stats = both.select(aggs).collect(engine=engine).row(0, named=True)
    if rows_out:
        exprs = list(KEYS)
        for c in COLUMNS:
            r, t = pl.col(c + "_cpu"), pl.col(c + "_run")
            exprs += [r, t, (t - r).abs().alias(c + "_abs_diff"),
                      pl.when(r != 0.0).then((t - r).abs() / r.abs() * 100.0)
                      .alias(c + "_pct_diff")]
        joined.select(exprs).collect(engine=engine).write_parquet(rows_out)
    return stats


def compare(reference, run, engine, rows_dir):
    size = cases_per_group(reference)
    ref_groups = groups(parquet_copy(reference, size))
    run_groups = groups(parquet_copy(run, size))
    total = {}
    if rows_dir:
        os.makedirs(rows_dir, exist_ok=True)
    for g in sorted(set(ref_groups) | set(run_groups)):
        rows_out = os.path.join(rows_dir, "group=%d.parquet" % g) if rows_dir else None
        s = group_stats(ref_groups.get(g), run_groups.get(g), engine, rows_out)
        for k, v in s.items():
            v = 0 if v is None else v
            if k.endswith(":abs_max") or k.endswith(":pct_max"):
                total[k] = max(total.get(k, 0.0), v)
            else:
                total[k] = total.get(k, 0) + v
    n = total.get("rows", 0)
    result = []
    for name, cols in VARIABLES:
        values = n * len(cols)
        zero = sum(total.get(c + ":zero", 0) for c in cols)
        pct_rows = values - zero
        result.append({
            "variable": name,
            "rows compared": values,
            "rows only in CPU run": total.get("only_cpu", 0) * len(cols),
            "rows only in this run": total.get("only_run", 0) * len(cols),
            "mean abs diff": (sum(total.get(c + ":abs_sum", 0.0) for c in cols) / values
                              if values else 0.0),
            "max abs diff": max(total.get(c + ":abs_max", 0.0) for c in cols),
            "mean % diff": (sum(total.get(c + ":pct_sum", 0.0) for c in cols) / pct_rows
                            if pct_rows else 0.0),
            "max % diff": max(total.get(c + ":pct_max", 0.0) for c in cols),
            "% rows over 1%": (100.0 * sum(total.get(c + ":over", 0) for c in cols) / values
                               if values else 0.0),
            "rows with CPU value 0": zero})
    return result


def has_table(run):
    return os.path.exists(os.path.join(run, FLAT)) or os.path.isdir(os.path.join(run, CACHE))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=("cpu", "gpu"), default="cpu",
                    help="cpu (default): Polars multi-threaded; gpu: Polars on cuDF")
    ap.add_argument("--gpu-memory-gb", type=float, default=16.0,
                    help="largest GPU memory the comparison may use (default 16)")
    ap.add_argument("--rows", action="store_true",
                    help="also write every matched row's differences (large)")
    args = ap.parse_args()
    engine = gpu_engine(args.gpu_memory_gb) if args.engine == "gpu" else "streaming"
    out = os.path.join(MATRIX, "out")
    report = []
    for group in sorted(os.listdir(out)):
        program, _, ranks = group.rpartition("-r")
        for name in sorted(os.listdir(os.path.join(out, group))):
            run = os.path.join(out, group, name)
            if "-run" in name or not has_table(run):
                continue
            network, _, label = name.rpartition("_")
            if program == "ours" and label == "cpu":
                continue
            reference = os.path.join(out, "ours-r" + ranks, network + "_cpu")
            if not has_table(reference):
                print("%s, %s ranks, %s %s: no optimized CPU run to compare with" % (
                    network, ranks, program, label))
                continue
            start = time.monotonic()
            rows_dir = (os.path.join(MATRIX, "accuracy_rows", "%s-%s" % (group, name))
                        if args.rows else None)
            result = compare(reference, run, engine, rows_dir)
            print("%s, %s ranks, %s %s vs optimized CPU (%.4f s to compare)" % (
                network, ranks, program, label, time.monotonic() - start))
            print("    %-22s %14s %14s %14s %14s %14s" % (
                "variable", "mean abs diff", "max abs diff", "mean % diff", "max % diff",
                "% rows > 1%"))
            for r in result:
                print("    %-22s %14.4f %14.4f %14.4f %14.4f %14.4f" % (
                    r["variable"], r["mean abs diff"], r["max abs diff"], r["mean % diff"],
                    r["max % diff"], r["% rows over 1%"]))
            first = result[0]
            print("    branch rows compared %d; only in CPU run %d; only in this run %d" % (
                first["rows compared"], first["rows only in CPU run"],
                first["rows only in this run"]))
            for r in result:
                report.append(dict({"network": network, "ranks": int(ranks), "program": program,
                                    "path": label}, **r))
    if report:
        with open(os.path.join(MATRIX, "accuracy.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(report[0]))
            w.writeheader()
            for r in report:
                w.writerow({k: ("%.4f" % v if isinstance(v, float) else v) for k, v in r.items()})
        print("written to matrix/accuracy.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
