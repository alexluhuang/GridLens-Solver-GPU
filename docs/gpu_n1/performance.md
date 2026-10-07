# Single-Spark measurements

Chapter 10 PERF-1–4 requires whole-study measurements before tuning.
These results use one GB10 Spark, full Polish N-1 (4,198 outcomes),
independent stock GridPACK `b32969b0`, source `d17bdaf4`, text output,
reactive limits enabled, zero shadows and raw voltage starts. Two sequential
trials were run per setting. The large validation queue was paused and image
builds finished before measurement. All 34 trials passed output comparisons.
`performance-trials.jsonl` preserves every timing and setting; the RAW hash
is in `study-evidence.jsonl`. External inputs and large output tables are
not copied into the repository.

Times below are medians of two trials, including startup, planning, CPU
cases, GPU cases, reporting and output. This is more useful for choosing a
deployment setting than kernel time alone. Stock and GPU use the same MPI
rank count in each pair.

| Batch, eight ranks | GPU-path seconds | Cases per second |
|---:|---:|---:|
| 32 | 75.774 | 55.40 |
| 64 | 42.418 | 98.97 |
| 128 | 28.019 | 149.83 |
| 256 | 17.578 | 238.82 |
| 512 | 13.729 | 305.79 |
| 1024 | 14.107 | 297.58 |
| 2048 | 13.300 | 315.63 |

Batch 512 is the smallest tested batch at least 95% as fast as the best
tested batch. Throughput is nearly flat from 512 to 2048, with a small
dip at 1024; it is not a monotonic curve or proof of a physical bandwidth
limit. The validated Algorithm 2 cap remains 2048. Its two best-batch
times were 13.362 and 13.239 s, against 26.916 and 26.168 s for stock.
Median stock/GPU time gives 2.00 times faster execution at eight ranks.

| Setting, eight ranks | GPU-path seconds | Interpretation |
|---|---:|---|
| Algorithm 2, GPU solve, batch 2048 | 13.300 | Faster tested solve placement. |
| Algorithm 2, host solve, batch 2048 | 29.236 | Supports retaining GPU solve as the default. |
| cuDSS, batch 128 | 15.434 | Validated alternative; its cap stays 128. |
| Automatic backend and batch | 16.809 | Selected Algorithm 2 and batch 1024; startup search costs about 2.78 s. |
| Algorithm 2, no backfill, batch 2048 | 14.205 | Slower than keeping slots filled; retain backfill. |
| Algorithm 2, base-case start, batch 2048 | 13.366 | Status/output checks pass; its start deliberately differs from stock. |

The automatic search measures reference factor/solve work. It does not
predict the fastest entire study exactly. Fixed settings avoid its startup
cost when the deployment has already measured a useful setting. These
measurements do not justify hard-coding a Polish-specific choice for other
grids or changing caps on untested platforms.

| Total ranks, batch 2048 | Stock seconds | GPU-path seconds | GPU cases/s |
|---:|---:|---:|---:|
| 1 | 151.549 | 26.095 | 160.87 |
| 2 | 81.875 | 19.831 | 211.69 |
| 4 | 44.256 | 15.985 | 262.62 |
| 8 | 26.313 | 13.369 | 314.01 |
| 16 | 19.085 | 12.686 | 330.91 |

One rank is the accelerator rank; other ranks solve CPU cases and report
completed GPU states. Sixteen total ranks were fastest among the tested
placements. Compared with stock at sixteen ranks, the speedup is 1.50.
The larger one-rank speedup is not a comparison against the fastest stock
placement. These results do not establish an optimum beyond the tested
rank counts, nor multi-Spark scaling.

## Kernel counters

The first ordinary-user Nsight attempt was denied. The approved retry used
`SYS_ADMIN` only in its temporary profiling container, without changing
host driver policy. NVIDIA documents this permission mechanism in its
[counter-access guide](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters).

Nsight Compute 2025.3 successfully captured ten selected launches from a
batch-2048 Polish run. `profile-evidence.json` preserves kernel names,
launch sizes, durations, register counts and achieved occupancy. Captured
mismatch/Jacobian occupancy was 67.62–74.17%; the six early factor levels
achieved 89.79–94.75%. This samples early launches, not every factor level
or the later inactive slots. Replay durations are not production timings.
Clock and cache control were disabled, so automatic clocks/cache contents
could vary; the profiler explicitly warned about that limitation.

The selected memory section did not return DRAM-byte metrics. Physical
DRAM bandwidth against the guide's 273 GB/s reference remains unmeasured.
Runtime phase bandwidth estimates count algorithm traffic and may include
cache hits. They must not be relabeled as memory-controller measurements.
This leaves the bandwidth part of PERF-3 open even though its batch sweep
and occupancy sampling are complete. No kernel optimization is justified
solely by these ten launches.

## Final reporting check

All ten selected repeat trials at `fc1f71a7` passed, including per-case
reported-state checks. `performance-reported-state.jsonl` preserves them.
Other builds and validation jobs began only after these trials finished.

| Configuration | Stock median seconds | GPU median seconds | GPU cases/s |
|---|---:|---:|---:|
| Batch 512, eight ranks, raw start | 26.210 | 13.843 | 303.26 |
| Batch 2048, eight ranks, raw start | 26.936 | 13.329 | 314.96 |
| Batch 2048, sixteen ranks, raw start | 19.101 | 12.455 | 337.05 |
| Automatic settings, eight ranks, raw start | 26.526 | 16.153 | 259.89 |
| Batch 2048, sixteen ranks, base-case start | 18.997 | 12.385 | 338.95 |

At eight ranks, batch-2048 time changed from 13.300 to 13.329 s; batch-512
time changed from 13.729 to 13.843 s. The new reporting loop costs no
material whole-study slowdown in these repeats. These are observed timing
differences, not a separately isolated measurement of the loop itself.
Batch 512 still exceeds 95% of the best eight-rank throughput. The final
sixteen-rank raw-start comparison gives a 1.53 speedup over stock at the
same rank count. Base-case starts may change iteration counts deliberately.

## Whole-study phases with CSV output

Source `d62432ec`, which adds study-phase timer categories to the driver
(`CA: Read Network`, `CA: Base Case`, `CA: Case List and Output Setup`,
`CA: Solve and Report Cases`, `CA: Merge Output Files`). They use GridPACK's
own coarse timer, so the phases appear in the normal timing dump of every
path. Each run was one full branch and generator N-1 study from the RAW file
and configuration to the final `csv_delta` tables: eight ranks, reactive
limits, default settings (base-case warm start on the GPU paths), no shadow
re-solves, one trial each, nothing else running. Phase times are the maximum
over ranks. Every path produced the same case, convergence and delta-row
counts. All rows are in `performance-phases.jsonl`.

| Grid | Path | Batch | Wall s | Read | Base | List | GPU prepare | Solve and report | Merge | GPU busy s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Polish | CPU | – | 26.7 | 0.1 | 0.1 | 0.1 | – | 25.9 | 0.0 | – |
| Polish | Alg2 | 512 | 12.0 | 0.1 | 0.1 | 0.0 | 0.1 | 11.0 | 0.0 | 8.9 |
| Polish | cuDSS | 128 | 13.5 | 0.1 | 0.1 | 0.0 | 0.2 | 12.4 | 0.0 | 11.9 |
| Texas7k | CPU | – | 251.4 | 0.3 | 0.2 | 0.1 | – | 243.9 | 6.5 | – |
| Texas7k | Alg2 | 512 | 191.7 | 0.2 | 0.2 | 0.1 | 0.2 | 181.2 | 9.0 | 157.1 |
| Texas7k | cuDSS | 128 | 146.4 | 0.2 | 0.2 | 0.1 | 0.2 | 136.1 | 8.8 | 60.7 |
| 10k | CPU | – | 570.8 | 0.3 | 0.3 | 0.1 | – | 557.2 | 12.4 | – |
| 10k | Alg2 | 512 | 321.8 | 0.3 | 0.2 | 0.1 | 0.2 | 303.0 | 17.1 | 123.6 |
| 10k | cuDSS | 128 | 309.1 | 0.3 | 0.2 | 0.1 | 0.3 | 289.9 | 17.3 | 125.1 |

Wall time by batch size (seconds; "auto" includes its sweep, shown as GPU
prepare time):

| Grid | Alg2 128 | Alg2 512 | Alg2 2048 | Alg2 auto | cuDSS 32 | cuDSS 64 | cuDSS 128 | cuDSS auto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Polish | 23.1 | 12.0 | 12.2 | 14.5 (1024; 1.8) | 35.4 | 21.0 | 13.5 | 14.2 (128; 0.8) |
| Texas7k | 346.4 | 191.7 | 212.3 | 234.3 (2048; 22.7) | 202.3 | 151.6 | 146.4 | 147.1 (128; 1.6) |
| 10k | 331.4 | 321.8 | 329.3 | 336.8 (2048; 11.2) | 402.7 | 322.6 | 309.1 | 310.3 (128; 1.9) |

Reading, prepare and merge are small. Almost all time is the case loop.
At small batches the GPU limits it (GPU busy time is close to the loop
time). At larger batches the GPU is busy for well under half of the loop on
Texas7k and 10k; the rest is GridPACK applying each case, checking it and
writing about 8,400 (Texas7k) or 9,800 (10k) delta rows per case on the
reporting ranks. The merge is about 40% slower on the GPU paths because
they write the tables in event order (guide R5). Polish writes no delta
rows: every solved case there is `SLACK_OVERLOAD`.

## Solve versus reporting inside the case loop

The driver also times each case's work with GridPACK's coarse timer:
`CA case: Apply Outage` (reset voltages, apply the outage, find islands),
`CA case: CPU Solve` (GridPACK's Newton solve and the driver's second solve
after a reactive-limit change), `CA case: Inject GPU Result`,
`CA case: Check and Report` (slack, voltage and branch checks, convergence
record and table rows), `CA case: Write Table Rows` (the rows alone, a part of
the previous category) and `CA case: Restore Network`. Times below are
averages per rank over eight ranks, in seconds. "Waiting" is the case-loop
time not covered by these categories: on the GPU paths, mostly waiting for
GPU results. Every run in `performance-split.jsonl` used the same setup as
the phase table above; repeated runs agree within about 2%.

| Grid | Output | Path | Wall | Apply | Solve | Inject | Check and report | of which rows | Restore | Waiting | GPU busy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Texas7k | delta | CPU | 252.3 | 7.3 | 127.8 | – | 98.9 | 92.1 | 4.7 | 5.2 | – |
| Texas7k | delta | Alg2 512 | 194.3 | 7.5 | 0.4 | 12.6 | 103.0 | 94.7 | 4.2 | 55.9 | 158.4 |
| Texas7k | delta | cuDSS 128 | 146.2 | 7.5 | 0.4 | 12.5 | 102.7 | 94.5 | 4.1 | 8.5 | 58.6 |
| Texas7k | flat | CPU | 228.4 | 7.4 | 129.7 | – | 78.7 | 71.7 | 4.7 | 3.3 | – |
| Texas7k | flat | Alg2 512 | 184.0 | 7.6 | 0.4 | 12.6 | 82.6 | 74.2 | 4.2 | 69.4 | 158.1 |
| Texas7k | flat | cuDSS 128 | 121.8 | 7.6 | 0.4 | 12.6 | 82.5 | 74.1 | 4.1 | 7.6 | 58.7 |
| 10k | delta | CPU | 565.4 | 19.3 | 302.9 | – | 203.6 | 189.5 | 11.9 | 13.9 | – |
| 10k | delta | Alg2 512 | 322.0 | 20.0 | 1.1 | 32.2 | 211.9 | 194.7 | 10.6 | 27.4 | 123.8 |
| 10k | delta | cuDSS 128 | 306.7 | 19.9 | 1.1 | 31.4 | 211.3 | 194.4 | 10.4 | 13.8 | 124.7 |
| 10k | flat | CPU | 517.1 | 19.7 | 306.1 | – | 162.0 | 147.7 | 12.1 | 9.8 | – |
| 10k | flat | Alg2 512 | 271.5 | 20.3 | 1.1 | 32.2 | 169.6 | 152.4 | 10.9 | 25.2 | 123.9 |
| 10k | flat | cuDSS 128 | 257.0 | 20.3 | 1.1 | 32.6 | 169.6 | 152.2 | 10.6 | 10.8 | 123.1 |

On the CPU path the solve is a little over half of the loop; the rest is
reporting, and writing the per-branch rows is most of the reporting. The
GPU paths remove the solve but keep all reporting work (plus a smaller
injection step), so the loop cannot be shorter than that work divided over
the reporting ranks. With cuDSS the ranks almost never wait, so this
reporting work sets the time. With Alg2 they also wait for the GPU, most on
Texas7k, whose deeper elimination (502 dependency levels) makes Alg2 slower.

GridPACK's own power-flow timers split the CPU solve further. On 10k, of
299 s per rank, building the Jacobian ("Map to Matrix") takes 89 s,
recreating the matrix and vector mappers for each solve 83 s, filling the
mismatch vector 29 s and the linear solve itself (PETSc KLU) 36 s.

`csv_flat` writes the same rows as `csv_delta` (plus the base case) with 15
columns instead of 32: half the bytes, and about 22% less row-writing time.
It saves 24–48 s on the CPU path and 25–50 s on the cuDSS path, so the best
speedup rises from 1.73 to 1.88 on Texas7k and from 1.84 to 2.01 on 10k.

Results reach the reporting ranks one chunk at a time, and a chunk is four
batches. At batch 2048 the first chunk holds 8,192 cases, so the other
ranks wait for it before they can report anything; this is why batch 2048
was not faster than 512 in the batch-size table.

## Rank count and GPU-driving ranks (10k, csv_delta)

Same setup as above at sixteen ranks (`performance-ranks.jsonl`). Seconds;
per-case categories are averages per rank.

| Path, 16 ranks | Wall | Apply | Solve | Inject | Check and report (rows) | Restore | Waiting | GPU busy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | 372.9 | 12.5 | 194.5 | – | 126.0 (116.5) | 6.9 | 8.4 | – |
| cuDSS 128 | 217.7 | 15.4 | 0.8 | 20.6 | 137.5 (125.6) | 6.8 | 15.8 | 122.2 |
| Alg2 512 | 217.6 | 13.7 | 0.8 | 19.5 | 135.1 (123.7) | 6.8 | 22.3 | 121.0 |
| cuDSS 128, ranks 0 and 8 drive the GPU | 229.3 | 12.9 | 0.7 | 18.7 | 134.0 (122.9) | 6.3 | 36.7 | 125.7 / 135.0 |

Doubling the ranks shortened the CPU path by 1.52 times, not 2: the total
work per case rose by about 30%, because the extra ranks run on the
efficiency cores and share memory bandwidth. The GPU paths improve by
1.4–1.5 times and stay 1.71 times faster than the CPU path. Two GPU-driving
ranks on the one GPU were slower: each took about as long for half of the
cases as one rank took for all of them, because the two processes share the
GPU in turns. One cuDSS rank at batch 128 already keeps the GPU busy.

## After the reporting and engine changes (16 ranks, csv_flat)

Source `9937c912`: rows built from numbers (P1), known-topology shortcuts
for GPU cases (P2), and the engine changes of P4 (several warps per long
factor column, CUDA graphs, refilling slots from the next submission, cuDSS
in the planner's order and in deterministic mode). Each run was one full
branch and generator N-1 study from the RAW file to the final `csv_flat`
tables, 16 ranks on every path (the `ca_run.sh` rule for 20 cores),
reactive limits, base-case warm start, batch 512 on both GPU backends, no
shadow re-solves, one trial each. All three paths are this same build; the
CPU path is the build without a `<GPUBatch>` block. Phase times are the
maximum over ranks, per-case categories the average per rank, in seconds.
All rows are in `performance-final.jsonl`.

| Grid | Path | Wall | Loop | Merge | Apply | Solve | Inject | Check and report (rows) | Restore | GPU busy | Occupancy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Polish | CPU | 19.5 | 18.2 | 0.0 | 0.9 | 16.3 | – | 0.0 (0.0) | 0.4 | – | – |
| Polish | Alg2 | 4.3 | 2.5 | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 (0.0) | 0.1 | 2.0 | 37% |
| Polish | cuDSS | 6.6 | 4.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 (0.0) | 0.1 | 4.2 | 37% |
| Texas7k | CPU | 110.2 | 104.6 | 4.2 | 4.7 | 85.0 | – | 10.3 (6.2) | 2.5 | – | – |
| Texas7k | Alg2 | 26.3 | 19.5 | 4.4 | 0.1 | 0.3 | 0.3 | 9.0 (7.0) | 1.3 | 16.4 | 61% |
| Texas7k | cuDSS | 24.9 | 18.2 | 4.3 | 0.1 | 0.3 | 0.3 | 8.7 (6.8) | 1.2 | 14.7 | 61% |
| 10k | CPU | 251.2 | 244.0 | 5.4 | 12.7 | 199.1 | – | 21.0 (12.5) | 6.9 | – | – |
| 10k | Alg2 | 42.2 | 30.8 | 8.7 | 1.1 | 0.6 | 0.8 | 18.6 (14.3) | 3.4 | 17.4 | 72% |
| 10k | cuDSS | 42.6 | 31.2 | 8.6 | 1.0 | 0.7 | 0.7 | 18.4 (14.2) | 3.5 | 27.0 | 72% |

The GPU paths are 4.6 (Polish), 4.4 (Texas7k) and 5.9 (10k) times faster
than the CPU path at the same rank count. Before these changes, the same
16-rank `csv_flat` studies took 167.1 s (Texas7k) and 181.5 s (10k) on Alg2
and 150.3 s (Texas7k) on the CPU path.

What each step did, on the same 16-rank `csv_flat` runs:

| Step | Texas7k Alg2 wall | 10k Alg2 wall | Where the time went |
|---|---:|---:|---|
| Before | 167.1 | 181.5 | Row text formatted, gathered and parsed |
| P1, rows from numbers | 163.2 | 139.9 | GPU busy 155 of 158 s (Texas7k): the GPU limits the loop |
| P2, known-topology reporting | 157.8 | 130.0 | Apply, inject and restore fall from 41.7 s to 5.3 s per rank (10k) |
| P4, several warps per column | 33.6 | 44.6 | Factorization 114.5 → 21.2 s (10k) |
| P4, refilling from the next submission | 27.2 | 39.7 | Occupancy 40% → 72% (10k); GPU 29.6 → 17.6 s |

Each step left every output byte-identical (see `validation.md`).

The GPU no longer limits the 10k study: it is busy 17 s of a 31 s loop.
The loop is now set by GridPACK's reporting on the other ranks (checks and
rows, 18.6 s per rank) and the study by rank 0's merge of the part files
into one 17 GB table (8.7 s). Writing the merged table from all ranks at
once, at offsets computed from the part sizes, would remove most of the
merge; that was not part of this work.

cuDSS's batch cap is now 2048. At batch 512 the two backends give the
same 10k wall time; cuDSS keeps the GPU busy longer there (27.0 against
17.4 s) without changing the wall time, because reporting sets it. On
Texas7k cuDSS was busy 1.7 s less and finished 1.4 s sooner; on Polish Alg2 was
faster. In isolated timings of one factorization and solve at batch 512
on the 10k Jacobian, Alg2 took 110 ms and cuDSS 186 ms.

## Multi-rank file writing, and comparison with stock GridPACK

Source `27042644` writes the `csv_flat`, `csv_delta` and violation tables
from all ranks at once (`ca_parallel_write.hpp`). Each rank finds the runs
of rows of one case in its own part file; the ranks exchange only the run
sizes and compute the same layout; each rank copies its own rows to their
offsets in the final file. The files are byte-identical to the old rank-0
merge (checked on Memphis, Texas7k and 10k against the previous build, and
by `batchpf.unit.reconcile`).

**Setup, the same for every run.** One GB10 DGX Spark. Every build is
`CMAKE_BUILD_TYPE=Release` (`-O3 -DNDEBUG`), compiled in the same container
image (`alh360/gridpack-n1-tools:1.1`) against the same GA, PETSc, Boost
and MPI, and run in that image with `mpiexec --bind-to none -n 16`. Each
run is one full branch and generator N-1 study from the RAW file and
configuration to the final `csv_flat` tables, reactive limits on,
`writeStats=false`, `printCalcFiles=false`, no shadow re-solves, nothing
else running. GPU paths use batch 512 and base-case warm starts.

- **Stock** is the untouched original `b32969b0` build. It has no
  per-step timers, so **timed stock** is `b32969b0` plus
  `reproductions/stock-timer-only.patch` (SHA256 `6611ab61…bf74b`), which
  adds only the coarse-timer categories of `d62432ec` and `e656133e`. Its
  wall times are within 1.5% of stock (10k 334.1 against 330.9 s; Texas7k
  150.8 against 149.1 s). Its tables equal stock's on Memphis. On Texas7k
  and 10k four cases differ; two runs of untouched stock differ in the same
  way (three late Texas7k cases, events 8863-8889): the original build's outage
  cleanup lets one case's controller adjustment reach later cases on the
  same rank, so its results depend on scheduling (`restoration.md`).
- **This build** (`27042644`) is shown on its CPU path (no `<GPUBatch>`
  block) and on the GPU path with Alg2 and with cuDSS.

Times are medians of two trials (stock: one). Phases are the maximum over
ranks; per-case steps are the average per rank of time summed over that
rank's cases. "Other" is the case-loop time not in a per-case step: on the
GPU paths mostly waiting for GPU results, on all paths the wait for the
last rank. All records are in `performance-stock-comparison.jsonl`.

| Step (s) | Stock | This build, CPU | Alg2 | cuDSS |
|---|---:|---:|---:|---:|
| **ACTIVSg10k** (10,000 buses, 15,191 cases) | | | | |
| Read network | 0.60 | 0.56 | 0.55 | 0.55 |
| Base case | 0.36 | 0.20 | 0.20 | 0.20 |
| Case list and output setup | 0.49 | 0.40 | 0.40 | 0.40 |
| Case loop (solve and report) | 323.8 | 244.1 | 29.7 | 31.5 |
| – apply outage | 12.9 | 12.7 | 1.2 | 1.0 |
| – CPU solve | 197.9 | 199.0 | 0.7 | 0.8 |
| – inject GPU result | – | – | 0.7 | 0.7 |
| – check and report | 100.9 | 21.0 | 18.7 | 18.6 |
| – of which table rows | 91.2 | 12.5 | 14.4 | 14.3 |
| – restore network | 6.1 | 6.9 | 3.5 | 3.4 |
| – other (waiting) | 6.0 | 4.4 | 5.0 | 7.0 |
| Merge output files | 8.2 | 5.8 | 4.0 | 4.1 |
| **Wall** | **334.1** (stock 330.9) | **251.7** | **36.5** | **38.2** |
| GPU busy | – | – | 17.1 | 27.2 |
| **Texas7k** (6,717 buses, 8,891 cases) | | | | |
| Read network | 0.42 | 0.39 | 0.39 | 0.39 |
| Base case | 0.25 | 0.16 | 0.14 | 0.14 |
| Case list and output setup | 0.34 | 0.31 | 0.28 | 0.28 |
| Case loop (solve and report) | 145.0 | 105.7 | 19.8 | 18.4 |
| – apply outage | 4.9 | 4.7 | 0.1 | 0.1 |
| – CPU solve | 85.6 | 86.0 | 0.3 | 0.3 |
| – inject GPU result | – | – | 0.3 | 0.3 |
| – check and report | 49.2 | 10.4 | 9.0 | 8.7 |
| – of which table rows | 44.4 | 6.2 | 7.1 | 6.8 |
| – restore network | 2.4 | 2.5 | 1.3 | 1.2 |
| – other (waiting) | 3.0 | 2.1 | 8.7 | 7.8 |
| Merge output files | 4.0 | 2.8 | 2.0 | 2.0 |
| **Wall** | **150.8** (stock 149.1) | **110.0** | **24.1** | **22.7** |
| GPU busy | – | – | 16.5 | 14.8 |
| **Memphis** (993 buses, 1,570 cases) | | | | |
| Read network | 0.08 | 0.07 | 0.07 | 0.07 |
| Base case | 0.04 | 0.02 | 0.02 | 0.02 |
| Case list and output setup | 0.06 | 0.05 | 0.05 | 0.05 |
| Case loop (solve and report) | 3.83 | 2.81 | 2.37 | 2.58 |
| – apply outage | 0.10 | 0.09 | 0.02 | 0.02 |
| – CPU solve | 2.45 | 2.37 | 0.87 | 0.90 |
| – check and report | 1.13 | 0.22 | 0.14 | 0.13 |
| – of which table rows | 1.02 | 0.13 | 0.12 | 0.12 |
| Merge output files | 0.10 | 0.08 | 0.08 | 0.08 |
| **Wall** | **4.9** (stock 4.9) | **3.9** | **4.0** | **4.3** |
| GPU busy | – | – | 0.7 | 1.0 |

Against untouched stock at the same 16 ranks, this build is 9.1 (Alg2) and
8.7 (cuDSS) times faster on 10k and 6.2 and 6.6 times on Texas7k; its
CPU path is 1.3 (10k) and 1.4 (Texas7k) times faster. Memphis is too small for the GPU to help:
its whole study takes 4 s, of which GPU setup and planning are a large
part, and 177 of its 1,569 GPU cases (11%) diverge on the GPU and are
re-solved by GridPACK, which is the CPU solve time on the GPU paths.

Where the time went: the solve itself is the same on both CPU builds; the
GPU replaces it (199 s on 10k). Building the table rows from numbers (P1)
cut row writing from 91 to 12–14 s per rank. The known-topology shortcuts
(P2) cut applying and restoring outages on the GPU path. The merge fell
from 8.2 s (stock, rank 0) to 4.0 s.

The merge is now limited by the disk, not by the ranks. A separate test
with 16 threads copying 17 GB of part files moved 3.3 GB/s (4 threads:
4.2 GB/s); finding the runs took 0.5 s. The parts written during the loop
already fill much of the page cache's dirty-page allowance (20% of
memory), so the final table is written at about the NVMe drive's speed.
In-kernel `copy_file_range` was slower (6.6 against 5.1 s). Going further
would mean not writing the rows twice, which a single ordered CSV file
does not allow; the Parquet format already avoids the large text table.

