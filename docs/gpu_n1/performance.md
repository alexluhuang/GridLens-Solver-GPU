# Single-Spark measurements

Chapter 10 PERF-1–4 requires whole-study measurements before tuning.
These results use one GB10 Spark, full Polish N-1 (4,198 outcomes),
independent stock GridPACK `b32969b0`, source `18f8e840`, text output,
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

All ten selected repeat trials at `e49d0084` passed, including per-case
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
