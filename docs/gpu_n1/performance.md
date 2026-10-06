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

The final outcome-reporting change follows these trials. A small repeated
check of selected configurations must confirm its overhead before treating
the older timings as representative of the final host adapter.
