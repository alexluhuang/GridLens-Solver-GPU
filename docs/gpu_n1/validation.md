# Validation record

This records evidence for Chapter 10 of the supplied architecture guide.
The implementation is not release-validated in its entirety. The live
machine-readable status is `validation-status.json`; this document explains
the limits behind those statuses. All commits are local and at most 1,000
added plus deleted lines. The stock baseline is `b32969b0`.

## Machine and tools actually used

One GB10 DGX Spark, Arm64, 128 GB shared memory, driver 580.178.04,
CUDA toolkit 13.0.88 / image 13.0.1, and cuDSS 0.8.0.10-1. GPU development
used Ubuntu 24.04 containers; the default CPU image used Ubuntu questing
and GCC 15. The host kernel is 7.0.0-1019-nvidia. This is not evidence of
DGX OS 8, amd64 runtime, another memory profile, or multi-Spark operation.
The supplied guide's proposed versions and throughput figures are not
measurements on this machine.

The raw cases are IEEE118, Polish_model_v33, ACTIVSg10k,
Texas7k_20210804 and MemphisCase2026_Mar7. No grid larger than 10k buses
was tested. The external RAW files are not redistributed. The small
remote-controller regression is included with the tests.

## Confirmed checks

| Check | Evidence and source snapshot |
|---|---|
| CPU-only build / stock regression | All 120 sequential CTest tests passed in 80.99 s through `e49d0084`. Parallel legacy tests share files, so the final full suite is sequential. |
| Fresh GPU build / standards CI | All supported real architectures plus PTX built. End-to-end standards CI passed all 28 batch tests in 26.94 s through `e49d0084`. Current analysis has 360 clang-tidy and eight cppcheck findings, below the recorded baseline; there are zero new-code compiler warnings and no increased findings. This does not grant maintainer approval of exceptions. |
| Comparison tools | Nine quality-audit regressions and thirteen output-comparison regressions pass. Native NVCC warning syntax is counted; equal rows still undergo shape, key and duplicate checks. |
| Final case records | `e49d0084` records CPU/GPU status, iterations, mismatch and final PV/PQ counts before cleanup. Unsolved cases do not inherit another case's history. All eligible IEEE118 cases and the remote-controller regression pass explicit PV/PQ membership and reporting checks on three backends. |
| Kernel equations | CPU/GPU mismatch relative difference 3.78e-16, Jacobian 1.49e-16, finite-difference error 4.21e-09, Alg2/KLU solve difference 4.61e-12. The component oracle covers IEEE14, IEEE118 and the existing 240-bus fixture. |
| Member failure / admission | A singular member is flagged while healthy members continue. IEEE118 tests enforce the validated cap and a small memory cap. Larger-budget pressure testing is still separate. |
| Restoration regression | Three backends pass the included two-case remote-controller regression. The old code lost a PV bus and differed by 0.0181 pu. Corrected voltage/angle errors are below 1e-15. A 65-case 10k reproduction also passes. |
| MPI / completeness | Two/four reporting ranks and two accelerator ranks on one visible GPU pass. Ordered output reconciliation detects missing, duplicate and unexpected event indices. Every generated case must have an outcome. |
| Optional loading | Absent/disabled blocks, missing GPU/plugin, invalid settings and explicitly required acceleration are tested. `ca.x` itself has no CUDA/cuDSS dependency. |
| Installation / images | The installed executable and separately rebuilt CA example both pass four-rank IEEE118/cuDSS CSV parity, exact PV/PQ sets and final-state reporting through `e49d0084`. The rebuilt example outside the installation prefix uses `GRIDPACK_BATCHPF_PLUGIN_PATH`. Earlier CPU/GPU image runtime checks passed; both images are being refreshed for the final reporting code. |
| Memory checking | Final GPU kernels through `18f15bca` passed Compute Sanitizer memcheck with zero errors. Subsequent changes affect host telemetry and the Python comparator, not kernels. |

## Full-study evidence and the fidelity problem

The older `build-dev` matrix used eight ranks, raw starts, reactive limits,
full `csv_delta` output and `shadowFraction=1`. Polish and Memphis passed on
both backends. Texas failed, and 10k finished both solver runs but exceeded
the old comparison timeout. Comparing saved 10k Alg2 output also found a
real fidelity failure; it was not solely a timeout.

| Grid | Stock seconds, separate backend comparisons | Alg2 seconds | cuDSS seconds |
|---|---:|---:|---:|
| Polish | 27.16 / 27.20 | 42.89 | 35.74 |
| 10k | 576.75 / 608.80 | 758.93 | 719.92 |
| Texas7k | 246.87 / 246.67 | 387.63 | 302.90 |
| Memphis | 7.18 / 6.95 | 10.93 | 9.77 |

These include full shadow solving and are **not production speedups**.
The old 10k output sort reached approximately 49 GB memory. The comparator
now caps GNU sort at 256 MB and streams already ordered GPU tables.

The failing shadows differed by about 0.022 pu on 10k and 0.00996 pu on
Texas, with one different PV bus. Commit `63f826af` fixes the reference
voltage retained by a CPU remote controller. The diagnosis and controlled
Texas reproduction are in `restoration.md`.

Corrected default-order Texas and 10k runs pass all GPU shadow voltage,
angle, status, classification and exact bus-set checks at `63f826af`.

| Study | Cases | Shadows | Maximum voltage difference, pu | Maximum angle difference, rad | Strict iteration differences |
|---|---:|---:|---:|---:|---:|
| Texas Alg2 | 8,891 | 8,803 | 6.861e-14 | 1.301e-13 | 2 |
| Texas cuDSS | 8,891 | 8,804 | 2.249e-13 | 1.241e-11 | 4 |
| 10k Alg2 | 15,191 | 14,581 | 1.095e-12 | 5.755e-13 | 4 |
| 10k cuDSS | 15,191 | 14,581 | 1.064e-12 | 5.684e-13 | 3 |

Stock's retained controller state causes the late iteration differences;
the isolated two-case reproduction passes. FID-2 remains open for
default-order studies. The cuDSS 10k run overlapped another validation
job, so its elapsed time is not an isolated measurement.
Text-mode checks omit the full delta table, so they do not prove full CSV
parity. Current full-CSV oracle checks are running with the controller
outage last, using identical complete explicit lists on both paths.
Texas moves `GN_240278_1`; 10k moves `GN_74332_1`. No outage is removed.
The list hashes are respectively
`8e4b4997804306b037d05bf66f0c2200be51121d58b648ddf7b3242f25cb8c76`
and `8475a0ec371fe6d8003365831580cedf07e372e0003b87643372fe76f5911f37`.
These checks cannot erase the default-order compatibility limitation.

Full current Polish and Memphis CSV checks pass on both backends through
`92a5e689`, with eight ranks, raw starts and every eligible GPU case shadowed.
Polish has 4,198 outcomes and 4,026 shadows; Memphis has 1,570 outcomes and
1,392 shadows. Maximum voltage differences are 2.028e-12 / 1.893e-12 pu
for Polish and 1.789e-12 / 1.729e-12 pu for Memphis (Alg2/cuDSS). Exact sets,
statuses, angle differences and all rounded output tables pass. The complete
Texas Alg2 oracle also passes its full CSV and exact shadow comparisons.
`study-evidence.jsonl` preserves these small summaries with RAW hashes,
source snapshots and timing context. It includes the failed default-order
studies as well as passes. Their shadow-heavy timings are not production
throughput figures.

Final reporting checks at `e49d0084` also pass full Polish/Alg2 and
Memphis/cuDSS CSV studies, all eligible shadows and explicit PV/PQ sets.
They cover all 4,198 and 1,570 case records, including CPU-only and retry
paths. Polish's 4,026 converged numerical solutions are labeled
`SLACK_OVERLOAD` by both stock and batch; a parity pass does not mean the
grid has sufficient slack capacity. Memphis has 1,387 `OK`, five
`SLACK_OVERLOAD`, 177 `DIVERGED` and one `ISLANDED` outcomes on both paths.
Maximum shadow differences are 2.028e-12 / 9.219e-13 pu/rad for Polish
and 2.146e-12 / 1.013e-12 for Memphis. A 65-case 10k sample and the two
independent Texas cases also pass new reported-state and exact PQ checks.
Their small summaries are appended to `study-evidence.jsonl`. These samples
do not replace the full large-grid CSV comparisons still running.

## Output comparison rules

Physics shadows use 1e-6 pu/rad and compare exact PV/PQ sets, in addition
to counts, statuses and classification. Rounded CSV values use a separate
1e-3 absolute output tolerance. The stricter physics checks remain required.

Warm starts deliberately differ from stock's raw starts (ADR-05), so the
benchmark records that choice and may have different iteration counts.
Raw-start parity keeps the iteration comparison. Stock writes stale
convergence history for islanded/no-slack cases it never solves; those
records are compared by status rather than a preceding case's iterations.
Diverged trajectories may differ, and both paths must report failure.

GridPACK retains a first strict maximum when results complete. Equivalent
worst-loading labels are accepted only when both named cases attain the
same loading in both violation tables. Equivalent worst-voltage bus labels
require both buses to attain the global reported minimum/maximum in both
tables within 1e-6 pu. A nearby or missing candidate fails the test. This
does not excuse a changed voltage, status, or controller state.

The component oracle preserves GridPACK's reference-bus reactive-limit
exclusion and generator-status restoration. A failed second solve is
reported as a flagged outcome and retried on the CPU. Reports use GridPACK's
existing branch and voltage checks rather than duplicate GPU formulas.

## Gates still requiring evidence

- PERF-1, PERF-2 and the tested rank sweep in PERF-4 have isolated evidence:
  all 34 repeated trials passed. Batch 512 reaches 95% of the best swept
  throughput; sixteen ranks are fastest among the tested counts. See
  `performance.md` and `performance-trials.jsonl`. A final selected repeat
  measures the reporting change's cost; physical DRAM bandwidth remains
  open under PERF-3. These results do not establish multi-Spark scaling.
- ROB-2 now has a full Polish N-1 capped-budget check: requesting 2,048
  slots with a 0.25 GB budget reduced admission to 150 slots. All full
  CSV comparisons and exact shadows passed in 67.02 s. A one-second
  sampler's maximum gap was 1.001 s, with no swap-in/out pages and at least
  120.95 GB node memory available. Other jobs were active; this tests a
  near-budget application, not a nearly exhausted host or production speed.
  A second full Polish CSV/shadow check with cuDSS passed: its validated
  cap first reduced 512 requested slots to 128, then a 0.125 GB budget
  reduced admission to 74. The runtime recorded zero swap pages.
- STD-1/4 passed end-to-end CI, including CMake analyzer properties,
  include-what-you-use, changed adapter coverage and both analyzer engines.
  Maintainer exception review remains pending.
  There are no hidden suppressions. See `standards.md` for the scope and
  the LLVM/CUDA and cppcheck/Boost tooling limits.
- Nsight Compute succeeded with the approved temporary container capability.
  Ten selected launches have measured occupancy and resource use, preserved
  in `profile-evidence.json`. The available GB10 metric query contains no
  DRAM metrics, so physical DRAM bandwidth remains unmeasured. Replay
  durations with clock/cache control disabled are not production timings.
  Telemetry bandwidth counts estimated algorithm traffic.
- PORT-1, PORT-3, PERF-5 and the amd64 part of DOCK-2 require unavailable
  hardware/OS workers. Do not turn these into passes based on a build flag.

The earlier handoff described a stock WECC240 failure in PETSc LU followed
by `__sprintf_chk`. It was not reproduced in this implementation session
and remains historical evidence, not a newly diagnosed solver failure.
