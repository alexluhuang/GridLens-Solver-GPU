# Handoff: GPU N-1 implementation

Repository `/home/alh360/Documents/GridLens-Solver-GPU`, branch
`feature/gpu-batch-n1`. The remote branch is at `d62432ec` (the user's
timer commit); the commits after it are local and not pushed.

## Instructions that still apply

- Read the guide (`~/Downloads/N1_Contingency_Solver_Architecture_Guide_v0.3.md`,
  also under `docs/gpu_n1/architecture/`) first. For physics, follow its source
  order: Kundur and Malik §6.4 (`PF.pdf`), then Zhou (batch LU) and D'Orto
  (KLU planning). For software, read GridPACK's code before vendor docs.
- Commits: at most 1,000 changed lines, messages of at most two sentences, no
  co-author lines. Put reasoning in code comments and `docs/gpu_n1/`.
- Do not push, rewrite history, weaken tolerances, or run grids above 10k buses.
- Run the CPU-only suite sequentially (legacy tests share files).

## State

Updated 2026-10-07. The user's last request (implement P1 with a runtime
`parquet` format, P2 and P4, not P3; choose the rank count from the cores;
test with `csv_flat` at 16 ranks) is done:

1. Ranks: `ca_run.sh` (`ce951584`) runs cores − 4 ranks with 8 or more
   cores, otherwise cores − 2 (at least 1); the test harness uses the same
   rule. CPU comparisons use the same count.
2. P1 (`4ffdeb13`): csv_flat and csv_delta rows are built from the solved
   numbers, byte-identical to the text route.
3. Parquet (`e1caa4ed`): `outputFormat=parquet`, a branches file plus a
   flows dataset directory; optional at build time (`GRIDPACK_ENABLE_PARQUET`).
4. P2 (`c9755ddb`): GPU cases whose topology the classifier knows skip
   GridPACK's full lone-bus and island searches, full admittance updates,
   injections and violation checks; byte-identical outputs.
5. P4 (`a6ce62aa`, `9669c544`, `9937c912`): several warps per long factor
   column (bitwise-identical factors), unrolled solves, CUDA graphs,
   refilling slots from the next submission plus immediate steps without a
   factorization, cuDSS in the planner's order, deterministic mode, cap
   2048. Device-side control was not needed: the per-step wait leaves no
   measurable idle time.
6. Checks: CPU-only suite 124/124, GPU batch suite 38/38, byte identity on
   four grids, and six full studies against the corrected reference
   (`docs/gpu_n1/validation.md`). Benchmarks: `docs/gpu_n1/performance.md`,
   last section. 10k: CPU 251 s, Alg2 42 s, cuDSS 43 s at 16 ranks.

Decisions left to the user:

- csv_flat and csv_delta (and parquet) list only the first circuit of a
  branch object that holds several parallel circuits. This is the stock
  text route's behavior, reproduced exactly by P1: it drops about 489
  circuits per case on 10k, 494 on Texas7k and 9 on Polish.
- The first convergence row written by each rank prints its tolerance with
  six significant digits instead of four (the precision is set after it).
  A one-line fix would change those rows.
- Rank rule for 7 cores (5) and for 1 to 3 cores (1) was not specified;
  the chosen values follow the cores − 2 rule.
- Multi-rank table writing (`27042644`) is done: the 10k merge fell from
  8.7 to 4.0 s and is now limited by the disk. Against untouched stock at
  16 ranks the GPU path is 9.1x faster on 10k, 6.2-6.6x on Texas7k, and no
  faster on the 993-bus Memphis case (`docs/gpu_n1/performance.md`, last
  section). Reporting (about 19 s per rank on 10k) now sets the loop time.

The previous handoff's items remain done (`2d3bcbb2` isolated-bus check,
corrected CPU reference at `/work/build-corrected-reference`, the original
build at `/work/build-stock`).

## Environment

Scratch `/home/alh360/.claude/jobs/23383591/tmp` is mounted as `/work` and is
job-managed. `dn.sh [--nogpu] '<cmd>'` there runs a command in
`alh360/gridpack-n1-tools:1.1` (1.0 plus Apache Parquet 25.0.1 and pyarrow
21) with the repository at `/src`. Builds:
`/work/build-quality` (GPU, standards CI, installs to `/work/quality-install`),
`/work/build-cpu` (CPU-only), `/work/build-corrected-reference`,
`/work/build-stock` (original `b32969b0`). Study scripts: `/work/run-mrw-bench.sh` and `/work/run-mrw-parity.sh`
(multi-rank writing; `bench_runs.py` runs any binary on any grids),
`/work/build-stock-timed` (stock plus timers only), `/work/run-p4-studies.sh` (outputs under
`/work/validation-p4/`) and the earlier `/work/run-corrected-studies.sh`
(`/work/validation-corrected/`). `/work/final_bench.py` produced the last
benchmark table; `/work/alg2bench/` is a standalone factorization benchmark
that reads Jacobians dumped from a run (the dump hook was removed from the
source). RAW files are in `/work/runs/grids/`; the
external RAW files are not redistributed.

## Not done (cannot be done on this machine, or needs a person)

- Maintainer review of the standards exception register (`standards.md`).
- PORT-1 (second Arm/NVIDIA profile), PORT-3 (DGX OS 8), PERF-5 (several
  Sparks), the amd64 part of DOCK-2.
- PERF-3 physical DRAM bandwidth: GB10 exposes no DRAM counters to Nsight.
- Full CUDA clang-tidy coverage: LLVM 18 cannot parse CUDA 13 headers.
- The user may want to confirm the isolated-bus compatibility change, since
  it alters iteration counts and warnings in the ordinary CPU path.
