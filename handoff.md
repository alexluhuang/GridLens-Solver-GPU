# Handoff: GPU N-1 implementation

Updated 2026-10-06. Repository `/home/alh360/Documents/GridLens-Solver-GPU`,
branch `feature/gpu-batch-n1`, local commits only (not pushed). Solver source
`3b16592d`; test-only follow-up `ef11605f`; records in the commits after it.

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

Every step of the previous handoff is done:

1. `3b16592d`: GridPACK's reactive-limit check skips disconnected buses, as the
   GPU check already did. A failed case no longer leaves an injection that
   converts a later disconnected bus. This is a second compatibility change,
   separate from the user's approved voltage cleanup, and it also applies
   without acceleration (see `docs/gpu_n1/restoration.md`).
2. Corrected CPU reference: `b32969b0` plus
   `reproductions/cpu-voltage-cleanup-reference.patch` and
   `reproductions/cpu-isolated-qlim-reference.patch`, built at
   `/work/build-corrected-reference`. The original build is kept at
   `/work/build-stock`.
3. Full default-order Texas7k and 10k studies pass every check against the
   corrected reference on Alg2 and cuDSS. The saved 10k Alg2 study with the
   controller outage last also passes. Comparisons against the original
   baseline stay failed and are kept as evidence.
4. CPU-only suite 121/121; GPU build 29/29; standards CI passes (360 clang-tidy,
   8 cppcheck, below baseline); install, the rebuilt installed example and both
   refreshed Arm64 images pass their checks.

Details: `docs/gpu_n1/validation.md`, `validation-status.json`,
`study-evidence.jsonl`, `image-evidence.json`.

## Environment

Scratch `/home/alh360/.claude/jobs/23383591/tmp` is mounted as `/work` and is
job-managed. `dn.sh [--nogpu] '<cmd>'` there runs a command in
`alh360/gridpack-n1-tools:1.0` with the repository at `/src`. Builds:
`/work/build-quality` (GPU, standards CI, installs to `/work/quality-install`),
`/work/build-cpu` (CPU-only), `/work/build-corrected-reference`,
`/work/build-stock` (original `b32969b0`). The default-order study script is
`/work/run-corrected-studies.sh`; its outputs (about 140 GB) are under
`/work/validation-corrected/`. RAW files are in `/work/runs/grids/`; the
external RAW files are not redistributed.

## Not done (cannot be done on this machine, or needs a person)

- Maintainer review of the standards exception register (`standards.md`).
- PORT-1 (second Arm/NVIDIA profile), PORT-3 (DGX OS 8), PERF-5 (several
  Sparks), the amd64 part of DOCK-2.
- PERF-3 physical DRAM bandwidth: GB10 exposes no DRAM counters to Nsight.
- Full CUDA clang-tidy coverage: LLVM 18 cannot parse CUDA 13 headers.
- The user may want to confirm the isolated-bus compatibility change, since
  it alters iteration counts and warnings in the ordinary CPU path.
