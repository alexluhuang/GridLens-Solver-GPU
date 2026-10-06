# Handoff: GPU N-1 implementation and remaining validation

Updated 2026-10-05. Repository: `/home/alh360/Documents/GridLens-Solver-GPU`.
Branch: `feature/gpu-batch-n1`. Implementation through `e49d0084` plus the
approved CPU cleanup correction; commits
are local. This is GridPACK C++/CUDA, not the Python GridLens repository.
The implementation has not passed every release gate in the supplied plan.

## Instructions

Read `/home/alh360/Downloads/N1_Contingency_Solver_Architecture_Guide_v0.3.md`
first. Its exact contents are also preserved in the five numbered files
under `docs/gpu_n1/architecture/`; start with `docs/gpu_n1/README.md`.
The concatenated guide has 2,637 lines and SHA256
`cbfce273ea1d2a192de51677d869b6707ee93cad2c57f4c5d16927456df6d524`.

For physics, follow the plan's source order: Kundur and Malik §6.4 in
`/home/alh360/Downloads/PF.pdf`, then the Zhou batch-LU and D'Orto Newton
papers supplied in Downloads. `PF.pdf` is not the D'Orto paper. For software,
read GridPACK's actual implementation before vendor documentation.

Make focused commits of at most 1,000 added plus deleted lines. Explain why
and record actual checks in each commit body. Do not push, rewrite history,
weaken tolerances or discard unfamiliar artifacts. No sub-agent delegation
is authorized. Do not run grids above 10k buses.

## What is implemented

The optional host path, CUDA-free executable, core/plugin C interfaces,
Algorithm 2 and cuDSS backends, CPU reference, shared-pattern planning,
Newton and reactive-limit control, classifier, GPU health/fallback,
GridPACK reporting/restoration, MPI reporting, ordered output reconciliation,
runtime settings, memory admission, topology/binding, telemetry, installation,
example, Docker arguments and standards CI are present. Read the module
README for the real defaults and settings, not only the guide's proposals.

Recent corrections after the older handoff:

- `63f826af`: restore voltage references changed by CPU remote regulation;
  a meaningful two-generator IEEE14 regression passes all three backends.
- `affe187c` and `2d653649`: initialize adapter state, use typed diagnostics
  and sized backend buffers.
- `163439b4` and `18f15bca`: standards checklist/register, individual analyzer
  records, working CI, reviewed includes, checked solver/probe boundaries.
- `504d7b3a` and `64515dbd`: verify tied voltage labels against both tables;
  count native CUDA compiler warnings.
- `92a5e689`: report local-node swap activity, including unavailable counters.
- `18f8e840`: skip per-column work for identical CSV rows after shape checks;
  seven regressions preserve mismatch, malformed-row and duplicate checks.

All new commits meet the size limit. Full evidence and limitations belong
in `docs/gpu_n1/validation.md` and `validation-status.json`.

## Confirmed evidence

- Latest CPU-only build: 120/120 sequential CTest tests, 80.99 s at `e49d0084`.
  Legacy parallel tests share files; run the complete suite sequentially.
- Latest GPU/standards CI: all supported real architectures plus PTX;
  28/28 batch tests, 26.94 s. Nine quality-parser tests and thirteen comparator
  regressions pass. Current analysis has 360 Clang and eight cppcheck findings,
  below the unchanged recorded baseline, and no new-code compiler warnings.
  Maintainer exception approval is still pending; a baseline is not approval.
- Final kernels passed Compute Sanitizer memcheck with zero errors.
  CPU/GPU mismatch 3.78e-16, Jacobian 1.49e-16, finite-difference 4.21e-09,
  Algorithm 2/KLU solve 4.61e-12. Later commits change host/Python code.
- Full default-order Texas and 10k shadows pass voltages, angles, status,
  classification and exact PV/PQ sets on both GPU backends. Strict stock
  output parity still fails for a few late generator cases, explained below.
- Full ordered Texas Alg2 CSV oracle passes: 8,891 cases, all outages retained.
- Full Polish capped-budget test passes CSV and exact shadows: 4,198 cases,
  requested batch 2,048, admitted 150, budget 0.25 GB. No swap; minimum
  available node memory 120.95 GB; maximum one-second sampler gap 1.001 s.
  This is near an application budget, not physical host exhaustion.
- Installed executable and a separately rebuilt installed example ran the
  GPU. The rebuilt example outside the prefix needs
  `GRIDPACK_BATCHPF_PLUGIN_PATH=<prefix>/lib/gridpack/plugins`.
- Refreshed CPU image imported Python bindings and passed serial no-plugin
  fallback. Multi-rank byte-for-byte stock files differ with completion order;
  do not mistake that harness limitation for different solved values.

## The remaining fidelity limitation

Read `docs/gpu_n1/restoration.md` before troubleshooting parity.
`PFBus::adjustVoltageForRemoteReg` changes the reference used by resetVoltage.
Tripping a bus's sole local regulator can activate another unit's remote
regulation and leave that reference changed after stock cleanup.

The batch driver interprets B8.2/B10.1 as independent cases. Originally the
no-batch path was unchanged under RT-4. On 2026-10-05 the user approved
restoring settings in both paths and accepting the documented correction to
existing behavior. That fix is now implemented and tested. Preserve the
unmodified `b32969b0` baseline; build a separately corrected CPU reference
for subsequent default-order comparisons. Do not relax comparisons or claim
FID-2 passed against the original baseline.

A one-rank Texas sequence `GN_240278_1`, `GN_250371_1`, `GN_270208_1` proves
this: stock takes two iterations on the latter cases after the first outage,
while batch takes three; delta differences reach 40.7604. The latter two
cases alone take three stock iterations and pass every comparison.
Default full-study iteration differences are Texas 2/4 and 10k 4/3
(Alg2/cuDSS). All GPU shadows pass.

Additional clean oracles move the controller-activating outage last in the
same full explicit list on both paths. Texas uses `GN_240278_1`; 10k uses
`GN_74332_1`. This retains every outage and isolates the independent-case
comparison, but does not erase the default-order limitation. Input hashes
are recorded in the validation report.

## Build environment

Scratch root: `/home/alh360/.claude/jobs/23383591/tmp`, mounted as `/work`.
It is job-managed: preserve small results/manifests in the repository.
Do not delete the large saved studies merely to clean the working tree.

Use `alh360/gridpack-n1-tools:1.0`, which includes GSL 4, clang-tidy 18,
cppcheck 2.13 and IWYU. The old `gridpack-ca-cudss:1.0` image lacks GSL.

```bash
docker run --rm --gpus all --ipc=host -u 1000:1000 \
  -v /home/alh360/Documents/GridLens-Solver-GPU:/src \
  -v /home/alh360/.claude/jobs/23383591/tmp:/work \
  -v /home/alh360/Documents/GridLensDemoData:/demo:ro \
  -w /work --entrypoint bash alh360/gridpack-n1-tools:1.0 \
  -lc 'ctest --test-dir /work/build-quality -R "^batchpf" -E external_ --output-on-failure'
```

Remove GPU access for CPU-only checks. `cfg.sh` supplies dependency paths
for a new build. Do not repurpose HOME or use the old `dk.sh` helper for
current builds. Request escalation if Docker/sandbox access fails.

- `/work/build-quality`: latest solver/telemetry and standards CI; installation
  prefix `/work/quality-install`.
- `/work/build-cpu`: latest CPU-only source.
- `/work/build-analysis`: solver through `18f15bca`, used by the ordered
  full-CSV studies; numerical code matches current. **Do not rebuild its
  plugins while those studies are running.**
- `/work/build-stock`: separate unmodified baseline `b32969b0`. Its full
  test tree was not built; use its `applications/contingency_analysis/ca.x`.
- `build-dev`, `build-standards`, `analysis-install` are older snapshots.

RAW files are `/work/runs/grids/IEEE118.raw`, `Polish_model_v33.raw`,
`ACTIVSg10k.RAW`, `Texas7k_20210804.RAW`; Memphis is
`/demo/MemphisCase2026_Mar7.RAW`. Do not redistribute external RAW inputs.

## Live work at this checkpoint

Check processes and actual log endings before repeating expensive work.
Tool session IDs are conveniences, not durable proof that a job runs.
Container process namespaces need `--pid=host` to inspect host jobs.

1. Ordered large-grid queue: session 76389, host bash PID 73168, **running**.
   It was paused after Texas Alg2 passed to isolate performance, then resumed
   after all 44 timing trials finished. Texas cuDSS finished with one strict
   count failure: event 7822 `BR_210326_210331_1`, CPU zero versus GPU two.
   Other output and shadow checks pass. The isolated case passes at two
   rounds in both paths. The CPU log shows a limit check on disconnected
   generator 210331 followed by a zero-round calculation; investigate this
   separately from voltage cleanup. Evidence and one-case XML are saved.
   10k Alg2 is now running; both 10k full-CSV oracles remain. Inputs mounted from
   `/tmp/{Texas7k_20210804,ACTIVSg10k}-remote-last.xml`; preserve them while
   mounted. Logs `/work/validation-ordered-<grid>-<backend>.log`; studies
   `/work/validation-ordered/<grid>_<backend>/{stock,parity}`.
2. Final Polish/Memphis checks finished; all four passed. Metrics are in
   `study-evidence.jsonl`. The cuDSS capped-budget check also passed.
3. Final reporting images finished building and runtime checks passed.
   IDs/packages are in `image-evidence.json`; both driver hashes match current
   source. GPU: cuDSS/four ranks, independent baseline, exact PV/PQ and final
   state; both: default Python imports and serial fallback. CPU comparison
   uses its own inactive loop because the older tools baseline's MPI library
   is absent. Build logs `/tmp/gridpack-image-{cpu,gpu}-reported-state.log`;
   runtime logs `/work/image-reported-state-*.log`. Retain version argument
   `n1-validation` for dependency-cache reuse. No image is pushed.
4. Isolated performance session 53554 finished all 34 trials successfully.
   Full Polish N-1, text output, zero shadows, two trials per setting;
   raw starts except explicitly tagged base-case trials. Results are in
   `/work/performance-final/{results.jsonl,summary.json}`. Best swept batch
   2048; smallest batch within 95% of its peak 512. Rank 16 was fastest of
   1/2/4/8/16. Preserve source tags: these precede final state reporting.
5. Nsight session 28116 finished with return code zero. The approved
   temporary container `SYS_ADMIN` capability enabled counters without
   changing host driver policy. Report `/work/profiling-final/polish.ncu-rep`,
   exported `counters.csv`, `result.json`, `profile.log`; ten selected kernel
   launches, clock/cache control disabled, replay profiling (not production
   elapsed time). Occupancy was harvested; this device exposes no DRAM counters.
6. Commit `e49d0084` completes guide 8.11 with final status, iterations, mismatch
   and PV/PQ counts captured before cleanup, for CPU and GPU outcomes.
   Shadow checks now explicitly compare PQ membership too. Thirteen Python
   comparator regressions pass. Standards CI and all 28 GPU tests passed in
   26.94 s; final-source 120 CPU tests passed in 80.99 s. No findings increase
   or new compiler warnings. Ten final selected production trials all passed
   in `/work/performance-reported-state`, source `e49d0084`; the completed
   timings are preserved in `performance-reported-state.jsonl`.
7. Installed executable and separately rebuilt installed example passed
   cuDSS/four-rank full IEEE118 parity, exact PV/PQ sets and final state checks
   at `e49d0084`. Logs `/work/{install,rebuilt}-reported-state-check.log`.
8. Current reporting validation finished: full Polish Alg2, full Memphis
   cuDSS, 65-case 10k sample and independent Texas cases all pass full CSV,
   exact PV/PQ shadows and final state checks. Outputs/logs are under
   `/work/validation-reported-state`; summaries are in `study-evidence.jsonl`.
9. User-approved voltage cleanup now runs even without acceleration.
   New `batchpf.parity.cpu_case_restoration` fails before the fix (eight
   rounds and delta difference 31.7952) and passes afterward in both builds.
   Seven targeted GPU-build checks and three CPU-build checks pass. The
   broader suites, standards, installed artifacts and images need a refresh
   after the final source changes. Leave `/work/build-analysis` unchanged
   until its queue finishes; it still tests the earlier GPU-only correction.

## Finish in this order

1. Diagnose the additional isolated-generator count failure, following
   guide 8.14/8.6.2, then actual GridPACK code. Do not attribute it to voltage
   carryover. The one-case reproduction passes; a preceding case may be
   needed to expose the disconnected bus's retained calculated injection.
2. Build a CPU reference from original `b32969b0` with only the approved
   voltage cleanup correction. Preserve the original build and label both
   references clearly. Check the Texas sequence, then default-order studies.
   Run broader suites/standards and refresh installation/images after the
   final source checkpoint. Previous 44 production timings remain valid
   evidence for their recorded snapshots; repeat only affected selected checks.
3. Finish the resumed PID 73168 full CSV comparisons. Keep the
   default-order failure evidence separate. Full tables are tens of GB;
   sorting is bounded to 256 MB and Python processes one event at a time.
4. Update validation.md, structured status and this handoff. Preserve small
   benchmark/pressure/shadow summaries with input/source hashes, not GB
   outputs. Commit logically within the line limit, then check all commit
   numstats, git diff --check and a clean working tree.
5. Remove only scratch files made for this task after mounted users stop.
   Preserve reproduction XML/hash evidence first. Do not remove unfamiliar
   user artifacts or the studies used as evidence.

External gates remain: second Arm/NVIDIA profile (PORT-1), DGX OS 8
(PORT-3), multiple Sparks (PERF-5), amd64 image build/runtime (DOCK-2),
maintainer exception review. The user has decided the voltage cleanup policy;
implement that decision without asking again.
One GB10 Spark/Arm64, 128 GB shared memory, driver 580.178.04 and CUDA 13.0.1
cannot establish these other-platform results. Do not mark the entire plan
release-validated or leave a paused queue undisclosed.
