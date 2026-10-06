# GPU batch N-1 contingency analysis

This module adds an optional batch solver to `ca.x`. GridPACK still reads
the network, solves the base case, creates contingencies, checks limits,
and writes results. The plugin solves eligible cases together on the GPU.
Cases that need GridPACK controls, form multiple islands, lack a usable
slack, or fail a numerical check return to GridPACK's CPU loop.

The implementation follows [architecture guide 0.3](../../../../docs/gpu_n1/README.md).
The guide is the first reference when changing or troubleshooting this
code. Its proposed settings are distinguished from the actual defaults
below and the measurements in the validation report.

## Build and installation

Use GridPACK's normal dependency configuration. The CPU executable links
the host adapter and the dynamic loader, with no CUDA or cuDSS dependency.

```sh
cmake -S src -B build -DGRIDPACK_ENABLE_GPU_BATCH=OFF [GridPACK dependency options]
cmake --build build
```

`GRIDPACK_ENABLE_GPU_BATCH` accepts `OFF`, `AUTO` (the source-build default),
or `ON`. `AUTO` builds plugins if a CUDA compiler is available. `ON` stops
configuration if it is unavailable. GPU builds require CMake 3.23 or newer,
CUDA, KLU from SuiteSparse, and Microsoft GSL 4 or newer. The cuDSS plugin
is built when its CMake package is found; otherwise Algorithm 2 remains
available. On Ubuntu, `libmsgsl-dev` supplies GSL.

```sh
cmake -S src -B build -DGRIDPACK_ENABLE_GPU_BATCH=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=all \
  [GridPACK dependency options]
cmake --build build
cmake --install build
```

`all` builds supported real architectures and PTX for later devices. An
explicit architecture list or `CUDAARCHS` overrides it. Kernel resource
reports are enabled for NVIDIA's compiler. No fast-math switches are used.

Installed files include `bin/ca.x`, the interface headers and host archive,
and plugins under `lib/gridpack/plugins`. The installed contingency example
includes `input_118_gpu.xml` and `IEEE118.raw`. It can also be rebuilt using
the installed GridPACK package. For a rebuilt executable outside the
installation, set `GRIDPACK_BATCHPF_PLUGIN_PATH` to the installed plugin
directory. Legacy GridPACK install scripts use the configured prefix:
configure that prefix before installing.

The repository Dockerfile builds both variants. Its default remains a
CPU-only Debug image. A GPU build uses a CUDA development base:

```sh
docker build -t gridpack-gpu \
  --build-arg BASE_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04 \
  --build-arg GRIDPACK_ENABLE_GPU_BATCH=ON \
  --build-arg GRIDPACK_BUILD_TYPE=Release .
```

Builds need no GPU. Container runs need the deployment's NVIDIA GPU access
configuration. Missing device access produces a logged CPU fallback unless
the configuration explicitly requires the accelerator. CPU images need no
NVIDIA runtime.

## Run

Start with the shipped `input_118_gpu.xml` in the contingency example
directory. An empty `<GPUBatch/>` under `<Contingency_analysis>` enables
automatic detection. Without that block, or with `enabled=off`, the stock
loop runs. These are separate from GridPACK's existing `Powerflow` settings.

```xml
<GPUBatch>
  <enabled>auto</enabled>
  <backend>auto</backend>
  <batchSize>auto</batchSize>
  <warmStart>base_case</warmStart>
  <telemetry>summary</telemetry>
</GPUBatch>
<Execution>
  <acceleratorRanks>auto</acceleratorRanks>
  <cpuBinding>auto</cpuBinding>
</Execution>
```

```sh
mpiexec --bind-to none -n 8 ca.x input_118_gpu.xml
```

The launcher option permits the module's CPU placement policy to choose
cores. Use `cpuBinding=none` to retain launcher placement. The rank count
is a measurement choice; eight is an example, not a universal optimum.

Automatic roles select one accelerator rank per visible GPU on each node.
Other ranks solve CPU cases and report GPU results. Accelerator workers
never call MPI or GridPACK. The GPU solves while CPU ranks work, and the
reporting protocol hands finished cases to available ranks. More than one
accelerator rank can be selected explicitly; assigning several to one GPU
is useful for protocol tests but is not a throughput recommendation.

Set `enabled=on` and `onUnavailable=error` to require the requested path.
Invalid values and unknown keys stop at startup. The log identifies each
effective setting and its source, loaded plugins and backend versions,
device capabilities, CPU placement, memory budget and admitted batch size.

## Actual settings

These keys are under `Contingency_analysis/GPUBatch`. Integer limits shown
are input limits; memory and backend admission can lower a batch further.

| Key | Accepted values | Default / behavior |
|---|---|---|
| `enabled` | `auto`, `on`, `off` | `auto` with a block; `off` without one |
| `onUnavailable` | `fallback`, `error` | `fallback`; `error` is enforced with `enabled=on` |
| `pluginPath` | Directory | Environment `GRIDPACK_BATCHPF_PLUGIN_PATH`, then paths relative to `ca.x` |
| `backend` | `auto`, `cudss`, `alg2`, `cpu_reference` | `auto`: benchmark available GPU backends during setup |
| `device` | Integer 0..1023 | 0; index among visible devices |
| `batchSize` | `auto`, integer 1..65536 | `auto`: reference factor/solve sweep, smallest size within 5% of the best |
| `maxValidatedBatch` | Integer 1..65536 | Backend cap: Algorithm 2 2048; cuDSS 128 |
| `backfill` | Boolean | `true`: refill free slots; guide's `false` was a proposal |
| `threadsPerBlock` | `auto`, multiples of 32 in 32..1024 | `auto`; kernel-specific selection |
| `memoryProfile` | `auto`, `unified`, `coherent`, `discrete` | Capability detection |
| `memoryHeadroomGB` | Nonnegative number | 10% of shared GPU memory for unified, 5% elsewhere |
| `maxMemoryGB` | Nonnegative number | 0: no additional cap |
| `formulation` | `superset` | `superset`; `grouped` is reserved and rejected |
| `plannerOrdering` | `amd`, `colamd` | `amd` |
| `pivotTolerance` | Number 0..1 | 0.001 for the reference KLU factorization |
| `solvePlacement` | `gpu`, `host` | `gpu`; host option applies to Algorithm 2 |
| `warmStart` | `base_case`, `raw` | `base_case`; use `raw` for comparable iteration records |
| `refinementSteps` | Integer 0..100 | 0; honored by supporting backends |
| `health/residualLimit` | Number 0..1 | 1e-6; 0 disables this check |
| `health/pivotLimit` | Number 0..1 | 1e-12; Algorithm 2 uses reference column scales; cuDSS uses available factor diagonals |
| `health/checkNonFinite` | Boolean | `true` |
| `shadowFraction` | Number 0..1 | 0; share of reported GPU cases re-solved by GridPACK |
| `telemetry` | `off`, `summary`, `detailed` | `summary` |
| `profilerRanges` | Boolean | `false`; NVTX batch range for a profiler |

Explicit batches above the selected cap shrink. With `backend=auto`, an
explicit batch above the cuDSS cap selects Algorithm 2, then applies its
cap. Increasing `maxValidatedBatch` is an explicit deployment override;
it does not turn an untested size into a validated one. Available memory
always imposes an additional cap. GB values use decimal bytes.

The automatic sweep uses reference factorization and solve timings. This
is a setup heuristic; whole-study time includes classification, nonlinear
iterations, CPU fallback, reporting and output. The batch-size log and
whole-study benchmarks must be read together.

Keys under `Contingency_analysis/Execution`:

| Key | Values | Default |
|---|---|---|
| `acceleratorRanks` | `auto`, comma- or space-separated world ranks | One per visible GPU per node |
| `cpuBinding` | `auto`, `none`, `performance_first` | `auto`, discovered CPU topology |
| `logLevel` | `error`, `warn`, `info`, `debug` | `info` |

`CUDA_VISIBLE_DEVICES` and `CUDA_CACHE_PATH` remain deployment controls
and are logged. XML overrides the plugin-path environment variable.
GridPACK retains ownership of tolerance, iteration limits, damping,
reactive limits and controls. Set `qlim` explicitly in both application
blocks when comparing runs. Current code defaults to true; the older
contingency README table's false value does not describe the implementation.

## Outputs and numerical checks

Normal GridPACK outputs keep their schemas. With the batch path active,
flat, delta and violation tables are merged in event order without loading
the entire study into memory. The convergence table has one outcome per
case. Missing outcomes are inserted as `MISSING`; missing, repeated and
unexpected indices produce an incomplete-study warning.

`<outputFile>_gpu_outcomes.csv` has one row per contingency, describing
classification, CPU reason, GPU outcome, numerical-health flags, iteration
counts, reactive-limit conversions and final tolerance. Its GPU status is
the original `converged`, `diverged`, `flagged` or `not_run` result. The
ordinary convergence table contains the final GridPACK result after any
CPU retry. A GPU failure is an outcome of one case, not a lost contingency.

`<outputFile>_gpu_shadow.csv`, when sampling is enabled, records CPU and
GPU convergence, maximum voltage difference in pu, wrapped angle difference
in radians, PV bus counts, exact PV/PQ set agreement, and fast/full
classification agreement. The set check compares each bus, not only counts.
Classification checks cover changed admittances, injections and generator
limits against GridPACK's full contingency routine.

Summary telemetry reports the path mix, retry reasons, iteration histogram,
phase times, active slot fraction, factor-plan size and superset overhead.
Bandwidth figures estimate algorithm traffic divided by phase time; they
are not measured DRAM counters and can exceed physical bandwidth when
caches serve repeated values. Compiler reports and profiler counters are
separate evidence.

When cuDSS does not expose factor diagonals for uniform batches, the linear
residual and nonfinite-solution checks provide the member health checks.
Keep them enabled. Stream-ordered allocation is queried at runtime; an
unsupported device uses legacy allocation and logs the reason.

## Verification and measurement

```sh
ctest --test-dir build -R batchpf --output-on-failure
```

The tests cover element formulas, KLU agreement, isolated singular members,
controller behavior, malformed models, ordered merging, missing/duplicate
outcomes, MPI reporting, admission limits, and CPU fallback. The example
uses GridPACK's run-test helper; the parity harness compares two studies.
Run the broader stock suite sequentially: some existing serial and parallel
tests share files.

To register external full N-1 tests, configure
`GRIDPACK_BATCHPF_TEST_RAW_FILES` as a semicolon-separated list of local RAW
paths. IEEE118 is bundled. Polish, ACTIVSg10k, Texas7k and Memphis are local
validation inputs and are not redistributed. Do not run grids above 10k
buses for this task.

```sh
python3 src/applications/modules/batch_pf/test/run_ca_test.py \
  --cax /absolute/path/to/ca.x --stock-cax /absolute/path/to/stock/ca.x \
  --raw /absolute/path/to/network.raw --workdir /absolute/path/to/results \
  --mode benchmark --backend alg2 --ranks 8 --output-format text
```

The harness saves both logs, XML, timing records and `benchmark.json`.
The batch summary reports available host memory and local-node swap pages
since preparation began (guide §8.12). Swap counters include other work on
the node; a missing `/proc/vmstat` is reported as unavailable, never as zero.
Benchmark mode defaults to the RAW initial state and no shadow solves.
Use `--warm-start base_case` for production-start measurements,
`--shadow-fraction` for sampling, and repeated `--gpu-setting` arguments
for batch size or solve placement. `--contingency-list` selects an existing
XML sample. Default parity mode still compares full CSV tables and samples
every GPU case. A different worst-loading name is accepted only if both
named cases attain the same maximum on the same branch in both violation
tables; this accounts for GridPACK's completion-order choice among ties.

## Troubleshooting order

1. Locate the relevant guide requirement and runtime scenario. Read the
   effective-settings log before changing configuration.
2. For software behavior, read GridPACK's `ca_driver.cpp`,
   `PFAppModule::solve`, factory and component routines first. Then consult
   vendor documentation; third-party reports come last.
3. For physical equations, use Kundur and Malik section 6.4 (`PF.pdf`)
   first, then Zhou's batch-LU paper and D'Orto's Newton-method comparison.
   The handoff's identification of `PF.pdf` as D'Orto is incorrect.

For an unavailable plugin, check the logged search paths and its dependent
shared libraries. For a device-code error, check the architecture list and
driver/runtime log. For a small admitted batch, check the cap and memory
budget before kernel tuning. For numerical failures, inspect the member's
health flags and CPU retry, then compare RAW starts with shadow sampling.
For slow studies, compare GPU phase time with CPU reporting and output
time before optimizing kernels.
