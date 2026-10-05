## Appendix F — Standards Compliance Matrix and Exception Register

### F.1 Reviewer checklists

**S-1, C++ Core Guidelines** [CG]:

| Check | Rules |
|---|---|
| Every resource has an RAII owner; no naked `new`/`delete`/`malloc`/`free`; raw pointers are non-owning | R.1, R.3, R.10, R.11, R.20, R.21, E.6 |
| Special members follow the rule of zero, or all five are defined or deleted | C.20, C.21 |
| Interfaces are strongly typed; sequences are spans; preconditions are stated; no ownership transfer by raw pointer | I.4, I.6, I.11, I.12, I.13, F.24 |
| Errors follow ADR-19: exceptions for failures, status values for outcomes, codes at C-style boundaries and in device code | E.1, E.2, E.3, E.14, E.25–E.27, I.10, I.26 |
| Signed arithmetic and indices; no narrowing; named casts only | ES.46, ES.48, ES.49, ES.102, ES.106, ES.107 |
| No new non-const globals or singletons; no macros for constants or functions | I.2, I.3, ES.30, ES.31, Enum.1 |
| Concurrency: no data races; RAII locks; tasks; no unjustified lock-free code | CP.1–CP.4, CP.20, CP.100 |
| Performance changes justified by measurement | Per.1, Per.6 |
| clang-tidy `cppcoreguidelines-*` clean, or suppressions cite register entries | P.12, App. D |

**S-2, CUDA C++ Best Practices Guide** [CUDA-BP]:

| Check | Sections |
|---|---|
| Profiled before optimizing; effective bandwidth reported | §4.1, §9.2 |
| Host–device transfers minimized; zero-copy used on integrated GPUs | §10.1, §10.1.3 |
| Global accesses coalesced; global-memory use minimized; warp divergence avoided | §10.2.1, §12.2, §13.1 |
| Block sizes multiples of 32; occupancy checked | §11.1, §11.3 |
| Signed loop counters; no fast-math or reduced-precision switches | §12.1.5, §12.1, §20.1 |
| Every CUDA call checked; `cudaGetLastError` after each launch | §17.2 |
| Device availability tested with fallback; built for all architectures plus PTX; static runtime | §17.1, §17.3, §17.4 |
| Shared host/device functions unit-tested on both; tolerance-based comparison | §7.1, §7.3 |

**S-3, Dockerfile reference** [DF]: pinned `syntax` with `check=error=true`; `ARG` before `FROM` only for base selection; exec-form `CMD`; explicit `WORKDIR`; `--no-install-recommends` and locked cache mounts for apt; checksums on remote downloads; no secrets in `ARG`; `ENV` only for values the running image needs.

**S-4, Effective Modern CMake** [EMC]: targets with explicit scopes; no directory-scoped commands or `CMAKE_CXX_FLAGS` edits; imported targets for dependencies; no `file(GLOB)`; namespaced aliases and exported interfaces; header-source pairing; analyzers configured in CI scripts; no `-Werror`; test naming convention.

### F.2 Register entry template

| Field | Content |
|---|---|
| ID | `EX-<standard>-<number>` |
| Standard and rule | For example, [CG ES.48] |
| Location | Target, file, function, or Dockerfile/CMake line |
| Required functionality | What cannot be achieved otherwise |
| Why no compliant alternative | The specific obstacle |
| Containment | The smallest unit that holds the deviation |
| Reviewer and date | — |
| Revisit trigger | The event that would allow removal |

### F.3 Initial register (known at design time)

| ID | Rule | Location | Justification and containment | Revisit trigger |
|---|---|---|---|---|
| EX-CG-01 | [CG P.2] Write in ISO Standard C++ | `.cu` files of the core and backend plugins | GPU kernels need CUDA's language extensions (kernel qualifiers, launch syntax), which ISO C++ does not provide. Confined to the plugins; host interfaces stay ISO C++. | None foreseen |
| EX-CG-02 | [CG T.10] Specify concepts for all template arguments | New templates | Language concepts need C++20, while ADR-18 fixes C++17 for compatibility with GridPACK and the CUDA toolchain. Requirements are documented and checked with static assertions. | Toolchain moves to C++20 |
| EX-CG-03 | [CG I.2] Avoid non-const global variables; [CG C.153] Prefer virtual function to casting | GridPACK adapter layer (B1, B5, B11 extensions) | Interoperating with GridPACK requires calling its existing static configuration setters and downcasting generic components to power-flow components, because GridPACK's interfaces offer no virtual alternative. The downcasts themselves follow [CG C.146]–[CG C.148] (`dynamic_cast` where hierarchy navigation is unavoidable). Confined to one adapter layer ([CG I.30]). | Upstream GridPACK interface changes |
| EX-CG-04 | [CG ES.48], [CG ES.49] Avoid casts | B13 Accelerator Loader, one function | POSIX dynamic loading returns an untyped address that must be converted to a function pointer. One named cast, in one function, behind a typed wrapper ([CG I.30]). | None foreseen |
| EX-CUDA-01 | [CUDA-BP §12.1] Medium priority: fast math when speed trumps precision | All kernels | Deliberately not applied: precision outranks speed (T-5). Recorded per Section 2.5.2. | Never, unless fidelity requirements change |
| EX-DF-01 | [DF] ENV persistence warning | Existing `ENV DEBIAN_FRONTEND=noninteractive` | Existing line; changing it would alter the image's environment. Left unchanged under S-3. New steps do not add persistent build-only variables. | Separate behavior-review clean-up |
| EX-DF-02 | [DF] Verifying remote sources (`ADD --checksum`) | Existing Boost and Global Arrays downloads and the PETSc clone | Existing lines; left unchanged under S-3. New downloads are checksum-verified. | Separate clean-up |
| EX-CM-01 | [EMC] No directory-scoped commands | GridPACK top-level and contingency-analysis `CMakeLists.txt` (`include_directories`, `add_definitions`) | Existing lines; left unchanged under S-4. New targets use target-scoped commands only. | Upstream modernization |
| EX-CM-02 | [EMC] Use exported targets of external packages | Link line of `gridpack::batchpf_host` | GridPACK exposes Boost, Global Arrays, PETSc, ParMETIS, and MPI through variables, not targets; the host library must link what `ca.x` links. Confined to one `target_link_libraries` call. | GridPACK exports targets |
| EX-CM-03 | [EMC] Avoid custom variables in target definitions | Existing `target_libraries` variable in the contingency-analysis directory | Existing pattern; left unchanged under S-4. | Upstream modernization |


**Resolved by this design (no exception needed):**
- The existing plain-signature `ca.x` link call gains `PRIVATE` (Section 8.19).
- The missing `CMD`/`ENTRYPOINT` is resolved by an explicit exec-form `CMD ["/bin/bash"]` (Section 7.3.3).
- C-style plugin boundaries are the guidelines' own recommendation for cross-compiler ABIs [CG I.26].
- Device-code error codes are the guidelines' own approach when exceptions are unavailable [CG E.25–E.27].

---

## Appendix G — Runtime Settings Reference

All keys below are under `Configuration/Contingency_analysis` unless shown otherwise. Defaults marked [P] are proposals, to be fixed by validation. Every key is read by B14 and logged with its source (RT-3).

### G.1 `GPUBatch` block

| Key | Values | Default | Meaning |
|---|---|---|---|
| `enabled` | `auto`, `on`, `off` | `auto` | Whether to use the accelerator. Block absent means `off` (stock GridPACK). |
| `onUnavailable` | `fallback`, `error` | `fallback` | Behavior when `enabled=on` but the accelerator cannot be used. |
| `pluginPath` | Directory | Relative to `ca.x` | Location of the plugins. |
| `backend` | `auto`, `cudss`, `alg2`, `cpu_reference` | `auto` | Batch linear-solver backend. |
| `device` | Integer | 0 | Index among visible devices (`CUDA_VISIBLE_DEVICES`). |
| `batchSize` | `auto` or integer | `auto` | Members per batch. |
| `maxValidatedBatch` | Integer | From validation | Upper limit admitted for the selected backend (Risk R-2). |
| `backfill` | `true`, `false` | `false` [P] | Refill freed batch slots during a batch. |
| `threadsPerBlock` | `auto` or a multiple of 32 | `auto` | Kernel launch configuration [CUDA-BP §11.3]. |
| `memoryProfile` | `auto`, `unified`, `coherent`, `discrete` | `auto` | Memory paradigm override (Section 8.8). |
| `memoryHeadroomGB` | Number | From validation | Memory kept free on unified-memory systems. |
| `maxMemoryGB` | Number | None | Optional cap on accelerator memory. |
| `formulation` | `superset`, `grouped` | `superset` | GPU Jacobian formulation; `grouped` is reserved. |
| `plannerOrdering` | `amd`, `colamd` | `amd` | Fill-reducing ordering for planning. |
| `pivotTolerance` | Number | 0.001 | Partial-pivoting tolerance of the reference factorization [PETSC]. |
| `solvePlacement` | `gpu`, `host` | `gpu` | Where triangular solves run (ADR-04). |
| `warmStart` | `base_case`, `raw` | `base_case` | Starting point for contingency cases (ADR-05). |
| `refinementSteps` | Integer ≥ 0 | 0 | Iterative-refinement steps, where the backend supports them. |
| `health/residualLimit` | Number | From validation | Per-member residual limit. |
| `health/pivotLimit` | Number | From validation | Smallest acceptable pivot magnitude. |
| `health/checkNonFinite` | `true`, `false` | `true` | Detect NaN or infinite values. |
| `shadowFraction` | 0 to 1 | 0 | Share of GPU cases re-solved by GridPACK and compared. |
| `telemetry` | `off`, `summary`, `detailed` | `summary` | Telemetry detail. |
| `profilerRanges` | `true`, `false` | `false` | Emit profiler ranges for Nsight. |

### G.2 `Execution` block

| Key | Values | Default | Meaning |
|---|---|---|---|
| `acceleratorRanks` | `auto` or rank list | `auto` | Ranks that drive GPUs; `auto` assigns one per visible GPU. |
| `cpuBinding` | `auto`, `none`, `performance_first` | `auto` | Thread placement policy (Section 8.9). |
| `logLevel` | `error`, `warn`, `info`, `debug` | `info` | Log verbosity. |

### G.3 Proposed GridPACK setting (extension E6)

| Key | Values | Default | Meaning |
|---|---|---|---|
| `Configuration/Powerflow/jacobianFormulation` | `standard`, `large` | `standard` | GridPACK's own Jacobian formulation; `large` replaces the compile-time `LARGE_MATRIX` switch. |

### G.4 Environment variables

| Variable | Meaning | Basis |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | Devices visible to the run | [CUDA-BP §18.5] |
| CUDA JIT cache variables | Location and size of the JIT cache (see the CUDA Programming Guide) | [CUDA-BP §18.4] |
| `GRIDPACK_BATCHPF_PLUGIN_PATH` | Plugin directory, if `pluginPath` is not set | [P] |
| `CUDAARCHS` | Build time only: default GPU architectures for configuration | [CMAKE] |

### G.5 Existing GridPACK keys used unchanged

`networkConfiguration` (and versioned variants), `tolerance`, `maxIteration`, `qlim`, `qlimDeadband`, `maxQlimIterations`, `FullBranchN1`, `FullGeneratorN1`, `contingencyList`, `minVoltage`, `maxVoltage`, `contingencyRating`, `violationSeverityThreshold`, monitor filters, `outputFormat`, `outputFile`, `LinearSolver/PETScOptions` [GPK-CA][GPK-PF].

---

## References

**Physics, methods, and planning**
- **[K]** P. Kundur, O. P. Malik, *Power System Stability and Control*, 2nd ed., McGraw-Hill, ISBN 9781260473544, Section 6.4 "Power-Flow Analysis."
- **[G]** L. L. Grigsby (ed.), *Power System Stability and Control*, CRC Press, 2012, Ch. 23.
- **[D]** M. D'Orto et al., "Comparing Different Approaches for Solving Large Scale Power-Flow Problems With the Newton-Raphson Method," *IEEE Access*, vol. 9, pp. 56604–56615, 2021. doi:10.1109/ACCESS.2021.3072338
- **[Z]** G. Zhou et al., "GPU-Based Batch LU-Factorization Solver for Concurrent Analysis of Massive Power Flows," *IEEE Trans. Power Systems*, vol. 32, no. 6, pp. 4975–4977, 2017. doi:10.1109/TPWRS.2017.2662322
- **[Z16]** G. Zhou et al., "GPU-Accelerated Batch-ACPF Solution for N-1 Static Security Analysis," *IEEE Trans. Smart Grid*, vol. 8, pp. 1406–1416, 2017.
- **[W21]** Z. Wang, S. Wende-von Berg, M. Braun, "Fast parallel Newton–Raphson power flow solver for large number of system calculations with CPU and GPU," *Sustainable Energy, Grids and Networks*, 2021. https://arxiv.org/abs/2101.02270
- **[EX]** "Towards Efficient Alternating Current Optimal Power Flow Analysis on Graphical Processing Units," OSTI 2001400. https://www.osti.gov/servlets/purl/2001400
- **[SPP]** Southwest Power Pool, *SPP 2020 TPL-001-4 Planning Assessment Scope*.
- **[GPS]** "A novel GPU-accelerated strategy for contingency screening of static security analysis."

**GridPACK and PETSc**
- **[GPK] [GPK-CA] [GPK-PF] [GPK-YM] [GPK-MATH] [GPK-PAR] [GP]** GridPACK, https://github.com/GridOPTICS/GridPACK (develop branch): `README.md`, `Dockerfile`, `install_gridpack_deps.sh`; `src/applications/contingency_analysis/`; `src/applications/modules/powerflow/`; `src/applications/components/pf_matrix/`; `src/applications/components/y_matrix/`; `src/math/petsc/`; `src/parallel/task_manager.hpp`; `src/parser/`.
- **[PETSC]** PETSc manual pages: MATSOLVERKLU, https://petsc.org/release/manualpages/Mat/MATSOLVERKLU/; MatSolverType; "Advanced Features of Matrices and Solvers."

**NVIDIA platform**
- **[SPK-HW]** NVIDIA DGX Spark product page and DGX Spark User Guide, Hardware Overview, https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- **[SPK-PG]** NVIDIA DGX Spark Porting Guide, https://docs.nvidia.com/dgx/dgx-spark-porting-guide/
- **[SPK-RN]** DGX Spark Release Notes, https://docs.nvidia.com/dgx/dgx-spark/release-notes.html
- **[DGXOS8]** NVIDIA DGX OS 8 User Guide, https://docs.nvidia.com/dgx/dgx-os-8-user-guide/
- **[CUDA-UM]** CUDA Programming Guide, "Unified and System Memory" and "Unified Memory," https://docs.nvidia.com/cuda/cuda-programming-guide/
- **[CUDA-RN]** CUDA Toolkit Release Notes (13.x), https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- **[TEGRA]** CUDA for Tegra Application Note, https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/
- **[NV-DSS]** NVIDIA cuDSS documentation and release notes, https://docs.nvidia.com/cuda/cudss/
- **[NV-RF]** NVIDIA cuSOLVER documentation (cuSolverRF), https://docs.nvidia.com/cuda/cusolver/
- **[NV-F]** NVIDIA Developer Forums, cuDSS 0.8.0 uniform-batch NaN report (2026).

**Engineering standards and build tooling**
- **[CG]** B. Stroustrup, H. Sutter (eds.), *C++ Core Guidelines*, version of June 14, 2026. https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
- **[CUDA-BP]** NVIDIA, *CUDA C++ Best Practices Guide*, Release 13.4, September 15, 2026. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **[DF]** Docker, *Dockerfile reference*. https://docs.docker.com/reference/dockerfile/
- **[EMC]** *Effective Modern CMake* (community guide based on talks by M. Ropert and D. Pfeifer), supplied as `effective_modern_cmake.md`.
- **[CMAKE]** CMake documentation: `CUDA_ARCHITECTURES`, https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html; `CMAKE_CUDA_ARCHITECTURES`, https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html; `CheckLanguage`, https://cmake.org/cmake/help/latest/module/CheckLanguage.html; `enable_language`, https://cmake.org/cmake/help/latest/command/enable_language.html
- **[CUDA-IG]** NVIDIA, *CUDA Installation Guide for Linux*, Release 13.4. https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- **[NV-IMG]** NVIDIA CUDA container images. https://hub.docker.com/r/nvidia/cuda ; https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda
- **[CRS]** ceres-solver issue #1125, "Build error with cudSS" (2024). https://github.com/ceres-solver/ceres-solver/issues/1125
- **[ULT]** Ultralytics, "YOLO26 on NVIDIA DGX Spark" (2026). https://docs.ultralytics.com/guides/nvidia-dgx-spark

**Third-party reports and documentation practice**
- **[KS]** Kubesimplify, "Day 3: DGX Spark Unpacked. GB10, Unified Memory, sm_121, and NVFP4" (2026).
- **[SR]** StorageReview, "NVIDIA DGX Spark Review" (2025).
- **[CT]** CpuTronic, "NVIDIA GB10: Detailed Specifications and Benchmark Ratings" (2026).
- **[OGKM]** NVIDIA open-gpu-kernel-modules issue #1358 (2026).
- **[GS]** GPUSmith, "NVIDIA DGX Spark Review" (2026).
- **[QT]** J. Probst, "Best Practices for Architecture Documentation," Qt Group, 2025.
- **[UG]** "The Ultimate Guide to Software Architecture Documentation," workingsoftware.dev.
- **arc42** template, https://github.com/arc42/arc42-template
