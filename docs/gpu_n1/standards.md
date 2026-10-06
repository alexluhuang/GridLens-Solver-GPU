# Standards review and exceptions

This records the implementation review against guide §§2.5, 8.17–8.21 and
Appendix F. It does not change the supplied design. Codex reviewed the
locations below on 2026-10-05; maintainer approval is pending. In particular,
a recorded analyzer finding is not evidence that a release gate passed.

## Analysis coverage and the CI entry point

`src/applications/modules/batch_pf/test/quality.cmake` configures clang-tidy
`cppcoreguidelines-*` and include-what-you-use on the new C++ targets through
CMake's target properties. It then records diagnostics from all new module
C++ and from changed adapter lines. The adapter scope is computed from
`git diff b32969b0`, including committed and working-tree changes. Compiler
warnings from those sources are counted separately from analyzer findings. Untouched
GridPACK code keeps its existing build and analysis settings.

cppcheck supplies a second analysis of the module. Clang's path-sensitive
static analyzer checks the changed adapters. cppcheck 2.13 failed while
preprocessing Boost 1.81 MPL in those adapters; suppressing that error would
give incomplete coverage. The audit rejects preprocessing and compiler
failures, even if their location is outside the selected lines.

LLVM 18 cannot parse CUDA 13 translation units because its CUDA wrapper
expects `texture_fetch_functions.h`. Those files instead receive NVCC
warnings and resource reports, manual review, CPU/GPU formula checks, and
Compute Sanitizer. Host/device formula headers are also parsed through the
C++ oracle test. Full CUDA clang-tidy coverage remains a tooling gap.

The findings are retained in `standards-findings.jsonl`, including file,
line, check, severity and message. CI rejects any increased count for a
diagnostic signature. Moving a line does not hide a second occurrence.
There are no blanket `NOLINT` directives or disabled guideline checks.
The baseline must be reviewed alongside this register when it changes.

Run the entry point with a full checkout and a configured Unix Makefiles
build containing GridPACK's normal dependency paths:

```bash
ctest -S src/applications/modules/batch_pf/test/quality.cmake \
  -DCTEST_SOURCE_DIRECTORY="$PWD/src" \
  -DCTEST_BINARY_DIRECTORY=/path/to/gridpack-build \
  -DBATCHPF_QUALITY_JOBS=6 -V
```

A fresh build can pass the usual dependency options in
`CTEST_CONFIGURE_OPTIONS`, separated by semicolons. Required tools are
clang-tidy, run-clang-tidy, include-what-you-use, cppcheck and Python.
Build, analysis and test logs stay in the build directory; CI does not
publish results or change the baseline. The Python unit test verifies
changed-line coverage, diagnostic counts, parser failures, and rejection
of analyzer errors. GPU tests require a worker with GPU access.

## C++ review

| Appendix F check | Implementation and evidence |
|---|---|
| Resource lifetime | Buffers, streams, events, loaded libraries and backend resources have scoped owners. Buffers outlive operations on their stream. The cuDSS wrapper unwinds partly created handles on setup failure. |
| Special members | Resource classes use the rule of zero, or explicitly define/delete all five operations. The classifier cannot be copied because it references one prepared network. |
| Interfaces | Bus, branch, case and member indices have distinct types. C++ backend sequences use GSL spans. Model boundaries check counts, adjacency and index ranges; buffer transfers assert their bounds. C and GridPACK views are covered below. |
| Errors | Internal operation failures throw purpose-specific errors. Expected case failures return outcomes. C entry points catch exceptions and return codes, including a separate allocation-failure code. Device health flags isolate members. |
| Arithmetic | New model boundaries check narrowing. Fixed records are value-initialized. Moved legacy power-flow blocks retain signed branch counts and initialize their accumulators. |
| State and concurrency | Each session owns its state. KLU worker tasks use separate common blocks. MPI communication and reporting remain ordered; existing GridPACK static flags stay in the adapter. |
| Source dependencies | Headers have guards; public headers do not import a global namespace. Analyzer choices are in the CI entry point. Include-what-you-use suggestions need review before changing GridPACK's includes. |

The remaining bounds diagnostics concern CUDA views, C callbacks and MPI
byte records, plus the existing GridPACK matrix pointer interface. Numeric
literal diagnostics include equation dimensions, configuration defaults,
serialization field positions, formatting widths and explicit test data.
They are recorded individually rather than suppressed. Named setting fields
already explain their defaults; literal findings are not numerical failures.
The findings still require a maintainer's standards review, especially the
host serialization code; the baseline is not a claim of warning-free C++.

The CPU KLU constructor-initialization suggestion is retained: its symbolic
analysis must follow `klu_defaults`, while the vectors are constructed first
so an allocation failure cannot strand a symbolic object. cppcheck's
`owned` variable warning concerns a `unique_ptr` whose destructor releases
the plugin; replacing it with an explicit delete would defeat its purpose.
The seven `useStlAlgorithm` suggestions concern ordinary loops, not defects.

## CUDA review

| Appendix F check | Implementation and limit |
|---|---|
| Profile first | Phase events and optional profiler ranges exist. The approved temporary container capability enabled ten Nsight launches after ordinary-user access was denied. Occupancy is measured; the GB10 metric query exposes no DRAM counters. See `performance.md`. No host driver policy was changed. |
| Effective bandwidth | Telemetry divides estimated bytes by phase time. It is not a measurement of physical DRAM traffic. Saturation and whole-study measurements are separate validation gates. |
| Transfers | The model and working arrays stay on the device. Integrated-GPU result exchange uses pinned zero-copy memory; discrete exchange uses copies. No CPU dereference of `cudaMalloc` memory is assumed. |
| Coalescing | A matrix position stores all batch members contiguously. Factor levels share a structure across members, following Zhou Algorithm 2. Masked members skip work. |
| Launch configuration | Explicit block sizes must be multiples of 32. Automatic selection queries kernel resource limits. NVCC resource reports cover all compiled variants; ten actual launches also have register/occupancy evidence. |
| Allocation | Stream-ordered pools are used when supported, with a checked older-allocation fallback. Admission accounts for model, work, factor, result and host-replica memory. Proposed headroom remains conservative pending pressure measurements. |
| Precision | All equations and factors are double precision. There is no fast-math or reduced-precision compiler switch. |
| Failures | Launches use `cudaGetLastError`; asynchronous phase/stream completion is checked. Destructors report cleanup failures without throwing. Capability queries distinguish absence from failure; device-code health uses status bits. |
| Compatibility | Kernels build for all supported real architectures plus PTX; the core links a static CUDA runtime. The CUDA-free executable loads plugins optionally. Missing device/library/code returns a logged CPU fallback. |
| Verification | Formula, Jacobian, finite-difference and KLU checks pass on CPU/GPU. A singular member is isolated. Compute Sanitizer previously reported zero memory errors; final-source evidence belongs in the validation report. |

## Docker and CMake review

The Dockerfile pins its syntax frontend and treats Docker build-check
warnings as errors. New build arguments select the base, plugin build,
architectures, build type and cuDSS package. Its default remains CPU-only.
New apt steps use locked cache mounts and `--no-install-recommends`.
`CMD` is an exec-form shell; there is no new entry-point executable.
The CPU and GPU images built successfully on Arm64, including Python
bindings. These builds do not prove amd64 or DGX OS 8 runtime support.

New CMake targets have explicit scopes, aliases, imported CUDA/KLU/GSL/cuDSS
dependencies, and explicit source lists. The executable links only the
host library. CUDA is enabled in the plugin subdirectory and is optional.
No new glob or global C++ flag assignment is used. The one local inherited
flag removal is recorded below. Compiler diagnostics from untouched
GridPACK sources remain outside the new-code warning count.

## Exception register

For every entry, the reviewer/date is **Codex, 2026-10-05; maintainer review
pending**. Each row states the required function, the unavailable compliant
alternative, its containment and the event that permits removal.

| ID and rule | Location | Required function, obstacle and containment | Revisit trigger |
|---|---|---|---|
| EX-CG-01, P.2 | Module `core/*.cu`, `cudss/backend_cudss.cu`, `test/test_kernels.cu` | GPU kernels require CUDA qualifiers and launch syntax. ISO C++ has no equivalent. Only plugins and GPU tests use the extensions; the host executable remains ISO C++. | A portable GPU language can provide the same required behavior. |
| EX-CG-02, T.10 | `core/common.hpp` buffers/exchange; host byte serialization templates | C++ language concepts require C++20, while ADR-18 requires C++17. Template element requirements use `static_assert`. | The supported GridPACK/CUDA toolchain moves to C++20. |
| EX-CG-03, I.2/C.153 | `host/classifier.*`, power-flow adapters and PF components | GridPACK exposes static configuration setters and generic shared component pointers. It provides no virtual power-flow interface. Existing flags and checked `dynamic_cast` stay in the adapter; the engine uses exported records. | Upstream component/configuration interfaces change. |
| EX-CG-04, ES.48/ES.49 | `host/accelerator.cpp` loader, `core/backend_plugin.cpp::openBackend`, `core/platform.cu::dmaBufSupported` | POSIX/runtime entry-point queries return untyped addresses. Typed function calls require one named cast per wrapper. No driver library is added to the host executable. | A loader returns a typed function pointer directly. |
| EX-CG-05, F.24/bounds profile | `core/pf_kernels.cuh`, CUDA backend views | Shared C++17 host/device functions need buffers accessible to NVCC. GSL 4 spans are host functions, so kernels use non-owning pointers with model/member counts. Owners and host span checks remain outside kernels. | Supported GSL/device spans can compile on all required targets. |
| EX-CG-06, F.24/bounds profile | PF component matrix/RHS methods switched from `LARGE_MATRIX` to runtime layout | GridPACK's existing matrix interface supplies raw output pointers with fixed block sizes. Changing that interface would change unrelated framework code. Keep the original formulas and sizes, initialize accumulators, and check them against the component oracle. | GridPACK adopts sized matrix interfaces. |
| EX-CG-07, bounds profile | `host/batch_path.cpp` classification packing; `host/batch_run.cpp` result packing/report callback | MPI and the stable C result interface use flat byte/pointer records. These views borrow session-owned vectors and carry counts; no pointer transfers ownership. The host remains independent of CUDA/GSL installation. This is a containment record, not a waiver for new unchecked host indexing. | A CUDA-free, supported C++17 span is available for the host, or the wire adapter is revised. |
| EX-CUDA-01, §12.1 | All GPU targets | Fast math is deliberately omitted because fidelity T-5 takes priority. Ordinary double-precision math supplies the required results. | Fidelity requirements explicitly change. |
| EX-DF-01, ENV persistence | Existing Dockerfile `DEBIAN_FRONTEND` | Preserve the original image's environment under S-3. New installation steps add no persistent build-only variable. | Separate image behavior review. |
| EX-DF-02, remote-source verification | Existing Boost/GA `wget` and PETSc clone | Preserve the original dependency installation under S-3. No new unverified remote source archive is added; GPU libraries come from the configured apt repository. | A separate dependency-pinning cleanup is authorized. |
| EX-CM-01, directory commands | Existing root and contingency-analysis CMake | Preserve GridPACK's directory-scoped flags/includes under S-4. New targets have scoped interfaces. | Upstream build modernization. |
| EX-CM-02, imported targets | `gridpack_batchpf_host` dependency link call | GridPACK supplies GA/MPI/other framework dependencies through variables. No exported package target is available to substitute. Variable use stays in the adapter's link/include calls. | GridPACK exports dependency targets. |
| EX-CM-03, custom target variables | Existing contingency-analysis `target_libraries` | Follow the original application link list under S-4; new `ca.x` linking has an explicit `PRIVATE` scope. | Upstream build modernization. |
| EX-CM-04, directory command | Batch module CMake `remove_definitions` | Parent `add_definitions` propagates GCC-only warning flags into NVCC. Target options cannot remove inherited flags. Remove only those inherited flags in the new subtree, then apply scoped C++/CUDA warnings. | The parent uses language-specific target options. |

Tool gaps (CUDA/LLVM parser, cppcheck/Boost parser, unavailable DRAM metrics)
must be revisited on toolchain upgrades. They do not justify suppressing
compiler failures or presenting missing measurements as passed checks.
