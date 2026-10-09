# How this work differs from the published work

This part compares the system of [01-methodology.md](01-methodology.md) with
each of the sources it was asked to be measured against, and then states
plainly which parts are borrowed, which are extended and which have no
precedent in these sources. Page or section numbers follow each citation.
Full references are at the end.

Most of the cited papers time one solver or one kernel against a
single-threaded CPU program. The times here are for **whole studies**, from
the network file to the final tables, against a **16-process** CPU program
(stock GridPACK) on the same machine. Their speedup figures and ours are
therefore not directly comparable; the comparison below is about method.

## 1. Source by source

### Palmer et al. 2014, GridPACK

*What it is.* The framework this work is built on: distributed network
objects, "mappers" that build matrices from per-bus and per-branch
contributions, PETSc solvers, and a task manager that hands independent
jobs to processes (69–73). Its contingency application solves one case per
process. The authors note that setup and input/output can be "a major
fraction of the total run time" (76).

*Difference.* Everything GridPACK does is kept: its parser, base case,
contingency rules, limit checks and writers produce every output. Added
are (a) a GPU path that solves the cases in batches and hands each solved
state back to a GridPACK network copy for reporting, and (b) changes to
GridPACK's own CPU path: rows built from numbers, a multi-rank file writer,
two corrections of order-dependent results, and DG and dc line models. The
second point matters: in stock GridPACK, results depended on which process
had run which case before (01-methodology §4.4); here every case is an
independent outage study.

### Zhou et al. 2017, "GPU-Based Batch LU-Factorization Solver"

*What it is.* Algorithm 2: factorize many matrices with one sparsity
pattern together, one thread block per column and one thread per matrix,
with the values of all matrices for each position stored side by side, so
memory reads are coalesced and threads never diverge (4976). Removed
elements are kept as stored zeros so every post-contingency matrix has the
pre-contingency pattern. On a K40 the time per 9,241-bus system levelled off
at 0.31–0.33 ms from batch 128, 76 times faster than KLU, with solutions
within 1e-14 of KLU (4977). The letter times linear systems only; it does
not describe the ordering, pivoting, the Newton loop or a full study.

*Difference.* Alg 2 is this algorithm, adopted faithfully. Around it this
work adds: the ordering and fixed pivots from one KLU plan (§5.4), the
full Newton loop with GridPACK's control sequence per case, a pattern that
also absorbs bus-type changes, slack moves and isolated buses (not just
removed branches), several warps per long column with bitwise-identical
factors, slot refilling, and the triangular solves on the GPU. On the GB10,
one factorization and solve of 512 ACTIVSg10k cases takes 110 ms, about
0.21 ms per case, the same efficiency class as Zhou's figure on newer
hardware.

### D'Orto et al. 2021, "Comparing Different Approaches"

*What it is.* A study of one Newton power flow on grids of 500 to 25,000
buses, comparing CPU, GPU and hybrid linear solvers. KLU with AMD ordering
gave the least fill at every size; the fastest variant analysed and
factorized the first iteration with KLU on the CPU and refactorized later
iterations on the GPU, keeping the triangular solve on the CPU, where it was
faster (56607–12). The linear solve took 81–93% of the CPU time. Gains were
9.6× and 13.1× at 25,000 buses against an Eigen-based CPU baseline.

*Difference.* The KLU and AMD analysis is adopted as the planner, but done
once per *study* (about 15,000 power flows on ACTIVSg10k) rather than once
per power flow. The refactorization is batched over cases. The solve moves
to the GPU, which reverses D'Orto's finding once many cases are solved at
once (2.2× faster whole studies on Polish). Their Jacobian kernels use one
thread per nonzero with atomic additions; here each bus sums its terms in
GridPACK's order without atomics, so results do not depend on thread timing.

### Wang, Wende-von Berg and Braun 2021, "Fast Parallel Newton-Raphson Power Flow Solver"

*What it is.* Batched Newton power flows for many grid states with the same
admittance pattern, on CPU (vectorized) and GPU, integrated with
pandapower. Batched sparse LU refactorization with fine-grained parallelism
for the serial final levels; failed cases get "a second chance" with KLU;
topology changes need a "costly" re-initialization, and discrete controls
need an external loop. GTX 1080; whole power flows on 300 and 2,869 buses,
refactorization timed up to 9,241 buses (0.138 ms per refactorization at
batch 2,048); more than 100× against pandapower.

*Difference.* Closest in spirit to the GPU engine. Three differences:
(1) Wang et al. assume one topology and one set of bus types per batch; the
superset pattern lets outages, slack moves, isolated buses and PV-to-PQ
switching share one plan, so N-1 cases need no re-initialization;
(2) reactive limits are enforced inside the batch, per case, with GridPACK's
deadband and controller limits, instead of by an outside loop;
(3) the extra threads for the final levels give bitwise-identical factors;
Wang et al.'s use atomic additions. The per-case CPU retry is the same idea
as their second chance, but through GridPACK's full solve.

### Zhou et al. 2016, "A Novel GPU-Accelerated Strategy for Contingency Screening"

*What it is.* GPU screening of contingencies with the dc (linear)
approximation in single precision, detecting islanding from a zero
denominator, on grids up to 8,503 buses; 47.6× against a single-threaded
laptop CPU (34–38). Screening targets the step that takes "about 30%" of a
study (38).

*Difference.* No screening and no linear approximation: every case gets a
full ac Newton solve in double precision, so no case can be missed by an
approximation. Islanding is decided from the network graph (bridges) and
handled with GridPACK's rules.

### W. Chen et al. 2022, GPU HELM

*What it is.* N-1 static security analysis with the holomorphic embedding
method on the GPU, with fine-grained parallelism and a GPU union-find for
islanding; islanded cases are dropped; about 10× against MATPOWER on a
2,869-bus grid; agreement with Newton of about 3e-5 (5–7).

*Difference.* Newton's method, as in GridPACK and in production tools, so
GPU and CPU answers agree to about 1e-12 per unit rather than 3e-5.
Islanded cases receive GridPACK's result instead of being dropped.

### Meng and Yun 2023, GPU N-1 with an improved fast decoupled method

*What it is.* Batched fast-decoupled ac power flow for N-1 on up to four
GPUs, with batch sizes set from GPU memory divided by the per-case
footprint; 11,062 outages of a 12,110-bus grid in 15.2 s; 29–53 iterations
per case (3–6).

*Difference.* Full Newton instead of fast decoupled (a median of two steps
per case, and at most six on the networks tested apart from one
ill-conditioned case, instead of 29–53), one GPU, and a memory budget that also accounts for
the CPU processes sharing the GPU's memory on a unified-memory machine. The
batch-size rule (memory, then a tested cap, then a measured throughput
sweep) extends theirs.

### Sai Nandini P et al. 2027, "HPC-accelerated parallel algorithms for large-scale contingency analysis"

*What it is.* N-1 and N-2 ranking with a composite performance index, using
the fast decoupled method in single precision, compared across sequential,
MPI, CUDA and hybrid MPI-CUDA versions on grids of 118, 246 and 2,383 buses.
Each GPU thread evaluates one whole contingency; the LU step uses cuBLAS's
dense batched routine (`cublasSgetrfBatched`, batch size 1 in their
Algorithm 6). MPI with a static split of the case list was fastest: about
19× over sequential for N-1 on the Polish 2,383-bus grid (5,877 s against
about 111,897 s for 3,223 cases); the CUDA version took about 88,494 s.

*Difference.* Here the GPU does sparse, not dense, factorization, with many
cases per factorization launch and one thread per case per matrix column
rather than one thread per whole case; cases are handed out dynamically,
not split statically; the solver is Newton in double precision with
reactive limits; and every case is reported through GridPACK's checks, not
reduced to an index. For scale: a full N-1 study of the Polish grid (4,198
outcomes) takes 4.3 s on the Alg 2 path at 16 ranks
(`docs/gpu_n1/performance.md`).

### Zhou, Cope, Foerster and Morstyn 2026, JAX batched ac power flow

*What it is.* Batched Newton power flow written in Python with JAX, using
an iterative linear solver (GMRES with a fast-decoupled preconditioner)
because JAX lacks batched sparse direct solvers; scenarios differ only in
load (±20%); grids up to 2,224 buses; 14.3× against the fastest
multi-threaded pandapower run on an H200; agreement with pandapower below
1e-10. The authors suggest cuDSS as future work.

*Difference.* A direct sparse solver (Alg 2 or cuDSS) instead of an
iterative one, contingencies (which change the network) instead of load
scenarios (which do not), grids up to 28,000 buses, and integration with an
existing production-style C++ code instead of a new Python solver.

### Kim and Kim 2022, GPU tracking of ac optimal power flow

*What it is.* A different problem: ac optimal power flow solved by ADMM
entirely on the GPU, tracking solutions over time with warm starts and
avoiding host-device transfers; up to 70,000 buses; 1.2–13.3× against
Ipopt (5–7).

*Difference.* Shared ideas only: warm starts from a solved case, and keeping
data on the GPU between steps. Here each Newton step still returns a few
numbers per case to the host to decide the next action; on the GB10's
shared memory this left no measurable GPU idle time.

### gpusim2grid (Grid2op, GitHub, 2026)

*What it is.* A GPU Newton power flow for batches of scenarios and N-k
contingencies, with cuDSS batched factorization and "one symbolic
factorization across the batch". Each batch runs a fixed number of Newton
steps with "no outer loop": it "does not enforce reactive-power limits (no
PV→PQ switching)" or adjust taps. Islanded cases are skipped, or only the
largest component is solved. It relies on lightsim2grid for set-up, and its
README reports no benchmark figures.

*Difference.* Convergence per case to a tolerance instead of a fixed step
count; reactive-limit switching, slack transfer and isolated buses inside
the batch; line-commutated dc lines solved by the sequential method;
GridPACK's limit checks and output tables; and the GridPACK CPU path for
cases the GPU cannot reproduce. gpusim2grid does solve controls this work
leaves to the CPU (remote voltage control and static var compensators when
seeded from lightsim2grid).

### ExaPF.jl (exanauts, GitHub)

*What it is.* A Julia power flow solver that runs fully on the GPU, with
automatic differentiation of the equations and Krylov iterative solvers
with an overlapping Schwarz preconditioner; its batch mode
(`BlockPolarForm`) solves several load scenarios of one network as a block
diagonal system, using cuDSS on the GPU. It is the modeller behind the
Argos.jl optimal power flow code.

*Difference.* ExaPF's batches vary loads on a fixed network; here each
batch member is a different outage of the network, with its own bus types
and slack, sharing one plan through the superset pattern. ExaPF has no
contingency driver, reactive-limit loop or reporting layer.

### EPRI 2025, prototype agentic AI platform for transmission planning

*What it is.* A project notice describing a platform in which a language
model calls engineering tools, such as power flow and contingency analysis,
as "agents", to shorten planning timelines (1–2). It sets no speed targets.

*Difference.* Not a solver. This work is a candidate tool for such a
platform: it keeps GridPACK's command line and input file, writes
machine-readable tables with one outcome row per case, and runs a full N-1
study of a 10,000-bus grid in well under a minute (03-results).

## 2. What is borrowed, extended and new

| Part | Relationship | Sources |
|---|---|---|
| Newton's method with GridPACK's equations and control sequence | Adopted | Kundur and Malik §6.4; Palmer et al. |
| Outaged elements as stored zeros; plan once | Adopted | Zhou et al. 2017 |
| KLU with AMD planning | Extended: once per study, not per power flow | D'Orto et al. |
| Algorithm 2 batched factorization | Adopted | Zhou et al. 2017 |
| Batched triangular solves on the GPU | Extended: measured to beat the CPU solve in batch form | Zhou et al. 2017; D'Orto et al. |
| Superset pattern covering bus-type changes, slack moves, isolated buses and dc outages | New combination of GridPACK's large-matrix layout and Zhou's stored zeros | GridPACK; Zhou et al. 2017 |
| Reactive-limit switching per case inside the batch | New placement of a standard rule | Kundur and Malik §6.4.2; gpusim2grid and Wang et al. leave it out |
| Several warps per long column with bitwise-identical factors | Extended | Wang et al. (with atomics) |
| Slot refilling across submissions | New in power flow | Named and left as future work by Huang and Dinavahi |
| Deterministic results end to end | Extended | cuDSS deterministic mode; no atomics in our kernels |
| Per-case health checks with a full GridPACK retry | Extended | Wang et al.'s second chance |
| Sequential ac/dc method inside a GPU batch; DG on loads | New placement of textbook methods | Khan and Bhowmick §1.7; Meena et al. |
| Reporting and file writing treated as a bottleneck | New as engineering; no algorithm | Palmer et al. 76 |
| Corrections of order-dependent stock results | New finding | — |

**Concurrent work.** Several of these ideas appeared in open-source tools
in the months before this code (5–7 October 2026): lightsim2grid 1.0–1.1
(bridge-based N-1 connectivity checks and reserved positions for PV-to-PQ
switching, on the CPU), gpusim2grid (cuDSS batches) and p3s (per-case
convergence masks). Claims of priority should therefore name the
combination above, not its individual parts.

## References

- Chen, W., et al. "GPU-Accelerated N-1 Static Security Analysis Based on
  Fine-Grained Parallelism HELM." *International Journal of Electrical
  Power & Energy Systems*, vol. 141, 2022, 108074.
- D'Orto, M., et al. "Comparing Different Approaches for Solving Large Scale
  Power-Flow Problems with the Newton-Raphson Method." *IEEE Access*, vol.
  9, 2021, pp. 56604–15.
- Electric Power Research Institute. *Building a Prototype Agentic
  Artificial Intelligence Platform for Transmission Planning*.
  Supplemental Project Notice 3002032336, 2025.
- exanauts. *ExaPF.jl*. GitHub, github.com/exanauts/ExaPF.jl, and its
  "Power flow: batch evaluation" tutorial. Accessed 8 Oct. 2026.
- Grid2op. *gpusim2grid*. GitHub, github.com/Grid2op/gpusim2grid. Accessed
  8 Oct. 2026.
- Huang, S., and V. Dinavahi. "Real-Time Contingency Analysis on Massively
  Parallel Architectures with Compensation Method." *IEEE Access*, vol. 6,
  2018, pp. 44519–30.
- Kim, Y., and K. Kim. "Accelerated Computation and Tracking of AC Optimal
  Power Flow Solutions Using GPUs." *ICPP Workshops '22*, ACM, 2022.
- Kundur, P. S., and O. P. Malik. *Power System Stability and Control*, 2nd
  ed., McGraw Hill, 2022, sec. 6.4.
- Meng, X., and Z. Yun. "GPU-Based N-1 AC Power Flow with an
  Improved-Convergence Fast Decoupled Method Considering Memory Limitations
  in Large Power Grids." *PowerCon 2023*, IEEE, 2023.
- Palmer, B., et al. "GridPACK: A Framework for Developing Power Grid
  Simulations on High Performance Computing Platforms." *WOLFHPC 2014*,
  IEEE, 2014, pp. 68–77.
- Sai Nandini P, et al. "HPC-Accelerated Parallel Algorithms for
  Large-Scale Power System Contingency Analysis." *Electric Power Systems
  Research*, vol. 265, 2027, 114205.
- Wang, Z., S. Wende-von Berg, and M. Braun. "Fast Parallel Newton-Raphson
  Power Flow Solver for Large Number of System Calculations with CPU and
  GPU." *Sustainable Energy, Grids and Networks*, vol. 27, 2021, 100483
  (arXiv:2101.02270).
- Zhou, G., et al. "A Novel GPU-Accelerated Strategy for Contingency
  Screening of Static Security Analysis." *International Journal of
  Electrical Power & Energy Systems*, vol. 83, 2016, pp. 33–39.
- Zhou, G., et al. "GPU-Based Batch LU-Factorization Solver for Concurrent
  Analysis of Massive Power Flows." *IEEE Transactions on Power Systems*,
  vol. 32, no. 6, 2017, pp. 4975–77.
- Zhou, Y., D. Cope, J. Foerster, and T. Morstyn. "JAX-Based Batched AC
  Power Flow for GPU Acceleration and AI Ecosystem Integration."
  arXiv:2605.14103, 2026.
