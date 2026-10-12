# Literature review: GPU acceleration of contingency analysis

This part reviews the published work that the system of
[02-methodology.md](02-methodology.md) builds on or is measured against. It
is organized by theme, because the sources answer different parts of one
question: how to solve many related power flows quickly and still report
what a planner needs. Sections 1–6 each take one theme and compare what the
sources do, what they measure and what they leave open. Section 7
synthesizes: which parts of this work are borrowed, which are extended,
which have no precedent in these sources, and how the gaps map to the
questions of [00-introduction.md](00-introduction.md). Appendix A keeps a
short note on each source, with what it is and how this work differs from
it.

**Sources.** The review covers the papers and repositories chosen at the
start of the work (Zhou et al. 2016 and 2017; D'Orto et al.; Wang, Wende-von
Berg and Braun; Meng and Yun; W. Chen et al.; Kim and Kim; Y. Zhou et al.;
Sai Nandini P et al.; Palmer et al.; EPRI; gpusim2grid; ExaPF.jl), the
textbooks the design relies on (Kundur and Malik, sec. 6.4; Khan and
Bhowmick, ch. 1), and sources found while building the system (Huang and
Dinavahi; KLU's documentation and the bridge-finding papers; NVIDIA's cuDSS
and CUDA documentation; concurrent open-source tools). Page or section
numbers follow each citation; full references are at the end.

**Thesis.** The published GPU work shows that many sparse linear systems
with one sparsity pattern can be factorized together far faster than one at
a time. It does not show how to keep a full N-1 study inside such a batch
when outages also move the slack, isolate buses and switch generators from
voltage control to fixed output, nor how much of a *whole study* that saves
once reading, reporting and writing results are counted. Most sources time
one kernel or one solver, on load scenarios or simplified methods, against
single-threaded CPU code.

**Comparability.** The times in this report are for whole studies, from the
network file to the final tables, against a 16-process CPU program (stock
GridPACK) on the same machine. Most cited speedups compare a GPU kernel or
solver with a single-threaded CPU program on different hardware. The two
kinds of figures are therefore not directly comparable, and the comparison
below is about method.

## 1. Where the time goes in a contingency study

The sources agree that solving linear systems dominates a *solver's* time,
and say little about a *study's*. D'Orto et al. measured that solving the
linear equations took 81–93% of a single Newton power flow's CPU time on
grids of 500 to 25,000 buses (Fig. 5, 56611), and cite an earlier figure of
about 85% for 3,493 buses (56607). W. Chen et al. state that solving the
sparse linear system and forming the Jacobian "account for 80% of N-1 SSA
computing time" (1), and Zhou et al. (2016) that "over 80% of the time is
spent on solving SLS" in ac power flow (33). Kim and Kim make the same
observation for optimization solvers, where factorization and triangular
solves take "more than 80%" of the time (1). These shares concern one solve;
none of these papers measures a whole contingency study.

Two sources look beyond the solver. Zhou et al. (2016) motivate screening by
noting that it "takes up to 30% of the total time of N-1 SSA" (34), a figure
they take from an earlier Chinese-language study rather than measure. Palmer
et al., describing GridPACK itself, warn that for power grid applications
"the cost of input, setup, initialization and output can be a major fraction
of the total run time", so that "reducing overall computational time
requires reduction in all phases of the computation" (76). Their evidence is
a single 777,646-bus power flow whose parsing and partitioning scaled poorly
(74).

This work measured the whole pipeline (03-results §3.2). In stock GridPACK
at 16 processes the Newton solves took 51–57% of a study on networks above
5,000 buses and building and writing the result rows 29–33%. Palmer et al.'s
warning therefore applies to contingency analysis too: removing the solve
alone could make a stock study at most about 2.3 times faster (03-results
§5.1). The literature's focus on the linear solve explains the size of its
kernel speedups; it does not predict a study's.

## 2. Solving many power flows at once

### 2.1 Batched factorization of one sparsity pattern

Zhou et al. (2017) set out the approach most later work follows. They
describe the existing single-matrix GPU method (their Algorithm 1: one
thread per column, columns grouped into dependency levels that run in
series) and name its three weaknesses for power-flow Jacobians: too little
parallelism, because a Jacobian of "less than 100 000 order" is small and
"the number of executable columns decreases rapidly with increasing level
number"; uncoalesced memory access; and warp divergence in the update loop
(4976). Their Algorithm 2 factorizes N matrices with one pattern together:
"thread block j for column j", thread *t* updating column *j* of matrix *t*,
with "the data of same numbered columns ... stored at the contiguous address
of GPU device memory", so every warp reads consecutive addresses and, the
patterns being identical, "will always execute the same branch and never
diverge" (4976). Reordering and symbolic analysis are done "only once" for
all matrices, and the same framework "can be applied to forward and backward
substitution ... directly" (4976). On a K40 with double precision, the time
per 9,241-bus linear system fell from 18.9 ms at batch 1 to 0.33 ms at batch
128 and 0.31 ms from batch 512, 76 times faster than KLU (25 ms), with
solutions within 1e-14 of KLU's (Tables I–II, 4977). The letter times linear
systems only. It does not name the ordering, does not mention pivoting, and
does not describe a Newton loop or a study.

Wang, Wende-von Berg and Braun built a whole batched Newton solver on the
same layout: batch data "aligned on the same element within the batch ...
and stored in contiguous memory address", which on the GPU "guarantees
automatically coalesced memory access" (4). They found that a batch of 512
gives "less than 1 active warps per SM" on a GTX 1080 in the final levels,
where most of the work sits in levels of one or two columns, and added two
further stages: partial factorization of columns that are not yet ready,
and, for a single remaining column, extra parallelism by the threads of one
block (7–8). They use atomic additions only in the batched forward and
backward substitution (8). On case9241pegase, one batched refactorization
took 0.138 ms per system at batch 2,048, against 0.238 ms for the plain
batch method (9, 11). Their benchmarks are profile simulations in which
every power flow has the *same* admittance matrix (Table 1, 11).

Other sources batch differently or not at all:

- **Meng and Yun** batch the fast decoupled method's two matrices with
  Zhou's batch LU, and factorize them once per case because their improved
  B̄′ uses the pre-outage angles (3–4).
- **gpusim2grid** uses cuDSS's uniform batch: "The cuDSS symbolic
  factorization is computed **once** and reused across the whole batch —
  only numeric refactorization/solve happen per iteration" (README), with
  64-bit or 32-bit values chosen at build time.
- **ExaPF.jl** solves several load scenarios of one network as one block
  diagonal system, either with cuDSS's uniform batch (one analysis for all
  blocks) or with Krylov solvers and a block-Jacobi preconditioner
  (documentation; `ExaPFCUDAExt`).
- **Huang and Dinavahi** avoid refactorization altogether: they factorize
  the fast decoupled method's B′ and B″ once for the base case and treat
  each outage as a rank-1 or rank-2 compensation update, with the triangular
  solves for many cases done as one sparse solve with many right-hand sides
  (sec. II–III, 44524).
- **Y. Zhou et al.** use an iterative solver (GMRES with a fast-decoupled
  preconditioner) "as efficient batched sparse direct solvers are currently
  limited in JAX", and name cuDSS's batched modes as future work (1, 3).
- **Sai Nandini P et al.** call a *dense* batched LU (`cublasSgetrfBatched`)
  "with a batch size of 1" inside one-thread-per-contingency kernels
  (Algorithm 6, 8); their fastest version is CPU-only MPI.
- **W. Chen et al.** batch the Padé systems of the buses within one power
  flow, and loop over contingencies one at a time (5–6).

The common thread is that one sparsity pattern, analysed once, lets many
factorizations share one plan. The sources differ in what varies across the
batch: load scenarios (Wang, ExaPF, Y. Zhou), or outages that change only
admittances (Zhou 2017, Meng). This work adopts Algorithm 2 as one of its
two factorization methods and NVIDIA's cuDSS uniform batch as the other
(02-methodology §6–7).

### 2.2 Ordering and analysis once

Zhou et al. (2017) analyse once but do not say how. D'Orto et al. compared
orderings for one Newton power flow and found that KLU with AMD ordering
"requests minimum filled in non-zero elements" at every size, for example
696,576 factor entries for a 25,000-bus Jacobian with 314,922 entries,
against 817,906 for Gilbert–Peierls with METIS and 1,210,501 for KLU with
COLAMD (Table 1, 56610). They describe KLU's refactorization: since the
Jacobian's pattern is the same at every iteration, later iterations skip the
block-triangular and ordering steps and use a left-looking factorization
"which does not perform partial pivoting", because "the non-zero patterns of
L and U are already known" (56607). Their fastest variant analysed and
factorized the first iteration with KLU on the CPU and refactorized later
iterations on the GPU with GLU (56613). Wang et al. also order with AMD
inside KLU, citing its "good performance in reducing fill-ins" on circuit
and power-flow matrices, and note that refactorization works by "presuming
the numerical stability" of the first factorization's permutations (6).

This work takes D'Orto's analysis step and applies it once per *study* on a
pattern every case shares (02-methodology §5.4), so that its cost is spread
over thousands of power flows instead of one. Wang's caveat about presumed
stability is the reason for the per-case health checks of 02-methodology
§5.5.

### 2.3 Where the triangular solve runs

D'Orto et al. kept the triangular solve on the CPU because, for one power
flow, "LU solver on CPU has best performance on solve" (Tables 4–5, 56612);
at 25,000 buses on the P100 node KLU solved in 1.658 ms against 20.06 ms for
GLU on the GPU. Their reason is empirical; they give no mechanism. Zhou et
al. (2017) state that batching applies to substitution directly (4976), and
Wang et al. batch it on the GPU (8). With many cases per launch the balance
reverses: moving the solve to the host made a whole Polish study 2.2 times
slower in this work (02-methodology §6).

## 3. Keeping one structure when the network changes

A batch can share one plan only while every member has the same sparsity
pattern. The sources handle network changes in four ways.

1. **Keep removed entries.** Zhou et al. (2017) make the post-contingency
   *admittance* matrix match the pre-contingency pattern "at the cost of
   adding 2x redundant nonzeros" for x outages (4976): an outaged branch's
   entries stay in the structure with value zero. The Jacobian then keeps
   its pattern as long as no bus changes type.
2. **Re-initialize.** Wang et al. state that "for simulations with topology
   changes ... the extra initialization (costly ...) needs to be performed
   for each new topology" (5).
3. **Vary loads only.** ExaPF.jl and Y. Zhou et al. batch load scenarios on
   one network ("±20% random load multipliers", Y. Zhou 3), so the question
   does not arise.
4. **Drop or skip what changes.** W. Chen et al. test connectivity with a
   union-find on the GPU, and an outage that splits the system "will not be
   an element" of the set they solve (6). Zhou et al. (2016) detect
   separation from a zero denominator and send such cases to the critical
   set directly (34–35). gpusim2grid patches each case's admittance matrix
   ("contingencies subtract the tripped-branch admittances"), finds
   splitting outages with a bridge search before batching ("Disconnected
   contingencies never enter the batch"), and can instead solve only the
   largest component (README; source).

ExaPF.jl's source also builds batches of line outages for
security-constrained optimal power flow, one block per outage, but its
documented batch mode varies only loads (`contingencies.jl`; tutorial).

Bus types are the part none of the batched sources keeps inside one pattern.
Kundur and Malik note that for a PV bus "terms corresponding to ΔQ and ΔV
would be absent" from the Newton equations (sec. 6.4.3.1), so in the usual
formulation a generator that reaches its reactive limit, a slack that moves
or a bus that becomes isolated changes the Jacobian's dimension. GridPACK's
optional large-matrix layout, in its source code rather than in Palmer et
al.'s paper, already gives PV buses a fixed-voltage row and the slack an
identity block. This work combines that layout with Zhou's kept entries into
a superset pattern in which all of these changes are changes of values
(02-methodology §5.3).

## 4. Solution method, precision and controls

### 4.1 Newton's method against the alternatives

Kundur and Malik describe Newton's method as "particularly suited for
applications involving large systems requiring very accurate solutions", and
note that for "very large angles across lines and with special control
devices ... full N-R formulation may be required" instead of the fast
decoupled method (sec. 6.4.5). D'Orto et al. add that Newton's method "is in
general more robust ... when it comes to heavily loaded systems" (56606).
The sources that chose otherwise show the trade-offs:

- **Fast decoupled.** Meng and Yun use it because the Newton method
  "requires more memory space" per power flow in batch mode (1), and show
  its weakness on their own data: on the 39-bus case its iterations rise
  from 10 to 202 as load increases (Table I, 2). Their improved version
  needed at most 29–33 iterations over the N-1 sets, against 40–53 for the
  standard version (Table II). Sai Nandini P et al. also use it, and
  acknowledge that it weakens under "heavily loaded, weak-grid,
  renewable-rich, or high R/X ratio conditions" (4).
- **Holomorphic embedding.** W. Chen et al. argue that iterative methods
  have "a risk of non-convergence and convergence to a meaningless solution"
  (1) and use HELM instead; their solutions differ from Newton's by up to
  3.42e-5 by a summed metric that is not a per-bus maximum (Eq. 18, Table 2,
  7).
- **dc approximation.** Zhou et al. (2016) screen with the linear dc model
  and check only branch active power, since screening "focus[es] more on the
  branch thermal violation" (34).
- **Optimization.** Kim and Kim solve ac optimal power flow by ADMM, a
  different problem, to residuals of 1e-2 to 1e-1 (Table 2, 6).

This work keeps Newton's method, as GridPACK and production tools do, so
that GPU results can be checked against GridPACK's to rounding level. The
converged cases needed a median of two steps (03-results §4.1).

### 4.2 Precision

Zhou et al. (2017), Wang et al. and Y. Zhou et al. use double precision;
Wang et al. explain that refactorization has a "high numerical stability
requirement" (8). Zhou et al. (2016) and Sai Nandini P et al. use single
precision (36; 9). Both GPU paths here are double precision throughout,
without fast-math compiler options (02-methodology §6).

### 4.3 Reactive limits and other controls

Reactive-limit switching is where the sources say least. Kundur and Malik
give the rule in their Gauss–Seidel procedure: if a generator's computed Q
"exceeds either the maximum or minimum limit, it is set equal to the limit"
and the bus is treated as a PQ node (sec. 6.4.2(b)); their Newton sections
do not discuss limits, and no section describes switching back. Among the
GPU sources:

- gpusim2grid "does not enforce reactive-power limits (no PV→PQ switching)"
  and runs a fixed number of Newton steps with "no outer loop" (README);
  lightsim2grid, which seeds it, also "does not enforce reactive power
  limits on generators" (disclaimer), and ExaPF.jl has no PV-to-PQ switching
  in its Newton loop (source);
- Wang et al. handle discrete controls (tap changers, shunts) with "an
  external control logic" that "might lead to extra loops" (5), and do not
  mention reactive limits;
- Y. Zhou et al. keep the PV set fixed (1);
- W. Chen et al.'s flow chart has a "Check generators reactive power limits"
  block, but the text does not describe the rule and the cases are solved
  one at a time (Fig. 1, 3);
- Meng and Yun, Sai Nandini P et al. and Zhou et al. (2016) do not address
  them.

GridPACK applies Kundur's rule between Newton solves, with a deadband and a
pass limit, and its contingency driver adds a second solve when a bus
switched. This work runs that sequence per case inside the GPU batch
(02-methodology §5.5). Controls that GridPACK applies with network-wide
logic (switched shunts, tap changers, area interchange) stay on the CPU.

For dc lines, Khan and Bhowmick describe two Newton-based approaches: the
unified method, which solves ac and dc quantities together and "increases
the size of the Jacobian matrix", and the sequential method, which solves
them separately and couples them "by injecting an equivalent amount of real
and reactive power at the terminal AC buses", being "easier to implement"
with a smaller Jacobian (sec. 1.7.1–1.7.2, 25). The sequential method keeps
the Jacobian's pattern unchanged, which is why it suits a batch
(02-methodology §4.5).

### 4.4 Reported accuracy

Zhou et al. (2017) report linear solutions within 1e-14 of KLU (4977); Y.
Zhou et al. agreement with pandapower below 1e-10 (3); W. Chen et al. about
3e-5 against Newton by their own metric (7); Sai Nandini P et al. up to
1.98e-4 per unit against Newton on the 118-bus case (Table 6, 9). None of
the batched sources compares a whole study's outputs with a production
tool's. This work compared every GPU case with a GridPACK re-solve in 70
validation studies (largest voltage difference 4.8e-12 per unit) and every
row of every table in the test matrix (03-results §4).

## 5. Uneven convergence and keeping the GPU busy

Cases in a batch converge after different numbers of steps; on ACTIVSg10k
the GPU cases needed from zero to eight (03-results §3.6). The sources
respond in three ways.

1. **Fixed step counts.** gpusim2grid runs every case for the same number of
   steps (four by default, "no convergence check on GPU"), pads the last
   batch with empty "phantom slots", and tests convergence afterwards from
   the final residual (source). D'Orto et al. ran "all cases in fixed
   iterations (6)" for comparability (56611). A fixed count wastes steps on
   fast cases and cannot confirm convergence of slow ones. ExaPF.jl's batch
   Newton stops on one norm of the whole stacked residual, so every scenario
   runs until the slowest converges (source).
2. **Masks.** Converged members are left out of later steps, so their slots
   idle until the batch's slowest member finishes; p3s skips converged
   systems this way (README). Huang and Dinavahi considered filling idle
   slots with new cases and rejected it: "if new tasks ... are assigned to
   these idle threads, coalesced access might be destroyed since the
   scenario number is not successive", so "a group of scenarios will
   terminate only if all of them finished". They estimated that refilling
   could improve efficiency "up to 40%" if the longest scenario takes 80%
   more iterations than the shortest, and left "full utilization of
   computational resource with coalesced access" to future work (44524).
3. **Fallback.** Wang et al. give a case that fails during refactorization
   "a second chance to get fixed with the KLU numerical factorization" (5).

Batch size is the other lever. Zhou et al. (2017) saw time per system level
off between batch 128 and 512 depending on the grid (Table I, 4977), and
Wang et al. "around 2000 calculations" (12). Meng and Yun size batches by
dividing GPU memory by the memory one power flow needs, counted from the
host program's variables (3).

This work refills a slot as soon as its case finishes, from the next
submission if needed (02-methodology §5.6). The layout is by slot, not by
case number, so a new case in an old slot keeps the coalesced access Huang
and Dinavahi feared to lose. On ACTIVSg10k refilling and light steps raised
slot occupancy from 40% to 72% and shortened the study by 11%
(02-methodology §9), below the 40% Huang and Dinavahi gave as the most
refilling could gain. Batch size is capped by memory, as in Meng and Yun, by
a tested limit, and by a measured throughput sweep.

## 6. From solver to study: integration and use

Most of the GPU solvers in these sources are standalone. Y. Zhou et al.
integrate with the JAX ecosystem; ExaPF.jl is the modeller behind the
Argos.jl optimal power flow code; gpusim2grid relies on lightsim2grid for
set-up; Kim and Kim keep the whole ADMM solve on the GPU "without any data
transfer between CPUs and GPUs" and warm-start each period from the last
(4–6).

GridPACK (Palmer et al.) is a framework of distributed network objects,
mappers that assemble matrices from per-bus and per-branch blocks, PETSc
solvers chosen by run-time options, and a task manager that hands
independent jobs to processes (69–73). Contingency analysis is the example
it gives of applications that "run many independent parallel tasks" (73),
but the paper does not describe the contingency application or time it. This
work keeps GridPACK as the host: its parser, base case, contingency rules,
limit checks and writers produce every output, and the GPU only solves
(02-methodology §5).

The baselines in the literature are mostly single-threaded: a laptop CPU in
Zhou et al. (2016, 36), single KLU solves in Zhou et al. (2017), single-
threaded pandapower in Wang et al. (with a 16-times bound for perfect
six-core parallelism, 12), and a sequential program in Sai Nandini P et al.,
whose own MPI version was 19 times faster than it (Table 9). Y. Zhou et al.
compare with the fastest of 1-, 8- and 16-thread pandapower runs (3). This
work compares with a 16-process production program on the same machine.

EPRI's project notice proposes a platform in which language models act as
"the top-level controller while engineering analysis tools can act as an
'agent'", with interfaces to "common power flow, dynamic stability,
protection studies, and production cost modeling tools" (1–2). It sets no
speed targets and does not mention contingency analysis. A contingency tool
that keeps GridPACK's command line, input file and machine-readable tables
fits such a platform's interface.

## 7. Synthesis

### 7.1 What is borrowed, extended and new

| Part | Relationship | Sources |
|---|---|---|
| Newton's method with GridPACK's equations and control sequence | Adopted | Kundur and Malik sec. 6.4; GridPACK |
| Outaged elements kept in the structure; analysis once | Adopted | Zhou et al. 2017 |
| KLU with AMD planning | Extended: once per study, not once per power flow | D'Orto et al.; Davis and Palamadai Natarajan |
| Algorithm 2 batched factorization | Adopted | Zhou et al. 2017 |
| cuDSS uniform batch | Adopted, with the planner's ordering and deterministic mode | NVIDIA |
| Batched triangular solves on the GPU | Extended: measured to beat the CPU solve in batch form | Zhou et al. 2017; Wang et al.; against D'Orto et al. |
| Superset pattern covering bus-type changes, slack moves, isolated buses and dc outages | New combination of GridPACK's large-matrix layout and Zhou's kept entries | GridPACK; Zhou et al. 2017 |
| Reactive-limit switching per case inside the batch | New placement of a standard rule | Kundur and Malik sec. 6.4.2(b); left out by gpusim2grid, Wang et al., Y. Zhou et al. |
| Several warps per long column with bitwise-identical factors | Extended | Wang et al.'s stage 3 (threads of one block on a single column) |
| Slot refilling across submissions | New in power flow | Named and left as future work by Huang and Dinavahi |
| Deterministic results end to end | Extended | cuDSS deterministic mode; no atomic additions in this work's kernels |
| Per-case health checks with a full GridPACK retry | Extended | Wang et al.'s second chance |
| Sequential ac/dc method inside a GPU batch; DG on loads | New placement of textbook methods | Khan and Bhowmick sec. 1.7.2; Meena et al. |
| Reporting and file writing treated as a bottleneck | New as engineering; no algorithm | Palmer et al. 76 |
| Corrections of order-dependent stock results | New finding | — |

### 7.2 Concurrent work

Several of these ideas appeared in open-source tools in the months before
this code (5–7 October 2026):

- **gpusim2grid** (repository created 28 June 2026): cuDSS uniform batches
  for N-k cases; a bridge search for splitting outages since 10 September
  2026.
- **lightsim2grid 1.1.0** (21 September 2026, CPU): N-1 connectivity decided
  "with a single graph search instead of one per contingency" (bridges,
  added 10 September 2026), and a reserved slot so that "a generator
  contingency switching a bus PV -> PQ costs no extra symbolic
  factorization". This covers generator outages, not reactive-limit
  switching, which lightsim2grid does not enforce.
- **p3s** (Fraunhofer IEE, released 28 August 2026): a GPU Newton solver
  with cuDSS and "per-case convergence and per-bus served masks" (8
  September 2026), with masking rather than refilling.

The bridge pre-pass and the reserved positions therefore have close
concurrent counterparts, and claims of priority should name the combination
above, not its individual parts. The same months also brought three reports
of defects in cuDSS 0.8.0's uniform batch on power-flow Jacobians, which
NVIDIA acknowledged (02-methodology §7).

### 7.3 Open questions this work addresses

The review leaves four gaps, which map to the questions of 00-introduction
§2:

1. No batched source keeps bus-type changes, slack moves and isolated buses
   inside one plan (§3), the subject of question M1.
2. No batched source runs a per-case reactive-limit sequence with a
   production tool's rules (§4.3), the subject of question M2.
3. No source times a whole contingency study against a multi-process
   production program on the same machine (§1, §6), the subject of questions
   1–3.
4. No batched source compares a study's full outputs with a production
   tool's (§4.4), the subject of question 4.

## Appendix A. Notes on each source

Each note says what the source is and how this work differs from it.

### Palmer et al. 2014, GridPACK

*What it is.* The framework this work is built on: distributed network
objects, "mappers" that build matrices from per-bus and per-branch
contributions of "on the order of 1 × 1 or 2 × 2" blocks, PETSc solvers
chosen by run-time options, communication through MPI and Global Arrays, and
a task manager that hands independent jobs to processes or processor groups
(69–73). The paper only mentions contingency analysis; GridPACK's
contingency application, in its code, solves one case per process. The
authors note that setup and input/output can be "a major fraction of the
total run time" (76).

*Difference.* Everything GridPACK does is kept: its parser, base case,
contingency rules, limit checks and writers produce every output. Added are
(a) a GPU path that solves the cases in batches and hands each solved state
back to a GridPACK network copy for reporting, and (b) changes to GridPACK's
own CPU path: rows built from numbers, a multi-rank file writer, two
corrections of order-dependent results, and DG and dc line models. The
second point matters: in stock GridPACK, results depended on which process
had run which case before (02-methodology §4.4); here every case is an
independent outage study.

### Zhou et al. 2017, "GPU-Based Batch LU-Factorization Solver"

*What it is.* Algorithm 2: factorize many matrices with one sparsity pattern
together, one thread block per column and one thread per matrix, with the
values of all matrices for each position stored side by side, so memory
reads are coalesced and threads never diverge (4976). Removed elements are
kept as redundant entries so every post-contingency matrix has the
pre-contingency pattern. On a K40 the time per 9,241-bus system levelled off
at 0.31–0.33 ms from batch 128, 76 times faster than KLU, with solutions
within 1e-14 of KLU (4977). The letter times linear systems only; it does
not describe the ordering, pivoting, the Newton loop or a full study.

*Difference.* Alg 2 is this algorithm, adopted faithfully. Around it this
work adds: the ordering and fixed pivots from one KLU plan (02-methodology
§5.4), the full Newton loop with GridPACK's control sequence per case, a
pattern that also absorbs bus-type changes, slack moves and isolated buses
(not just removed branches), several warps per long column with
bitwise-identical factors, slot refilling, and the triangular solves on the
GPU. On the GB10, one factorization and solve of 512 ACTIVSg10k cases takes
110 ms, about 0.21 ms per case, the same efficiency class as Zhou's figure
on newer hardware.

### D'Orto et al. 2021, "Comparing Different Approaches"

*What it is.* A study of one Newton power flow on grids of 500 to 25,000
buses, run for a fixed six iterations, comparing CPU, GPU and hybrid linear
solvers. KLU with AMD ordering gave the least fill at every size; the
fastest variant analysed and factorized the first iteration with KLU on the
CPU and refactorized later iterations on the GPU with GLU, keeping the
triangular solve on the CPU, where it was faster (56607–13). The linear
solve took 81–93% of the CPU time. Gains were 9.6× (P100 node) and 13.1×
(V100 node) at 25,000 buses against an Eigen-based CPU baseline (abstract;
the text credits the 9.6× bar to a different variant than its figure). Their
mismatch kernel uses one thread per admittance entry with atomic additions;
their Jacobian kernels write each entry once (Listings 1–2).

*Difference.* The KLU and AMD analysis is adopted as the planner, but done
once per *study* (about 15,000 power flows on ACTIVSg10k) rather than once
per power flow. The refactorization is batched over cases. The solve moves
to the GPU, which reverses D'Orto's finding once many cases are solved at
once (2.2× faster whole studies on Polish). Here each bus sums its terms in
GridPACK's order without atomic additions, so results do not depend on
thread timing.

### Wang, Wende-von Berg and Braun 2021, "Fast Parallel Newton-Raphson Power Flow Solver"

*What it is.* Batched Newton power flows for many grid states with the same
admittance pattern, on CPU (vectorized) and GPU, integrated with pandapower.
Batched sparse LU refactorization with KLU's AMD ordering, and two extra
stages for the serial final levels; atomic additions in the batched
substitution; failed cases get "a second chance" with KLU; topology changes
need a "costly" re-initialization, and discrete controls need an external
loop. GTX 1080; whole power flows on up to 9,241 buses with one admittance
matrix per benchmark (0.138 ms per refactorization at batch 2,048); more
than 100× against single-threaded pandapower.

*Difference.* Closest in spirit to the GPU engine. Four differences: (1)
Wang et al. assume one topology and one set of bus types per batch; the
superset pattern lets outages, slack moves, isolated buses and PV-to-PQ
switching share one plan, so N-1 cases need no re-initialization; (2)
reactive limits are enforced inside the batch, per case, with GridPACK's
deadband and controller limits, instead of not at all; (3) the extra threads
for the final levels spread one column over up to 32 warps per case and give
factors bitwise identical to one thread per case; (4) the triangular solves
use no atomic additions, so results do not depend on thread timing. The
per-case CPU retry is the same idea as their second chance, but through
GridPACK's full solve.

### Zhou et al. 2016, "A Novel GPU-Accelerated Strategy for Contingency Screening"

*What it is.* GPU screening of contingencies with the dc (linear)
approximation in single precision, from a precomputed dense reactance
matrix, detecting islanding from a zero denominator, on grids up to 8,503
buses; 47.6× against a single-threaded program on a laptop CPU (Table 5,
38). Screening targets a step said to take "up to 30%" of a study (34).

*Difference.* No screening and no linear approximation: every case gets a
full ac Newton solve in double precision, so no case can be missed by an
approximation. Islanding is decided from the network graph (bridges) and
handled with GridPACK's rules.

### W. Chen et al. 2022, GPU HELM

*What it is.* N-1 static security analysis with the holomorphic embedding
method on the GPU, with fine-grained parallelism inside each power flow and
a GPU union-find for islanding; islanded cases are dropped; cases are solved
one at a time; about 10× against MATPOWER on a 2,869-bus grid (read from
Fig. 10); agreement with Newton of about 3e-5 by a summed metric (5–7).

*Difference.* Newton's method, as in GridPACK and in production tools, so
GPU and CPU answers agree to about 1e-12 per unit rather than 3e-5. Cases
are batched across contingencies, not parallelized within one. Islanded
cases receive GridPACK's result instead of being dropped.

### Meng and Yun 2023, GPU N-1 with an improved fast decoupled method

*What it is.* Batched fast-decoupled ac power flow for N-1 with Zhou's batch
LU, on four GPUs, with batch sizes set from GPU memory divided by the
per-case footprint; 11,062 outages of a combined 12,110-bus grid in 15.2 s;
at most 29–33 iterations per case with their improvement (40–53 without)
(3–6).

*Difference.* Full Newton instead of fast decoupled (a median of two steps
per case, and at most six on the networks tested apart from one
ill-conditioned case, instead of 29–33), one GPU, and a memory budget that
also accounts for the CPU processes sharing the GPU's memory on a
unified-memory machine. The batch-size rule (memory, then a tested cap, then
a measured throughput sweep) extends theirs.

### Sai Nandini P et al. 2027, "HPC-accelerated parallel algorithms for large-scale contingency analysis"

*What it is.* N-1 and N-2 ranking with a composite performance index, using
the fast decoupled method in single precision, compared across sequential,
MPI, CUDA and hybrid MPI-CUDA versions on grids of 118, 246 and 2,383 buses.
Each GPU thread evaluates one whole contingency; the LU step uses cuBLAS's
dense batched routine (`cublasSgetrfBatched`, batch size 1 in their
Algorithm 6). MPI with a static split of the case list was fastest: about
19× over sequential for N-1 on the Polish 2,383-bus grid (5,877 s against
about 111,897 s for 3,223 cases); the CUDA version took about 88,494 s
(Table 7, 10). Islanding and reactive limits are not discussed.

*Difference.* Here the GPU does sparse, not dense, factorization, with many
cases per factorization launch and one thread per case per matrix column
rather than one thread per whole case; cases are handed out dynamically, not
split statically; the solver is Newton in double precision with reactive
limits; and every case is reported through GridPACK's checks, not reduced to
an index. For scale: a full N-1 study of the Polish grid (4,198 outcomes)
takes 4.3 s on the Alg 2 path at 16 ranks (`docs/gpu_n1/performance.md`).

### Y. Zhou, Cope, Foerster and Morstyn 2026, JAX batched ac power flow

*What it is.* Batched Newton power flow written in Python with JAX, using an
iterative linear solver (GMRES with a fast-decoupled preconditioner) because
JAX lacks batched sparse direct solvers; scenarios differ only in load
(±20%); transmission grids up to 2,224 buses; 14.3× against the fastest of
1-, 8- and 16-thread pandapower runs on an H200 at batch 1,000; agreement
with pandapower below 1e-10. The authors suggest cuDSS as future work.

*Difference.* A direct sparse solver (Alg 2 or cuDSS) instead of an
iterative one, contingencies (which change the network) instead of load
scenarios (which do not), grids up to 29,637 buses, reactive-limit
switching, and integration with an existing production-style C++ code
instead of a new Python solver.

### Kim and Kim 2022, GPU tracking of ac optimal power flow

*What it is.* A different problem: ac optimal power flow solved by ADMM
entirely on the GPU, tracking solutions over time with warm starts and
avoiding host-device transfers; up to 70,000 buses; 1.2–13.3× against Ipopt
from a cold start (Table 2, 6).

*Difference.* Shared ideas only: warm starts from a solved case, and keeping
data on the GPU between steps. Here each Newton step still returns a few
numbers per case to the host to decide the next action; on the GB10's shared
memory this left no measurable GPU idle time.

### Huang and Dinavahi 2018, real-time contingency analysis with the compensation method

*What it is.* N-1 line-outage analysis on one or two GTX 1080 GPUs with the
compensation method applied to fast decoupled power flow: B′ and B″ are
factorized once (AMD ordering) and each outage is a rank-1 or rank-2 update,
with the triangular solves done by cuSPARSE for many cases at once; each
scenario is given to one warp, and one flag per iteration returns to the
host. On grids of 300 to 13,659 buses, one GPU averaged 0.9–5.8 ms per
scenario, 2.8–38.4 times faster than one CPU thread (Tables 3, 5; 44526–28).
The authors discuss refilling threads freed by converged scenarios and
reject it to keep coalesced access, estimating a possible gain of up to 40%,
and leave it to future work (44524). Islanding and reactive limits are not
discussed.

*Difference.* Full Newton instead of fast decoupled with compensation: each
case is factorized on its own in the batch, which keeps Newton's convergence
with the case's own Jacobian and handles any outage the superset pattern
covers, not only rank-limited updates. Slots are refilled as cases finish;
because memory is laid out by slot, not by case number, a new case in an old
slot keeps the coalesced layout.

### gpusim2grid (Grid2op, GitHub, 2026)

*What it is.* A GPU Newton power flow for batches of scenarios and N-k
contingencies (research code from RTE, repository created June 2026), with
cuDSS uniform-batch factorization and "one symbolic factorization across the
batch". Each case's admittance matrix is patched in place, and each batch
runs a fixed number of Newton steps (four by default) with "no outer loop":
it "does not enforce reactive-power limits (no PV→PQ switching)" or adjust
taps. Splitting outages are found by a bridge search and left out, or only
the largest component is solved. It relies on lightsim2grid for set-up and
its base case, and its README reports no benchmark figures. Its roadmap
lists reusing lightsim2grid's KLU ordering in cuDSS, which it does not yet
do.

*Difference.* Convergence per case to a tolerance instead of a fixed step
count; reactive-limit switching, slack transfer and isolated buses inside
the batch; line-commutated dc lines solved by the sequential method;
GridPACK's limit checks and output tables; and the GridPACK CPU path for
cases the GPU cannot reproduce. gpusim2grid does solve controls this work
leaves to the CPU (remote voltage control and static var compensators when
seeded from lightsim2grid).

### ExaPF.jl (exanauts, GitHub)

*What it is.* A Julia power flow solver that runs fully on the GPU, with
automatic differentiation of the equations and either cuDSS (uniform batch
since version 0.12) or Krylov solvers with a block-Jacobi preconditioner
(the README calls it overlapping Schwarz); its batch mode (`BlockPolarForm`)
solves several load scenarios of one network as a block diagonal system,
stopping on one norm of the whole batch's residual. Its source also builds
batches of line outages for security-constrained optimal power flow. It is
the modeller behind the Argos.jl optimal power flow code.

*Difference.* ExaPF's documented batches vary loads on a fixed network; here
each batch member is a different outage of the network, with its own bus
types and slack, sharing one plan through the superset pattern, and each
converges on its own test. ExaPF has no contingency driver, reactive-limit
loop or reporting layer.

### EPRI 2025, prototype agentic AI platform for transmission planning

*What it is.* A two-page project notice proposing a platform in which
language models act as the controller and engineering tools (power flow,
dynamic stability, protection and production cost tools) as callable
"agents", to shorten planning work (1–2). It reports no results, sets no
speed targets and does not mention contingency analysis.

*Difference.* Not a solver. This work is a candidate tool for such a
platform: it keeps GridPACK's command line and input file, writes
machine-readable tables with one outcome row per case, and runs a full N-1
study of a 10,000-bus grid in well under a minute (03-results §3.1).

## References

- Chen, W., J. Xu, and K. Wang. "GPU-Accelerated N-1 Static Security
  Analysis Based on Fine-Grained Parallelism HELM." *International Journal
  of Electrical Power & Energy Systems*, vol. 141, 2022, 108074.
- Davis, T. A., and E. Palamadai Natarajan. "Algorithm 907: KLU, a Direct
  Sparse Solver for Circuit Simulation Problems." *ACM Transactions on
  Mathematical Software*, vol. 37, no. 3, 2010, article 36.
- Donnot, B. *lightsim2grid*, release 1.1.0. GitHub,
  github.com/Grid2op/lightsim2grid, 21 Sept. 2026.
- D'Orto, M., S. Sjöblom, L. S. Chien, L. Axner, and J. Gong. "Comparing
  Different Approaches for Solving Large Scale Power-Flow Problems with the
  Newton-Raphson Method." *IEEE Access*, vol. 9, 2021, pp. 56604–15.
- Electric Power Research Institute. *Building a Prototype Agentic
  Artificial Intelligence Platform for Transmission Planning*. Supplemental
  Project Notice 3002032336, Apr. 2025.
- e2nIEE (Fraunhofer IEE). *p3s: parallel-pandapower-solver*. GitHub,
  github.com/e2nIEE/parallel-pandapower-solver. Accessed 9 Oct. 2026.
- exanauts. *ExaPF.jl*. GitHub, github.com/exanauts/ExaPF.jl, and its "Power
  flow: batch evaluation" tutorial. Accessed 8 Oct. 2026.
- Grid2op. *gpusim2grid*. GitHub, github.com/Grid2op/gpusim2grid. Accessed 8
  Oct. 2026.
- Huang, S., and V. Dinavahi. "Real-Time Contingency Analysis on Massively
  Parallel Architectures with Compensation Method." *IEEE Access*, vol. 6,
  2018, pp. 44519–30.
- Khan, S., and S. Bhowmick. *Power-Flow Modelling of HVDC Transmission
  Systems*. CRC Press, 2023, ch. 1.
- Kim, Y., and K. Kim. "Accelerated Computation and Tracking of AC Optimal
  Power Flow Solutions Using GPUs." *ICPP Workshops '22*, ACM, 2022.
- Kundur, P. S., and O. P. Malik. *Power System Stability and Control*, 2nd
  ed., McGraw Hill, 2022, sec. 6.4.
- Meena, G., et al. "Steady-State Models of AC Microgrid for Load Flow
  Solutions." *ISTEMS 2024*, IEEE, 2024.
- Meng, X., and Z. Yun. "GPU-Based N-1 AC Power Flow with an
  Improved-Convergence Fast Decoupled Method Considering Memory Limitations
  in Large Power Grids." *PowerCon 2023*, IEEE, 2023.
- Palmer, B., et al. "GridPACK: A Framework for Developing Power Grid
  Simulations on High Performance Computing Platforms." *WOLFHPC 2014*,
  IEEE, 2014, pp. 68–77.
- Sai Nandini P, R. Krishan, P. Neerudu, and V. R. Kagita. "HPC-Accelerated
  Parallel Algorithms for Large-Scale Power System Contingency Analysis."
  *Electric Power Systems Research*, vol. 265, 2027, 114205.
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
