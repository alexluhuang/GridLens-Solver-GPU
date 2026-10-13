# GPU batch N-1 contingency analysis

This report has four parts:

0. this file: the problem, the three goals (speed, accuracy and
   compatibility), the contributions, and the background needed to read the
   rest;
1. [01-literature.md](01-literature.md): what published work has done for
   each goal and what it leaves open;
2. [02-methodology.md](02-methodology.md): how the system meets each goal,
   in enough detail to rebuild it, with the reasons for each choice;
3. [03-results.md](03-results.md): the test setup and the measured speed,
   accuracy and coverage of every version.

Readers new to power-flow analysis or GPU programming should read Section 5
of this file first.

## Abstract

An N-1 contingency study solves one ac power flow for every single-element
outage of a transmission network and checks each solution against thermal
and voltage limits. A 10,000-bus network has about 15,000 such cases. This
work solves the cases in batches on a GPU, inside GridPACK, an open-source
framework for parallel power-grid software. All outages share one Jacobian
sparsity pattern, so one ordering and symbolic analysis serves the whole
study. Each batch member runs its own Newton iterations and reactive-limit
switching, so each case follows the same sequence as on a sequential solver.
The factorization is either a batched LU written from Algorithm 2 of Zhou et
al. or NVIDIA's cuDSS library. Cases that the batch cannot represent, or that
fail a numerical check, are solved on the CPU. Building and writing the
result tables also had to be made faster, because once the solves moved to
the GPU these steps set the runtime.

On one NVIDIA DGX Spark, across nine networks of 993 to 29,637 buses, the GPU
paths ran whole studies 6.4–9.1 times faster than unmodified GridPACK with 16
CPU processes on the six public networks above 5,000 buses, and 8.3–10.2
times faster with 8 processes. On a 29,637-bus planning case that unmodified
GridPACK cannot solve, they ran 7.2–9.0 times faster than the corrected CPU
path. Every case received the same outcome on every version except one
ill-conditioned case. Flows and voltages agreed with the CPU path to the last
printed digit (0.0001) on the public networks above 1,000 buses, apart from
one case that the CPU path stopped just inside the solver tolerance.

## 1. The problem

A transmission planner must show that the grid stays within its limits after
the loss of any single element: a line, a transformer, a generator or one
pole of a dc line. This is the N-1 criterion. Each outage is one case, and
each case needs a full ac power flow, a system of nonlinear equations in the
voltage magnitude and angle at every bus (Section 5.1). The ACTIVSg10k
synthetic network has 15,191 cases; the 29,637-bus planning case used here
has 39,628.

A CPU program solves these cases one at a time per process. With 16
processes on the 20-core DGX Spark, GridPACK's contingency program took 318 s
for ACTIVSg10k and 2,242 s for the 25,000-bus ACTIVSg25k. Planners rerun such
studies for each change of dispatch, topology or proposed project. The
Electric Power Research Institute (EPRI) has also proposed planning platforms
in which software agents call "common power flow, dynamic stability,
protection studies, and production cost modeling tools" on an engineer's
behalf (2), which would increase the number of runs.

Published GPU work shows that many sparse linear systems with one sparsity
pattern can be factorized together much faster than one at a time. Zhou et
al. report 0.33 ms per 9,241-bus linear system at batch 128 on a K40, 76
times faster than the KLU library's 25 ms (4977). Three questions remain open
(01-literature §7): whether a full N-1 study fits such a batch, including
outages that move the reference bus, isolate buses or push generators to
their reactive limits; whether the batched results match a sequential
solver's case by case; and how much of a whole study's runtime the batch
removes once reading the network, checking limits and writing results are
counted.

## 2. Goals and questions

The work has three goals, each defined so that it can be measured.

**Speed.** Reduce the wall time of a complete study, from starting the
program to the final result tables, on the same machine as the CPU baseline
and with the same number of CPU processes.

**Accuracy.** Give every case the outcome (converged, diverged, islanded,
slack overload) and the reported flows and voltages that a sequential Newton
solver with the same rules gives. Newton's method stops at any point whose
mismatch is below the tolerance, so two correct solvers can stop at slightly
different points (Section 5.2). "The same" therefore means flows and
voltages that agree within the effect of that tolerance, and identical
outcomes and violation lists.

**Compatibility.** Accept the same input files and settings as the CPU
program and write the same output tables; complete every case of a study,
using the CPU for any case the GPU cannot represent; run on machines without
a GPU; and read the large PSS/E planning models used in practice.

The report answers one question per goal:

1. **Speed.** How much faster is each version than the CPU baseline on
   networks of 1,000 to 30,000 buses, which stage of the study sets its
   runtime, and how does that change with the number of processes and the
   size of the network? (03-results §3)
2. **Accuracy.** Do the faster versions give the same outcomes, flows,
   voltages and violation lists? (03-results §4)
3. **Compatibility.** What share of each study runs on the GPU, which inputs
   and cases fall back to the CPU, and does the program run where the
   baseline does? (03-results §5)

## 3. Approach

The system is built on GridPACK (Palmer et al.), a C++ framework from Pacific
Northwest National Laboratory for writing parallel power-grid applications.
GridPACK is the base platform because it is modular: the network model, the
file parsers, the per-bus and per-branch equation components, the matrix
builders, the solvers, the task scheduler and the contingency application
are separate modules. A new solver can replace the Newton solves of the case
loop while the parser, the contingency definitions, the limit checks and the
output writers are reused. This serves compatibility, because inputs and
outputs remain those of an existing program. It also serves accuracy,
because the CPU reference and the GPU path then differ only in how the power
flow is solved.

The GPU path keeps the CPU processes for set-up and reporting and moves the
Newton solves of most cases into a batched engine on the GPU. Each GPU
solution is loaded back into the CPU program's network model, so the reused
limit checks and writers produce the reported values. 02-methodology §1
describes the overall design.

## 4. Contributions

Each item is described in 02-methodology and compared with published work in
01-literature §7. Several parts appeared separately in open-source tools in
the months before this work (01-literature §7.2), so the contribution is the
combination.

Speed:

1. A **superset Jacobian pattern** that every N-1 case fits. Every bus and
   every connected bus pair of the intact grid has a 2×2 block;
   voltage-controlled and reference buses get fixed rows instead of losing
   rows; an outaged branch keeps its entries with value zero. Outages,
   reference-bus moves, isolated buses and reactive-limit switching then
   change only values, so one ordering and symbolic analysis serves the
   whole study. It costs 0.9–5.8% more Jacobian entries on the networks
   tested.
2. **Batch scheduling** that keeps the GPU busy while cases converge at
   different rates: a finished case's slot is refilled at once from the next
   submission, cases that need no factorization take a cheap step, and the
   long final columns of the factorization are spread over several warps
   with bitwise-identical results.
3. **CPU-side changes** that the faster solve made necessary: result rows
   built from solved values instead of a text round trip, final tables
   written by all processes at once, and reporting that skips topology work
   already done before the batch.

Accuracy:

4. **Per-case control inside the batch.** Each slot runs the CPU program's
   Newton loop, divergence rules, reactive-limit switching with its deadband
   and pass limit, the sequential update of dc lines, and the driver's
   second solve.
5. **Numerical safeguards and determinism.** Every factorization and solve is
   checked per case. A failing or diverged case is solved again on the CPU,
   with the CPU solver's own pivoting, before it is reported. Results do not
   depend on the batch, the slot, the process or the run.
6. **Two corrections of the CPU baseline**, whose results depended on which
   process had solved which case before.

Compatibility:

7. **One program for all machines and all cases.** The same executable and
   input file run with or without a GPU; the GPU code is a separately loaded
   library; every case the batch cannot represent is solved on the CPU path.
8. **Network models for PSS/E v34 planning cases**: parser fixes,
   distributed generation on loads, and line-commutated dc lines with the
   sequential ac/dc method on both CPU and GPU.

Evaluation:

9. A **whole-study evaluation** of four versions on nine networks at three
   process counts, with every pipeline stage timed and every result row
   compared.

## 5. Background

This section introduces the concepts the rest of the report uses.
02-methodology §2 gives the exact rules the system implements.

### 5.1 The network and the power-flow problem

A transmission network is modelled as a graph. Its nodes are **buses**
(substation busbars) and its edges are **branches** (lines and
transformers). Generators, loads and shunt devices are attached to buses.
Each branch is an equivalent π circuit: a series impedance between its two
buses and a shunt admittance at each end. A transformer adds a tap ratio and
possibly a phase shift (Kundur and Malik, sec. 6.4). Quantities are in
**per unit**: power is divided by a system base of 100 MVA, so 1e-4 per unit
is 0.01 MW or MVAr, and each voltage is divided by its bus's nominal voltage,
so normal voltages are near 1.0.

The branches and shunts define the **bus admittance matrix** **Y** = **G** +
j**B**. Its off-diagonal entry Y_km is minus the series admittance of the
branches between buses *k* and *m*; its diagonal entry Y_kk is the sum of
all admittances attached to bus *k*. **Y** is sparse because a bus connects
to only a few others: ACTIVSg10k has 10,000 buses, about 12,000 branches,
and roughly 34,000 nonzero entries in **Y** out of 10⁸.

The **power-flow problem** is to find the voltage magnitude V_k and angle
θ_k at every bus, given what each generator produces and each load
consumes. The power injected into the network at bus *k* follows from the
voltages (Kundur and Malik, eq. 6.101):

> P_k = Σ_m V_k V_m (G_km cos θ_km + B_km sin θ_km)
>
> Q_k = Σ_m V_k V_m (G_km sin θ_km − B_km cos θ_km)

with θ_km = θ_k − θ_m and the sum over bus *k* and its neighbours. The
**mismatch** at a bus is the difference between this computed injection and
the scheduled one. The equations are nonlinear in V and θ and are solved
iteratively.

Each bus fixes two of the four quantities P, Q, V and θ according to its
**bus type**:

- a **PQ bus** (most buses) has fixed real and reactive power, and V and θ
  are unknown;
- a **PV bus** has a generator that holds the voltage magnitude at a set
  point, so P and V are known and θ and Q are unknown;
- the **slack** (reference) bus has fixed V and θ = 0 and supplies the real
  power that balances the losses.

A generator can supply reactive power only between its limits Q_min and
Q_max. If holding the set voltage would need more, the generator is fixed at
the limit and its bus becomes PQ (Kundur and Malik, sec. 6.4.2(b)). This
**reactive-limit switching** changes the set of equations during the solve.

The solved voltages determine the flow on every branch. A branch is
**overloaded** when its apparent power exceeds its rating; a bus violates its
**voltage limits** when V lies outside a band such as 0.9–1.1 per unit.

### 5.2 Newton's method

Newton's method solves **F**(**x**) = 0, where **x** holds the unknown angles
and voltage magnitudes and **F** the mismatches. Each step:

1. evaluates the mismatch **F** at the current **x**;
2. builds the **Jacobian** **J**, the matrix of partial derivatives of each
   mismatch with respect to each unknown;
3. solves the linear system **J** Δ**x** = **F** for the correction;
4. updates **x** ← **x** − Δ**x**.

The iteration **converges** when the largest absolute mismatch, the infinity
norm ‖**F**‖∞, falls below a tolerance (here 1e-4 per unit, or 0.01 MW). Near
the solution the error roughly squares at each step, so from a good starting
point two or three steps suffice. A case **diverges** when it reaches a step
limit or its mismatch grows. **J** has the sparsity of **Y**: the rows of a
bus have nonzeros only for that bus and its neighbours. Step 3 is the most
expensive; D'Orto et al. measured 81–93% of a single power flow's time in it
(56611).

Newton's method stops at the first point whose mismatch is below the
tolerance. Two correct implementations that start from different points, or
round differently, can therefore stop at slightly different points, and
their flows can differ by about the tolerance (0.01 MW), more on a poorly
conditioned case.

### 5.3 Sparse direct solution

The linear system **J** Δ**x** = **F** is solved by **LU factorization**. Rows
and columns are permuted by matrices P and Q, and the permuted matrix is
split into a lower triangular factor **L** and an upper triangular factor
**U**, so that P**J**Q = **LU**. Forward substitution with **L** and backward
substitution with **U** (the **triangular solves**) then give Δ**x**.

Elimination creates nonzeros in **L** and **U** where **J** had zeros; these
are called **fill**. The **fill-in ratio** is the number of entries in **L**
and **U** divided by the number in **J**. The permutation determines the
fill: an ordering such as approximate minimum degree (AMD) keeps it small,
and less fill means less memory and less arithmetic. A sparse direct solver
therefore has three phases:

1. **analysis** chooses the ordering and finds where the nonzeros of **L**
   and **U** will be; it depends only on the sparsity pattern of **J**;
2. **factorization** computes the values of **L** and **U**;
3. **solve** runs the two triangular substitutions.

Factorization divides each column by a diagonal entry, the **pivot**. A tiny
pivot makes the factors inaccurate. **Partial pivoting** guards against this
by swapping in a larger entry during factorization, which changes the
structure of the factors. **Refactorization** computes new values for a
matrix with the same pattern while reusing the previous ordering and pivot
sequence, without pivoting. It is faster and is what makes batching
possible, but it assumes the reused pivots stay large enough, which has to
be checked.

A column of the factors depends on earlier columns: column *j* needs column
*i* < *j* when U(*i*, *j*) is nonzero. Columns that do not depend on one
another can be computed at the same time. Grouping them gives a sequence of
**levels**; the levels run one after another and the columns within a level
run in parallel.

### 5.4 An N-1 study

A study has five stages:

1. **read** the network file (here PSS/E RAW, the common exchange format for
   transmission models);
2. **solve the base case**, the intact network;
3. **build the case list**: one outage per in-service branch circuit, per
   generator and, optionally, per dc line pole;
4. **run the case loop**: for each case, take the element out of service,
   find the buses it disconnects, move the slack role if the slack generator
   was lost, solve the power flow, check the slack generator's output and
   every voltage and monitored branch against their limits, write the result
   rows, and restore the network;
5. **write the final tables**: per-case outcomes, violations, and one row per
   monitored branch per case (149 million rows for ACTIVSg10k).

An outage can split the network into **islands**. A bus left with no
connection is **isolated** and removed from the equations. A split into
several multi-bus islands is reported as islanded and not solved, under the
rules of the program used here. Cases are independent of one another, so the
case loop parallelizes across processes. The CPU program runs as several
cooperating operating-system processes started by MPI, each called a
**rank**. Each rank holds a full copy of the network and takes the next case
when it is free.

### 5.5 GPU execution

A GPU runs thousands of threads. Threads are grouped into **warps** of 32
that execute the same instruction together, and warps into **thread
blocks**, each of which runs on one **multiprocessor** (the GB10 GPU of the
DGX Spark has 48). A **kernel** is one GPU program launch over many blocks.
Each launch has a fixed overhead of microseconds; a **CUDA graph** records a
sequence of launches so that it can be replayed with one call.

Two rules decide whether GPU code is fast (NVIDIA, *CUDA C++ Best Practices
Guide*, secs. 10.2.1 and 13.1):

- **coalesced memory access**: when the 32 threads of a warp read 32
  adjacent addresses, the hardware serves them in a few memory transactions,
  while scattered addresses cost one transaction each;
- **no divergence**: when threads of a warp take different branches of a
  conditional, the branches run one after another.

A single power flow suits a GPU poorly. Its Jacobian has tens of thousands
of rows; its factorization has hundreds of levels, many with only a few
columns, so most of the GPU idles; and sparse indexing scatters memory
accesses (Zhou et al. 4976). A **batch** of many matrices with one sparsity
pattern changes this. If thread *t* handles matrix *t*, every thread of a
warp runs the same instruction on the same position of a different matrix.
Storing the values of one position for all matrices next to each other makes
every access coalesced, and the available parallelism grows with the batch
size. One analysis serves every matrix in the batch.

The GB10's GPU is integrated with the CPU and shares its memory. On a GPU
attached by PCI Express, inputs and results would also have to cross that
link.

## 6. Conventions

- **Versions.** "Stock GridPACK" is GridPACK commit `b32969b0`. "Optimized
  CPU", "Alg 2" and "cuDSS" are the three ways the modified program runs
  (02-methodology §1.2).
- **Times** are wall-clock seconds of a whole study, from starting the MPI
  processes to their exit, unless a stage is named. A **speedup** is a ratio
  of median wall times on the same machine at the same process count.
- **Batch terms.** The GPU engine has B **slots** (B is the batch size, 512
  in the tests), each holding one case. A **chunk** is a group of cases
  submitted together, four batches.
- **Citations** follow MLA 9: author and page or section in the text, full
  entries at the end of each part. "02-methodology §3.4" points to another
  part of this report.
- **File names and settings** are pointers for readers who want to check the
  code; the text does not depend on them. Appendix A of 02-methodology maps
  each section to its source files.
- **The private planning case** is Critical Energy/Electric Infrastructure
  Information (CEII). It is reported only in aggregate and is called
  "private case" everywhere.

## References

- D'Orto, M., et al. "Comparing Different Approaches for Solving Large Scale
  Power-Flow Problems with the Newton-Raphson Method." *IEEE Access*, vol.
  9, 2021, pp. 56604–15.
- Electric Power Research Institute. *Building a Prototype Agentic
  Artificial Intelligence Platform for Transmission Planning*. Supplemental
  Project Notice 3002032336, 2025.
- Kundur, P. S., and O. P. Malik. *Power System Stability and Control*, 2nd
  ed., McGraw Hill, 2022, sec. 6.4.
- NVIDIA. *CUDA C++ Best Practices Guide*, release 13.4.
- Palmer, B., et al. "GridPACK: A Framework for Developing Power Grid
  Simulations on High Performance Computing Platforms." *WOLFHPC 2014*,
  IEEE, 2014, pp. 68–77.
- Zhou, G., et al. "GPU-Based Batch LU-Factorization Solver for Concurrent
  Analysis of Massive Power Flows." *IEEE Transactions on Power Systems*,
  vol. 32, no. 6, 2017, pp. 4975–77.
