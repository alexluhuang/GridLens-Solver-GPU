# GPU batch contingency analysis in GridPACK

This report has four parts:

0. this file: the problem, the questions the report answers, its
   contributions, and the terms used throughout;
1. [01-literature.md](01-literature.md): what the published work does and
   leaves open, organized by theme, and what this work borrows, extends or
   adds;
2. [02-methodology.md](02-methodology.md): how each version of the program
   works, what changed between versions, why each choice was made and which
   alternatives were rejected, in enough detail to rebuild the system
   without reading the code;
3. [03-results.md](03-results.md): the test matrix, the measured runtime and
   accuracy of every version, a discussion of the trends, and the limits of
   the measurements.

## Abstract

An N-1 contingency study solves one ac power flow for every single-element
outage of a transmission network and checks each solution against thermal
and voltage limits. This report describes how GridPACK's contingency
analysis program (`ca.x`) was changed to solve these cases in batches on a
GPU, what each change borrows from published work, and what the changes did
to runtime and accuracy.

The GPU paths keep GridPACK for everything except the solve: GridPACK reads
the network, solves the base case, defines the outages, checks limits and
writes every output. The cases are solved by a batched Newton method that
reproduces GridPACK's equations and control sequence for each case, on one
Jacobian sparsity pattern that every N-1 case fits. The factorization is
either a custom batched LU (Algorithm 2 of Zhou et al.) or NVIDIA's cuDSS
library. GridPACK's own CPU path also had to change: once the solve moved to
the GPU, building and writing the result tables set the runtime.

On one NVIDIA DGX Spark, across seven networks of 993 to 29,637 buses at 8,
16 and 20 processes, the GPU paths ran whole studies 6.4–9.1 times faster
than stock GridPACK at 16 processes on the four public networks above 5,000
buses (8.9–10.2 times at 8 processes). On a 29,637-bus planning case that
stock GridPACK cannot solve, they ran 7.2–9.0 times faster than the
optimized CPU path. Every case received the same outcome on every version
except one ill-conditioned case, and flows and voltages agreed with the
optimized CPU path to the last printed digit (0.0001) on the public networks
above 1,000 buses.

## 1. The problem

A transmission planner checks that the grid stays within its limits after
the loss of any single element: a line, a transformer, a generator or a dc
line pole. This is the **N-1 criterion**. Each outage is one **case**, and
each case needs a full ac power flow: a set of nonlinear equations in the
voltage magnitude and angle at every bus. A network of 10,000 buses has
about 15,000 such cases (15,191 for ACTIVSg10k); the 29,637-bus planning
case used here has 39,628.

Stock GridPACK (Palmer et al.) solves an N-1 study by giving whole cases to
its MPI processes one at a time. On the 20-core DGX Spark with 16 processes,
a full study of ACTIVSg10k took 318 s, and stock GridPACK could not solve
the planning case at all (its base case aborts; 02-methodology §4.4–4.5).
With the fixes of this work, the CPU path solves that case in 3,114 s at 16
processes. Studies of this size are run many times during planning, and
platforms that let a language model call engineering tools on an engineer's
behalf, such as the agentic planning platform EPRI proposes around "common
power flow, dynamic stability, protection studies, and production cost
modeling tools" (2), would call them more often still.

Published GPU work has shown that many sparse matrices with one sparsity
pattern can be factorized together on a GPU far faster than one at a time on
a CPU: on a K40, Zhou et al. report 0.33 ms per 9,241-bus linear system at
batch 128, 76 times faster than KLU's 25 ms, levelling off at 0.31 ms from
batch 512 (4977). What the published work does not show is whether a
production N-1 program can move its solves into such a batch without
changing any result, and how much of a *whole study* that saves once
reading, reporting and file writing are counted. Sections 1 and 6 of
[01-literature.md](01-literature.md) show that most published speedups time
one kernel or one solver, mostly against single-threaded CPU code.

## 2. Questions

The method answers two design questions:

- **M1.** Can every N-1 case that GridPACK solves share one GPU batch
  structure, including cases that move the slack, isolate buses or switch
  generators from voltage control (PV) to fixed reactive output (PQ)?
- **M2.** Can the batch reproduce GridPACK's control sequence for each case,
  so that every outcome and every reported value matches GridPACK's?

The test matrix answers four evaluation questions (03-results §1):

1. How much faster is each version than stock GridPACK, and than the
   optimized CPU path, on networks of 1,000 to 30,000 buses?
2. Which stage of the pipeline sets each version's runtime, and which stage
   accounts for each version's speedup?
3. How do the versions scale with the number of processes and with network
   size?
4. Do the faster versions give the same answers?

## 3. Contributions

Each item is described in 02-methodology and compared with the literature in
01-literature §7. Several parts appeared separately in open-source tools in
the months before this code (01-literature §7.2), so the claim is the
combination, not each part.

1. **A superset Jacobian pattern** that every N-1 case fits: a 2×2 block per
   bus and per connected bus pair of the intact grid, with GridPACK's
   fixed-voltage rows for PV buses, identity rows for the slack and isolated
   buses, and outaged branches kept as stored zeros. Outages, slack moves,
   isolated buses and PV-to-PQ switching become changes of values, never of
   structure, so one factorization plan serves the whole study. It costs
   0.9–5.8% more Jacobian entries than GridPACK's own layout on the networks
   tested.
2. **GridPACK's per-case control sequence inside the batch:** Newton steps,
   divergence rules, reactive-limit switching with GridPACK's deadband and
   pass limit, the sequential dc line updates and the driver's second solve,
   run per slot. A case that fails a numerical check or diverges is solved
   again by GridPACK before it is reported.
3. **Engine changes that keep a batch busy without changing results:** slot
   refilling across submissions, light steps for cases that need no
   factorization, CUDA graphs, and several warps per long factor column with
   factors bitwise identical to one thread per case.
4. **Reporting through GridPACK**, by injecting each GPU solution into a
   GridPACK network copy, with shortcuts for what the classifier already
   knows, so every output is produced by GridPACK's own checks and writers.
5. **Changes to GridPACK's CPU path** that the GPU work exposed: result rows
   built from numbers instead of text, final tables written by all processes
   at once, two corrections of results that depended on which process had
   run which case before, and network models (distributed generation,
   line-commutated dc lines, parser fixes) for a large PSS/E v34 planning
   case.
6. **A whole-study evaluation** of four versions on seven networks at three
   process counts, with every pipeline stage timed and every result row
   compared.

## 4. Terms

The first column gives the plain meaning; the second gives the precise
definition this report uses. Equations are in 02-methodology §2.

| Term | Plain meaning | Precise use in this report |
|---|---|---|
| Bus, branch | A bus is a node of the grid (a substation busbar). A branch is a line or transformer between two buses. | GridPACK's network objects. Parallel circuits between one bus pair are one branch object with several circuits, but separate outages. GridPACK adds a star bus for every three-winding transformer, so its bus count can exceed the file's. |
| Per unit (pu) | Values divided by a base so that normal values are near 1. | Power is divided by the system base of 100 MVA, so 1e-4 pu is 0.01 MW or MVAr; voltage is divided by the bus's base kV. Angles are in radians internally and degrees in reports. |
| Power flow | Given generation and demand, find the voltage (size and angle) at every bus, and from those the flow on every branch. The equations are nonlinear and are solved by repeated correction. | The polar-form ac power-flow equations of Kundur and Malik (sec. 6.4), with GridPACK's load model, solved by Newton's method. |
| Contingency, case | One outage of one element (a branch, a generator or a dc line pole). An **N-1 study** solves one power flow per contingency and reports overloaded branches and out-of-range voltages. A 10,000-bus grid has about 15,000 cases. | One entry of GridPACK's contingency list, generated by `FullBranchN1` (one per in-service circuit), `FullGeneratorN1` (one per in-service generator) and `FullHVDCN1` (one per in-service dc pole), or read from a file. |
| Bus types | A **PV** bus has a generator holding its voltage; a **PQ** bus has fixed demand; the **slack** bus balances the losses. A generator that hits its reactive power limit can no longer hold its voltage, and its bus becomes PQ ("reactive-limit switching"). | PV: real power and voltage magnitude specified. PQ: real and reactive power specified. Slack (reference): voltage magnitude and angle specified. An **isolated** bus is cut off by an outage and removed from the equations. |
| Mismatch | How far the equations are from balance. | Specified minus computed real and reactive power injection at each bus, in pu. The largest absolute value over all equations (the infinity norm) is compared with the tolerance. |
| Newton's method | Each step measures how far the equations are from balance (the **mismatch**), builds the **Jacobian** (the table of how each mismatch changes with each voltage), and solves one large linear system for the correction. A case **converges** when the largest mismatch is below the tolerance (here 1e-4 per unit, i.e. 0.01 MW on a 100 MVA base). | Newton–Raphson in polar form (Kundur and Malik, sec. 6.4.3). A case **diverges** when it reaches 50 steps or when the mismatch grows past 100 times its starting value (02-methodology §2.4). |
| Sparse factorization | The Jacobian is mostly zeros. It is solved by splitting it into two triangular factors (**LU factorization**) and substituting. **Analysis** (choosing an elimination order that keeps the factors sparse, which depends only on where the nonzeros are) can be done once; **factorization** (the numbers) and **solve** are repeated every step. New nonzeros created by elimination are called **fill**. A **pivot** is the diagonal value each column is divided by; a tiny pivot means the factorization is unreliable. | Row and column permutations P and Q are chosen so that PJQ = LU with few fill entries. **Fill-in ratio**: entries of L and U together divided by entries of J. **Refactorization**: computing new L and U values for a new J with the same pattern, permutations and pivot order, without pivoting. |
| Levels | Columns of the factorization that do not depend on each other can be computed at the same time. Grouping them gives a sequence of **levels**; levels run one after another, columns within a level run together. | Column *j* depends on column *i* < *j* when U(*i*,*j*) ≠ 0. A column's level is one more than the highest level among the columns it depends on (levels of the elimination dependency graph). |
| Process (rank) | GridPACK runs as several cooperating operating-system processes started by MPI. Each is called a rank. | An MPI rank. Each holds a full copy of the network. One rank per GPU becomes the **accelerator rank** on the GPU paths. |
| GPU terms | A GPU runs thousands of threads in groups of 32 (**warps**) that execute the same instruction. It is fast when neighbouring threads read neighbouring memory addresses (**coalesced** access) and take the same branch of the code. A **kernel** is one GPU program launch. | Threads are grouped into **thread blocks** that run on one **multiprocessor** (the GB10 has 48). A **CUDA graph** is a recorded sequence of kernel launches that can be replayed with one call. |
| Batch, slot | The GPU path solves many cases at once. The engine has B **slots** (B = batch size, 512 here); each slot holds one case. | A slot is one column of every slot-major array (02-methodology §5.5): slot *s* of a stored position *p* is at index *p*·B + *s*. A **chunk** is a group of cases submitted together (four batches, 2,048 cases at B = 512). |
| Speedup | How many times faster one version is than another. | The ratio of median wall times of whole studies on the same machine at the same process count. |

## 5. Conventions

- **Versions.** "Stock GridPACK" is commit `b32969b0`; "optimized CPU", "Alg
  2" and "cuDSS" are the three ways this branch runs (02-methodology §1).
- **Times** are wall-clock seconds of a whole study, from starting the MPI
  processes to their exit, unless a stage or step is named.
- **Citations** follow MLA 9: author and page (or section) in the text, full
  entries at the end of each part. A citation of the form "02-methodology
  §5.3" points to another part of this report.
- **File names, commits and settings** are given as pointers for readers who
  want to check the code; the text does not depend on them.
- **The private planning case** is Critical Energy/Electric Infrastructure
  Information (CEII). It is reported only in aggregate and is called
  "private case" everywhere.

## References

- Electric Power Research Institute. *Building a Prototype Agentic
  Artificial Intelligence Platform for Transmission Planning*. Supplemental
  Project Notice 3002032336, 2025.
- Kundur, P. S., and O. P. Malik. *Power System Stability and Control*, 2nd
  ed., McGraw Hill, 2022, sec. 6.4.
- Palmer, B., et al. "GridPACK: A Framework for Developing Power Grid
  Simulations on High Performance Computing Platforms." *WOLFHPC 2014*,
  IEEE, 2014, pp. 68–77.
- Zhou, G., et al. "GPU-Based Batch LU-Factorization Solver for Concurrent
  Analysis of Massive Power Flows." *IEEE Transactions on Power Systems*,
  vol. 32, no. 6, 2017, pp. 4975–77.
