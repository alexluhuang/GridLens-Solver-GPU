# From stock GridPACK to GPU batch contingency analysis: methodology

This report describes, in the form of a methodology section, every change
made to GridPACK's contingency analysis program (`ca.x`) on this branch.
It has three parts:

1. this file: how each version of the program works, what changed between
   versions, why, and which published work each change draws on;
2. [02-literature.md](02-literature.md): how the result differs from the
   published work it builds on;
3. [03-results.md](03-results.md): the test matrix, the measured runtime and
   accuracy of every version, and the trends in them.

The aim is that a software engineer can rebuild the system from these
pages without reading the code. File names are given only as pointers.

## 1. The four programs compared

| Name in this report | What it is |
|---|---|
| **Stock GridPACK** | GridPACK's `develop` branch at commit `b32969b0` (release 3.7.0 line), with its contingency analysis program and its table outputs (`csv_flat`, `csv_delta`, violation tables, monitoring filters). It is timed with a patch that adds timer calls and nothing else (`docs/gpu_n1/reproductions/stock-timer-only.patch`). |
| **Optimized CPU** | This branch with the GPU switched off (`GPUBatch/enabled = off`, or no `GPUBatch` block). Every case is solved by GridPACK's own solver, as in stock, but reporting, file writing and several defects are changed (Section 4). |
| **Alg 2** | This branch with the GPU on and the custom batched factorization (`GPUBatch/backend = alg2`). |
| **cuDSS** | This branch with the GPU on and NVIDIA's cuDSS library doing the factorization (`backend = cudss`). Everything else is shared with Alg 2. |

The same executable runs all three "this branch" versions; the input file
decides which. The GPU code is a separate library loaded at start-up, so the
executable also runs on machines without a GPU.

## 2. The problem in plain terms

| Term | Meaning |
|---|---|
| Bus, branch | A bus is a node of the grid (a substation busbar). A branch is a line or transformer between two buses. |
| Power flow | Given generation and demand, find the voltage (size and angle) at every bus, and from those the flow on every branch. The equations are nonlinear and are solved by repeated correction. |
| Contingency, case | One outage of one element (a branch, a generator or a dc line pole). An **N-1 study** solves one power flow per contingency and reports overloaded branches and out-of-range voltages. A 10,000-bus grid has about 15,000 cases. |
| Bus types | A **PV** bus has a generator holding its voltage; a **PQ** bus has fixed demand; the **slack** bus balances the losses. A generator that hits its reactive power limit can no longer hold its voltage, and its bus becomes PQ ("reactive-limit switching"). |
| Newton's method | Each step measures how far the equations are from balance (the **mismatch**), builds the **Jacobian** (the table of how each mismatch changes with each voltage), and solves one large linear system for the correction. A case **converges** when the largest mismatch is below the tolerance (here 1e-4 per unit, i.e. 0.01 MW on a 100 MVA base). |
| Sparse factorization | The Jacobian is mostly zeros. It is solved by splitting it into two triangular factors (**LU factorization**) and substituting. **Analysis** (choosing an elimination order that keeps the factors sparse, which depends only on where the nonzeros are) can be done once; **factorization** (the numbers) and **solve** are repeated every step. New nonzeros created by elimination are called **fill**. A **pivot** is the diagonal value each column is divided by; a tiny pivot means the factorization is unreliable. |
| Levels | Columns of the factorization that do not depend on each other can be computed at the same time. Grouping them gives a sequence of **levels**; levels run one after another, columns within a level run together. |
| Process (rank) | GridPACK runs as several cooperating operating-system processes started by MPI. Each is called a rank. |
| GPU terms | A GPU runs thousands of threads in groups of 32 (**warps**) that execute the same instruction. It is fast when neighbouring threads read neighbouring memory addresses (**coalesced** access) and take the same branch of the code. A **kernel** is one GPU program launch. |
| Batch, slot | The GPU path solves many cases at once. The engine has B **slots** (B = batch size, 512 here); each slot holds one case. |

## 3. Stock GridPACK: the baseline pipeline

Stock GridPACK runs an N-1 study in six stages. Every rank holds a complete
copy of the network and handles one case at a time.

1. **Read network.** The PSS/E RAW file is parsed and distributed to all
   ranks.
2. **Base case.** The intact grid is solved by Newton's method with PETSc's
   KLU sparse solver (a CPU direct solver). With `qlim` on, generators over
   their reactive limits are switched to PQ and the case is solved again, up
   to `maxQlimIterations` (3) times.
3. **Case list and output setup.** The outage list is generated
   (`FullBranchN1`, `FullGeneratorN1`) or read from a file, and each rank
   opens its own part file for each table.
4. **Case loop.** A shared counter (GridPACK's task manager, built on the
   Global Arrays library) hands each rank the next case number when it is
   free. For each case the rank:
   - resets the bus voltages to the values in the RAW file;
   - **applies the outage**: sets the element out of service, moves the
     slack to the bus with the most online capacity if the slack generator
     was lost, searches the network for isolated single buses and for
     islands (keeping the largest), and updates the admittances and power
     injections of the whole network;
   - **solves** the case with Newton's method and KLU, with the same
     reactive-limit loop as the base case, and then, if `qlim` is on, checks
     the limits once more and solves again if a bus switched. Each solve
     rebuilds GridPACK's matrix mappings. A case diverges after 50 steps
     (`maxIteration`), when the mismatch grows past 100 times its starting
     value, or when it stalls for five steps with nothing left to switch;
   - **checks and reports**: slack overload, bus voltages against 0.9 and
     1.1 per unit, and branch loading against rating C, then writes one row
     per monitored branch to the `csv_flat` part file. To build the rows,
     GridPACK formats the state of every bus and branch as text, gathers the
     text through Global Arrays, and the driver reads the numbers back out of
     the text;
   - **restores** the network (undoes the outage).
5. **Merge output files.** Rank 0 reads every rank's part files and writes
   the final tables.

Section 3 of [03-results.md](03-results.md) shows where the time goes. In
short, two things set stock GridPACK's runtime: the Newton solves (about
three fifths of the case loop) and building and writing the table rows
(most of the rest). Earlier profiling of the solve on ACTIVSg10k found that
of 299 s per rank, building the Jacobian took 89 s, recreating the matrix
mappings 83 s, the mismatch 29 s and the linear solve itself only 36 s
(`docs/gpu_n1/performance.md`).

## 4. Stock GridPACK to optimized CPU

The optimized CPU path keeps GridPACK's solver and every rule above. The
changes fall into three groups: speed (4.1–4.3), correctness (4.4) and
network modelling (4.5). Every speed change was checked to leave the output
files byte-identical, on IEEE118, Polish, Texas7k and ACTIVSg10k.

### 4.1 Table rows built from numbers

*What.* The `csv_flat` and `csv_delta` rows are filled directly from each
bus's and branch's solved values, instead of formatting every bus and branch
as text, gathering the text and parsing the numbers back.

*How.* At the first case, the driver builds two lists: the monitored buses
(sorted by bus number, as GridPACK's text map sorted them) and the branches
(in GridPACK's global order), with each branch's circuit name, its ends'
positions in the bus list, its monitoring decision and its rating looked up
once. For each case it reads each bus's voltage and angle and each branch's
first circuit's flows straight from the network objects. To keep the files
byte-identical, every value is rounded to 13 significant digits, which is
exactly what printing with 13 digits and reading back produced, and a
branch whose circuit name would not read back as one word still takes the
old text route.

*Why.* Once the solve was on the GPU, reporting set the speed: on Texas7k,
checks and reporting took 103 s per rank of a 194 s study, 95 s of it
building rows (`performance.md`). The text round trip did the same work as
a direct read plus formatting and parsing every value of the network.
Palmer et al. already note that input and output can be "a major fraction"
of a GridPACK run (76). No algorithm from the literature is involved; this
is an engineering change.

*Effect.* A 16-rank Texas7k CPU study fell from 150 to 112 s (commit
`4ffdeb13`), and row writing on ACTIVSg10k from 91 to 12.5 s per rank.

### 4.2 Final tables written by all ranks at once

*What.* The merge of part files into the final tables is done by every
rank together instead of by rank 0 alone.

*How.* Each rank scans its own part file for runs of rows that belong to
one case. The ranks exchange only the run sizes (two collective exchanges
of a few numbers per case). Every rank then computes the same layout of the
final file: rank order, then file order on the CPU path; case order, ties
kept in rank order, on the GPU paths. The final file is sized once, and
each rank copies its own runs to their offsets with positional writes. The
bytes are identical to the old single-rank merge (`ca_parallel_write.hpp`).
This is the pattern of collective parallel writes in MPI-IO: compute
offsets from exchanged sizes, then write independently.

*Why.* On ACTIVSg10k the 17 GB `csv_flat` table took rank 0 8.7 s to merge,
a fifth of a 42 s GPU study.

*Effect.* 8.7 to 4.0 s on that study. The merge is now limited by the disk:
16 threads copying the part files moved 3.3 GB/s.

### 4.3 Measurement, rank count and output options

- **Step timers** (`d62432ec`, `e656133e`). GridPACK's own coarse timer
  gained one category per stage (read, base case, case list, case loop,
  merge) and per case-loop step (apply outage, CPU solve, inject GPU
  result, check and report, write rows, restore). Stock is timed with the
  same categories through the timer-only patch; its wall time is within
  1.5% of untouched stock.
- **Rank count** (`ce951584`). `ca_run.sh` picks the number of ranks from
  the cores: cores − 4 with eight or more cores, so 16 on the 20-core DGX
  Spark.
- **Parquet output** (`e1caa4ed`, optional, not used in the tests here):
  `outputFormat = parquet` writes the branch table as a columnar dataset in
  case order.
- **Run-time matrix layout** (`308277ff`). GridPACK's alternative
  "large matrix" Jacobian layout, previously a compile-time switch, can be
  chosen at run time. The GPU path uses it as a test reference; the CPU
  path's default is unchanged.

### 4.4 Correctness fixes

Each fix applies to the CPU and GPU paths alike.

1. **Voltage settings restored between cases** (`c81a8387`, `131827e3`).
   A bus can have a generator holding its own voltage and another
   generator holding it remotely. When the local unit trips, GridPACK's
   remote-control logic changes both the bus voltage and the stored setting
   that later cases reset to. Stock cleanup restores the generator but not
   that setting, so the next case on the same rank starts with a different
   held voltage. Results then depend on which rank ran which case before:
   two runs of stock GridPACK on Texas7k differ in three late cases, and on
   ACTIVSg10k a later case lost a PV bus and moved by 0.022 per unit. The
   fix saves every bus's reset values once and restores any that a case
   changed after the case is reported. *Reasoning:* Kundur and Malik
   (sec. 6.4) distinguish a specified generator voltage from a starting
   estimate; the retained value is a specification, so it must not leak
   between independent outages. The user approved this change to stock
   behaviour on 2026-10-05 (`docs/gpu_n1/restoration.md`).
2. **No reactive-limit check at disconnected buses** (`2d3bcbb2`). A bus cut
   off by an outage still held the reactive output computed for it by an
   earlier case. GridPACK's equations already skip such buses, but its
   limit check did not, so it could switch a disconnected bus and run a
   needless extra solve. Kundur and Malik (sec. 6.4.2(b)) compare a
   generator's *computed* output with its limits, and a disconnected bus has
   none. Solved values do not change; one extra solve, its count and its
   warning disappear.
3. **Bounded log and error messages** (`8e86d415`, `44d81e51`). Solver error
   messages were printed into fixed 128-byte buffers. A long PETSc error,
   or a diverging case's long mismatch line, overflowed the buffer and
   aborted the whole study. All such messages are now bounded or built as
   strings.
4. **Converter state restored** (`f430ad74`). Removing a contingency now
   also resets the dc converter injections (Section 4.5), as it already
   reset switched shunts and taps.
5. **Outages that cut off the reference bus** (`668dd8f7`) are left to
   GridPACK's loop, which reports the solver failure. (Only the GPU path
   handled these differently.)

### 4.5 Network modelling for large planning cases

These additions came from a 28,000-bus PSS/E v34 interconnection planning
case (CEII; reported here in aggregate only) on which stock GridPACK's
first Newton step met a singular Jacobian. Full detail is in
`docs/markdown/WECC_COMPATIBILITY.md`.

- **Parser fixes.** A `/` inside any quoted name no longer starts a comment
  (it had dropped 86 in-service transformers); system switching devices
  (breakers) are read as low-impedance branches instead of being skipped;
  transformer winding voltages are converted to per unit of the bus base
  voltage for every winding code; and a transformer defined in the opposite
  direction to a parallel branch keeps its orientation. On pairs of
  physically identical test cases written two ways, these removed voltage
  differences of up to 1.07 per unit.
- **Distributed generation (DG) on loads.** PSS/E v34+ load records carry
  generation behind the meter (DGENP, DGENQ, DGENF). It is stored apart
  from the load and subtracted from the demand as a constant-power
  injection, everywhere the demand enters: the equations, the reactive-limit
  check, slack output and reports. This is the constant-power model of
  inverter DG reviewed by Meena et al.; NERC's DER modelling guideline asks
  for DG to be kept in the load record rather than netted. Ignoring the
  case's 11.4 GW of DG left it 10.6 GW out of balance.
- **Two-terminal (line-commutated) dc lines.** Each converter is modelled by
  the standard bridge equations (Kundur, ch. 10), with the exact
  power-factor formula, tap limits and the three PSS/E control modes
  (normal, inverter at minimum extinction angle, rectifier at minimum firing
  angle). The dc lines are coupled to the ac grid by the **sequential
  method** (Khan and Bhowmick, sec. 1.7): during each Newton solve the
  converters are fixed power injections at their buses; after each
  converged solve the lines are solved again at the new ac voltages and,
  if any converter's power moved by more than `hvdcTolerance`, the
  controller loop repeats. Each case starts its dc lines from the
  base-case operating point, so results do not depend on case order.
- **dc line contingencies.** A `DCLine` contingency blocks one or more poles;
  `FullHVDCN1` adds one outage per in-service pole.

In the matrix of [03-results.md](03-results.md), only the private planning
case has DG and dc lines; on the public networks these additions do not
change any result.

## 5. Optimized CPU to the GPU paths: the shared design

Both GPU paths share everything except the factorization and the
triangular solves. The design rule was to reuse GridPACK for everything it
already does (reading, the base case, contingency rules, limit checks,
writing) and to add new code only where the cases are solved (design
guide, ADR-07, ADR-10, ADR-12).

### 5.1 Start-up and roles

With a `GPUBatch` block, each rank reads the settings and checks them.
One rank per GPU becomes the **accelerator rank** (rank 0 on the one-GPU
DGX Spark); it loads the GPU library ("plugin", a separate shared library
with a versioned C interface), finds the device and records its memory
type. The other ranks run GridPACK as before. Missing GPU or library falls
back to the CPU loop, or stops if `onUnavailable = error` (as in the tests
here). Every effective setting is written to the log.

### 5.2 Classification: which cases go to the GPU

Every case is classified once, on the CPU, before any solving. The
classifier applies the outage with GridPACK's own routine (slack transfer,
lone-bus and island search), records what changed, and restores the
network. A case goes to the GPU if every change it makes can be expressed
as a change of numbers in the shared Jacobian pattern of Section 5.3:
branch and generator outages, a single isolated bus, a slack move, loss of
voltage control, dc pole outages. It stays with GridPACK if:

- the study enables controls the GPU does not reproduce (switched shunts,
  tap changers, area interchange) or remote voltage control would act;
- the outage splits the grid into islands of two or more buses (GridPACK
  reports these as `ISLANDED` without solving), leaves no slack, or cuts off
  the reference bus;
- an element cannot be found.

Most cases cannot change the topology, so a fast path decides them from
the network graph without running GridPACK's full search: one search for
**bridges** (branches whose loss splits the graph; Tarjan's linear-time
method) is done at start-up. A branch that keeps a parallel element in
service, or is not a bridge, changes nothing; a bridge with a single bus at
one end isolates that bus; any other bridge splits the grid. Generator and
dc pole outages change only injections. The fast path is used only after
checking that the intact network is one island with a working slack, and it
was cross-checked against GridPACK's full routine on every case of the
validation set. On ACTIVSg10k, classifying all cases takes well under a
second.

### 5.3 One Jacobian pattern for every case (the superset pattern)

A factorization plan is valid only while the nonzeros stay in the same
places. In GridPACK's default layout they move: an outage removes entries,
a PV bus has one equation instead of two, the slack and isolated buses
have none. The batch path uses one **superset pattern** that every case
fits:

| Element | Rule |
|---|---|
| Every bus | a 2×2 block (two equations, two unknowns), including the slack and any bus that may become isolated |
| Every bus pair joined by a branch in the intact grid | a 2×2 block, kept even when the branch is out |
| PQ bus | GridPACK's real and reactive power equations |
| PV bus | GridPACK's real power equation, and a fixed-voltage row (1 on its own voltage) |
| Slack and isolated buses | identity rows, with their coupling entries set to zero |
| Branch outage | the off-diagonal entries become zero and the diagonal loses the branch's contribution, taken from GridPACK's own per-circuit values |

Every effect of an N-1 case, including reactive-limit switching during the
solve, is then a change of numbers, never of structure. The rows come from
GridPACK's large-matrix layout (from 2013), which already gives PV and slack
buses fixed rows; keeping outaged entries as stored zeros is the method of
Zhou et al. ("GPU-Based Batch" 4976). The cost is a slightly larger
Jacobian: 0.9–5.8% more entries than GridPACK's standard layout on the
networks tested (logged per run; `figures/size.csv` of
[03-results.md](03-results.md)).

### 5.4 One-time plan

At start-up the planner builds the superset pattern, fills it with the
base-case Jacobian, and factorizes it once with SuiteSparse KLU, using AMD
ordering (approximate minimum degree), partial pivoting with tolerance
0.001, and no block-triangular pre-ordering, so the result is one block.
This fixes the row and column order and the structure of the factors for
the whole study. From that structure it derives what the GPU kernels need:
where each Jacobian entry lands in the factor storage, the list of updates
each column receives, and the dependency levels for the factorization and
for both triangular solves. This is the analysis step of D'Orto et al.'s
fastest hybrid (KLU with AMD on the CPU, refactorization on the GPU), but
done once per study instead of once per power flow. Planning takes 0.04 s
on Wisconsin and well under a second on every network tested.

### 5.5 The batched Newton engine

**Data layout.** For every stored position (a Jacobian entry, a factor
entry, a bus voltage) the values of all B slots sit next to each other in
memory. GPU thread *t* handles slot *t*, so the 32 threads of a warp read 32
consecutive values (coalesced) and, because every slot has the same
structure, never take different branches. This is Zhou et al.'s layout
(4976).

**Equations.** The mismatch, the Jacobian blocks, the branch flows, the
voltage update and the reactive-limit check are GridPACK's formulas, written
once as functions that compile for both CPU and GPU. Each bus adds its
terms in the order GridPACK adds them, without atomic additions, so the
result of a case never depends on thread timing. Measured against
GridPACK, the mismatch agrees to a relative 3.8e-16 and the Jacobian to
1.5e-16.

**Control sequence.** A small state machine on the host replays, per slot,
exactly what GridPACK's `PFAppModule::solve()` and the contingency driver
do: Newton steps until the largest mismatch is below the tolerance;
divergence after 50 steps, on growth past 100 times the starting mismatch,
or on stalling; reactive-limit checks with GridPACK's deadband and limit on
controller passes; dc line updates between solves; and the driver's extra
limit check and second solve. All slots advance together, one step at a
time. A step runs, for the slots that need it: apply the previous
correction, check reactive limits, then evaluate (mismatch, Jacobian,
factor, solve). The host then reads back a handful of numbers per slot and
decides each slot's next action.

**Start values.** Every GPU case starts from the solved base case (the
production setting `warmStart = base_case`). Kundur and Malik recommend a
previously solved similar case over a flat start (sec. 6.4.5); stock
GridPACK starts each case from the RAW file's voltages, which the CPU paths
keep. The start changes the number of Newton steps, not the converged
answer beyond the tolerance.

**Health checks and CPU retry.** After each factorization and solve, a slot
is marked unhealthy if a pivot is tiny relative to its column in the
reference matrix (below 1e-12), if any value is not finite, or if the
linear solve leaves an error above 1e-6. An unhealthy or diverged slot does
not stop the batch; the case is solved again by GridPACK on a CPU rank,
from GridPACK's own start, before anything is reported as diverged. This is
the "second chance" of Wang et al., applied per case.

**dc lines and DG on the GPU.** The converter equations are GridPACK's own
inline functions compiled for the GPU, and the sequential method runs per
slot inside the control sequence. A dc pole outage changes only
injections, so it is a GPU case. DG is part of each bus's fixed demand.
These need plugin interface 1.1; an older plugin sends such networks to the
CPU loop.

### 5.6 Keeping the GPU busy

The host submits GPU cases in **chunks** of four batches (2,048 cases at
batch 512). Four mechanisms keep the slots full and the GPU working:

1. **Refilling slots.** When a case finishes, its slot takes the next queued
   case at once, also from the next chunk, so the batch empties only at the
   end of the study. Because memory is laid out by slot, not by case
   number, a new case in an old slot keeps the coalesced layout. Huang and
   Dinavahi had rejected refilling for that reason and left it for future
   work (44524).
2. **Light steps.** Steps that need no factorization (a reactive-limit check
   or the final update of a converged case) run at once in a small extra
   step, so the case leaves its slot before the next full step.
3. **Recorded launch sequences.** The hundreds of kernel launches of one
   factorization and of one solve are recorded once as CUDA graphs and
   replayed every step (2% faster factorization, 7% faster solve on
   ACTIVSg10k).
4. **One wait per step.** Each step waits once for its results; the measured
   GPU phases add up to the engine's whole time, so no idle time was left to
   remove.

The batch size is fixed at 512 in the tests. With `batchSize = auto` the
engine times one factorization and solve at several sizes and takes the
smallest within 5% of the best; the result is capped at the tested limit
(2,048 for both factorization methods) and by the memory budget (available
memory less a 10% headroom on unified-memory machines, less the GridPACK
ranks' share).

### 5.7 Reporting GPU results through GridPACK

GridPACK still produces every output. The run has two phases:

1. **CPU cases.** Cases classified for the CPU go through GridPACK's loop
   and shared counter exactly as in stock, while the GPU already works on
   its first chunks.
2. **GPU results.** The accelerator rank splits each finished chunk into
   packets of at most 16 cases (fewer when there are many ranks). Any rank
   asks for a packet, loads each case's voltages, angles, reactive outputs
   and final bus types into its own network copy ("state injection"), and
   runs GridPACK's checks and row writers. The accelerator rank reports
   packets itself when no one is asking. All hand-offs are non-blocking MPI
   messages.

For GPU cases the classifier already knows the topology, so reporting skips
GridPACK's full lone-bus and island searches and its network-wide
admittance and injection updates, and recomputes only the outaged elements
(`c9755ddb`). The rows are built as in Section 4.1. The final tables are
written in case order by all ranks (Section 4.2). Two extra tables are
written: one row per case describing its path and GPU outcome, and, when
`shadowFraction` is above zero, a comparison of each sampled GPU case with a
GridPACK re-solve.

### 5.8 Where the speed comes from

The GPU replaces the solve, which is most of the CPU case loop, by about a
second to a few tens of seconds of GPU time for the whole study. Reporting
cannot move to the GPU without giving up "GridPACK writes every output", so
after the solve moved, the case loop is set by reporting on the other ranks
and by the merge; Sections 4.1, 4.2 and 5.7 exist for that reason. Small
networks gain little: start-up, planning and the CPU retries of cases that
diverge on the GPU are a large part of their few seconds.

## 6. Alg 2: the custom batched factorization

Alg 2 is Algorithm 2 of Zhou et al. ("GPU-Based Batch" 4976):

- **Factorization** is left-looking, column by column, in the fixed order of
  the plan, without pivoting. Each level is one kernel launch. Within a
  launch, one thread block handles one column, and thread *t* handles slot
  *t*: it subtracts the contributions of the earlier columns the column
  depends on, then divides by the pivot. Because the plan stores the list
  of updates per column, no thread searches for structure at run time.
- **Triangular solves** use the same arrangement by rows, one launch per
  level, on the GPU. D'Orto et al. kept the solve of a single power flow on
  the CPU because it was faster there (56612); in a batch the result
  reverses: the host solve made a whole Polish study 2.2 times slower
  (29.2 against 13.3 s).
- **Long final columns.** The last levels of a factorization hold a few
  long columns. With one thread per case, each column is worked through by
  a single thread per slot while most of the GPU idles. Those levels now
  spread each column over up to 32 warps per case, chosen to give each
  multiprocessor about 16 warps; each warp takes every *n*-th entry of an
  update, with a barrier between updates. Every factor entry receives the
  same operations in the same order as before, so the factors are bitwise
  identical to one thread per case (checked by a unit test). The idea of
  giving narrow levels more threads is Wang et al.'s fine-grained
  parallelism for serial levels; their version uses atomic additions, this
  one does not. One factorization of 512 ACTIVSg10k cases fell from 511 to
  92 ms, and of 512 Texas7k cases from 1,055 to 136 ms.

Alg 2 is double precision throughout and uses no fast-math compiler
options.

## 7. cuDSS: the library factorization

cuDSS is NVIDIA's sparse direct solver; its "uniform batch" mode factorizes
many matrices with one pattern. Its use here:

- **Analysis once.** cuDSS's own reordering is switched off and the matrix
  is handed over already in the planner's KLU and AMD order, so both paths
  factorize the same permuted matrix. This made factorization 10–15% faster
  than cuDSS's own ordering.
- **Deterministic mode** is on, so repeated runs give identical files, at
  about 5% extra time.
- **Layout.** cuDSS wants each member's values stored one after another,
  while the engine stores members side by side, so two small kernels
  transpose the values and vectors each step. Members switched off by the
  mask keep the reference matrix in their slots, so they stay well defined.
- **Health.** cuDSS 0.8 does not expose the pivots of a uniform batch, so the
  finiteness and linear-error checks of Section 5.5 do this job.
- **Batch limit.** A defect above about 160 members per uniform batch had
  been reported for cuDSS 0.8.0, so the first limit was 128. With the planner's
  order and deterministic mode, batches up to 2,048 matched Alg 2 on the
  Texas7k and ACTIVSg10k Jacobians, and the limit was raised to 2,048
  (`9937c912`).

In isolated timings of one factorization and solve of 512 ACTIVSg10k cases
on the GB10, Alg 2 took 110 ms and cuDSS 186 ms.

## 8. How the current design was reached

The work followed one rule, written into the design guide before any code:
measure the whole study before tuning any part of it (guide §2.6). Each
step below was chosen because the previous measurement showed where the
time went. Times are 16-rank studies unless stated.

| Step | Commit | Finding that led to it | Result |
|---|---|---|---|
| Design guide 0.1 to 0.3 | docs | Version 0.1 was a standalone solver that sent every structure-changing case (slack moves, isolated buses, reactive-limit switching) to the CPU. Reusing GridPACK for everything but the solve, and the superset pattern, kept those cases on the GPU. | One plan for almost every case; GridPACK's outputs by construction |
| First GPU path | `7aededd4`, `4cb05851` | Zhou's Algorithm 2 and cuDSS behind one interface; cuDSS limited to batches of 128 | Polish at 8 ranks: 13.3 s against 26.9 s for stock (2.0×); at 16 ranks 1.5× |
| Stage timers | `d62432ec`, `e656133e` | Almost all time is the case loop. On Texas7k and ACTIVSg10k the GPU was busy for under half of it; the rest was GridPACK reporting (103 of 194 s per rank on Texas7k, 95 s of it rows) | Reporting identified as the bottleneck |
| Shadow audit | `131827e3`, `c81a8387`, `2d3bcbb2` | Re-solving every GPU case on the CPU exposed the voltage-setting leak and the disconnected-bus limit check in stock GridPACK | Both fixed in both paths (Section 4.4) |
| Rows from numbers | `4ffdeb13` | Text round trip for every row | ACTIVSg10k Alg 2: 181.5 → 139.9 s; Texas7k CPU: 150 → 112 s |
| Known-topology reporting | `c9755ddb` | Applying, injecting and restoring GPU cases redid network-wide work the classifier had already done | ACTIVSg10k: 41.7 → 5.3 s per rank for those steps; 139.9 → 130.0 s |
| Several warps per long column | `a6ce62aa` | Now the GPU limited the loop (busy 155 of 158 s on Texas7k); the factorization's last levels ran on one thread per case | Factorization 114.5 → 21.2 s (ACTIVSg10k); studies 130.0 → 44.6 s (ACTIVSg10k), 157.8 → 33.6 s (Texas7k) |
| Refilling slots, light steps | `9669c544` | Batches drained while their slowest cases finished: 40% of slots busy | 72% occupancy; 44.6 → 39.7 s (ACTIVSg10k), 33.6 → 27.2 s (Texas7k) |
| cuDSS order, determinism, limit | `9937c912` | cuDSS reordered a matrix the planner had already ordered; results varied between runs | 10–15% faster factorization; identical files; batches to 2,048 |
| Multi-rank writer | `27042644` | Rank 0's merge of a 17 GB table took 8.7 s of a 42 s study | 4.0 s; ACTIVSg10k GPU study 36.5 s against 330.9 s for stock (9.1×) |
| Planning-case compatibility | `7ba457e3` … `0e6d3b9d` | Stock GridPACK could not solve a 28,000-bus PSS/E v34 case | Parser fixes, DG, dc lines on CPU and GPU, plugin interface 1.1 |

Each optimization step left every output byte-identical. The validation
record covers 70 full studies (35 networks, both GPU methods) with a CPU
re-solve of every GPU case: 142,337 re-solves, all with the same status and
exact PV/PQ bus sets as GridPACK, and a largest voltage difference of
4.8e-12 per unit (`docs/gpu_n1/validation.md`).

## 9. Reproducing the system

Everything below is in the repository; `tools/ca_matrix/README.md` gives the
commands.

1. **Build** stock GridPACK at `b32969b0` with the timer-only patch, and this
   branch with `GRIDPACK_ENABLE_GPU_BATCH=ON`, both as release builds
   (`-O3 -DNDEBUG`) in the same container image
   (`tools/ca_matrix/build.sh`; image `alh360/gridpack-n1-tools:1.2`:
   Ubuntu 24.04, CUDA 13.0, cuDSS 0.8.0, OpenMPI 4.1.6, and PETSc with
   SuiteSparse KLU and Global Arrays as GridPACK's Dockerfile builds them).
2. **Configure** with `test_runs/input.xml` (Section 2 of
   [03-results.md](03-results.md) lists the settings). Only the path
   (`GPUBatch/enabled` and `backend`), `outputFile` and
   `networkConfiguration` change between runs.
3. **Run** `mpiexec --bind-to none -n <ranks> ca.x input.xml`.
4. **Compare** the `csv_flat` tables row by row on (case, from bus, to bus,
   circuit) (`tools/ca_matrix/compare_runs.py`).

To rebuild the GPU path from this description alone, the essential parts
are: the superset pattern of Section 5.3, one KLU and AMD plan of the
base-case Jacobian, a slot-major data layout, GridPACK's equations and
control sequence per slot, a per-slot health check with a GridPACK retry,
slot refilling, and reporting by injecting each GPU state into a GridPACK
network copy.

## References

- D'Orto, M., et al. "Comparing Different Approaches for Solving Large Scale
  Power-Flow Problems with the Newton-Raphson Method." *IEEE Access*, vol.
  9, 2021, pp. 56604–15.
- Huang, S., and V. Dinavahi. "Real-Time Contingency Analysis on Massively
  Parallel Architectures with Compensation Method." *IEEE Access*, vol. 6,
  2018, pp. 44519–30.
- Khan, S., and S. Bhowmick. *Power-Flow Modelling of HVDC Transmission
  Systems*. CRC Press, 2023, sec. 1.7.
- Kundur, P. S., and O. P. Malik. *Power System Stability and Control*, 2nd
  ed., McGraw Hill, 2022, sec. 6.4 and ch. 10.
- Meena, G., et al. "Steady-State Models of AC Microgrid for Load Flow
  Solutions." *ISTEMS 2024*, IEEE, 2024.
- North American Electric Reliability Corporation. *Reliability Guideline:
  Distributed Energy Resource Modeling*. 2017.
- NVIDIA. *cuDSS Documentation* (release notes 0.6–0.8) and *CUDA C++ Best
  Practices Guide*.
- Palmer, B., et al. "GridPACK: A Framework for Developing Power Grid
  Simulations on High Performance Computing Platforms." *WOLFHPC 2014*,
  IEEE, 2014, pp. 68–77.
- Siemens PTI. *PSS/E Program Operation Manual*, RAW format, versions 33–36.
- Tarjan, R. E. "A Note on Finding the Bridges of a Graph." *Information
  Processing Letters*, vol. 2, no. 6, 1974, pp. 160–61.
- Wang, Z., S. Wende-von Berg, and M. Braun. "Fast Parallel Newton-Raphson
  Power Flow Solver for Large Number of System Calculations with CPU and
  GPU." *Sustainable Energy, Grids and Networks*, vol. 27, 2021, 100483
  (arXiv:2101.02270).
- Zhou, G., et al. "GPU-Based Batch LU-Factorization Solver for Concurrent
  Analysis of Massive Power Flows." *IEEE Transactions on Power Systems*,
  vol. 32, no. 6, 2017, pp. 4975–77.
