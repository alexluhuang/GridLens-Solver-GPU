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
fail a numerical check, go to the CPU path. Building and writing the
result tables also had to be made faster, because once the solves moved to
the GPU these steps set the runtime.

On one NVIDIA DGX Spark, across nine networks of 993 to 29,637 buses, the GPU
paths ran whole studies 6.4–9.1 times faster than unmodified GridPACK with 16
CPU processes on the six public networks above 5,000 buses, and 8.3–10.2
times faster with 8 processes. On a 29,637-bus planning case on which
unmodified GridPACK aborts, they ran 7.2–9.0 times faster than the optimized
CPU path, which keeps GridPACK's CPU solver, corrects two of its defects and
adds the models the case needs. Every
case received the same outcome on every version except one ill-conditioned
case. Flows and voltages agreed with the optimized CPU path to the last
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
(01-literature §5.3): whether a full N-1 study fits such a batch, including
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
01-literature §5. Several parts appeared separately in open-source tools in
the months before this work (01-literature §5.2), so the contribution is the
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
   library; every case the batch cannot represent goes to the CPU path, which
   solves it or, for an outage that splits the grid, reports it as islanded.
8. **Network models for PSS/E v34 planning cases**: parser fixes,
   distributed generation on loads, and line-commutated dc lines with the
   sequential ac/dc method on both CPU and GPU.

Evaluation:

9. A **whole-study evaluation** of four versions on nine networks at three
   process counts, with every pipeline stage timed and every result row
   compared.

## 5. Background

This section introduces the concepts the rest of the report uses.
02-methodology §2.2 and §3.3 give the exact equations and control rules the
system implements.

### 5.1 The network and the power-flow problem

A transmission network is modelled as a graph. Its nodes are **buses**
(substation busbars) and its edges are **branches** (lines and
transformers). Generators, loads and shunt devices are attached to buses.
Each branch is an equivalent π circuit: a series impedance between its two
buses and a shunt admittance at each end. A transformer adds a tap ratio and
possibly a phase shift (Kundur and Malik, sec. 6.4). Quantities are in
**per unit**: power is divided by a system base of 100 MVA, so
$10^{-4}$ per unit is 0.01 MW or MVAr, and each voltage is divided by its
bus's nominal voltage, so normal voltages are near 1.0.

The branches and shunts define the **bus admittance matrix**
$\mathbf{Y} = \mathbf{G} + j\mathbf{B}$, with entries
$Y_{km} = G_{km} + jB_{km}$, where $j$ is the imaginary unit, $G_{km}$ the
conductance and $B_{km}$ the susceptance. An off-diagonal entry $Y_{km}$ is
minus the series admittance of the branches between buses $k$ and $m$; a
diagonal entry $Y_{kk}$ is the sum of all admittances attached to bus $k$.
$\mathbf{Y}$ is sparse because a bus connects to only a few others:
ACTIVSg10k has 10,000 buses, about 12,000 branches, and roughly 34,000
nonzero entries in $\mathbf{Y}$ out of $10^8$.

The **power-flow problem** is to find the voltage magnitude and angle at
every bus, given what each generator produces and each load consumes. The
real and reactive power injected into the network at bus $k$ follow from
the voltages:

```math
\begin{aligned}
P_k(\boldsymbol{\theta}, \mathbf{V}) &= V_k^2 G_{kk} + V_k \sum_{m \in \mathcal{N}(k)} V_m \left( G_{km} \cos\theta_{km} + B_{km} \sin\theta_{km} \right) \\
Q_k(\boldsymbol{\theta}, \mathbf{V}) &= -V_k^2 B_{kk} + V_k \sum_{m \in \mathcal{N}(k)} V_m \left( G_{km} \sin\theta_{km} - B_{km} \cos\theta_{km} \right)
\end{aligned}
```

where

- $P_k(\boldsymbol{\theta}, \mathbf{V})$, $Q_k(\boldsymbol{\theta}, \mathbf{V})$
  are the real and reactive power injected into the network at bus $k$, in
  per unit, as functions of the bus voltages;
- $n$ is the number of buses, indexed $1, \dots, n$ by position (not by
  their numbers in the network file);
- $\boldsymbol{\theta} = (\theta_1, \dots, \theta_n)$ and
  $\mathbf{V} = (V_1, \dots, V_n)$ are the bus voltage angles (radians) and
  magnitudes (per unit), and $\theta_k$, $V_k$ those of bus $k$;
- $\theta_{km} = \theta_k - \theta_m$ is the angle difference between buses
  $k$ and $m$;
- $G_{km}$, $B_{km}$ are the conductance and susceptance of entry $Y_{km}$,
  and $G_{kk}$, $B_{kk}$ those of the diagonal entry $Y_{kk}$;
- $\mathcal{N}(k)$ is the set of buses joined to bus $k$ by at least one
  branch in service, not including $k$ itself;
- $m$ is the summation index: one neighbouring bus of $k$.

Kundur and Malik write one sum over all $n$ buses (eq. 6.101). The two forms
are equal: a bus not joined to $k$ has $G_{km} = B_{km} = 0$, so its term
vanishes, and the term $m = k$, where $\cos\theta_{kk} = 1$ and
$\sin\theta_{kk} = 0$, is the first term above. Writing the sum over
$\mathcal{N}(k)$ shows that each injection involves only bus $k$ and its
neighbours, which is why the Jacobian of Section 5.2 is sparse.

The equations are nonlinear in $\mathbf{V}$ and $\boldsymbol{\theta}$ and
are solved iteratively. The **mismatch** at a bus is the difference between
the computed injection and the **specified** one; Kundur and Malik use the
superscript sp "to denote specified values" (before eq. 6.102).

Each bus fixes two of the four quantities $P$, $Q$, $V$ and $\theta$
according to its **bus type**, and so contributes zero, one or two
equations:

| Bus type | Known | Unknown | Equations |
|---|---|---|---|
| **PQ** (most buses: loads) | $P_k$, $Q_k$ | $\theta_k$, $V_k$ | $P_k(\boldsymbol{\theta}, \mathbf{V}) = P_k^{\mathrm{sp}}$ and $Q_k(\boldsymbol{\theta}, \mathbf{V}) = Q_k^{\mathrm{sp}}$ |
| **PV** (a generator holds $V_k$ at a set point) | $P_k$, $V_k$ | $\theta_k$, $Q_k$ | $P_k(\boldsymbol{\theta}, \mathbf{V}) = P_k^{\mathrm{sp}}$ |
| **Slack** (reference; balances the losses) | $V_k$, $\theta_k = 0$ | $P_k$, $Q_k$ | none |

Take bus 1 as the slack. The power-flow problem is then the system (Kundur
and Malik, eq. 6.102)

```math
\begin{aligned}
P_k(\boldsymbol{\theta}, \mathbf{V}) &= P_k^{\mathrm{sp}}, & k &\in \mathcal{PV} \cup \mathcal{PQ} \quad (n - 1 \text{ equations}) \\
Q_k(\boldsymbol{\theta}, \mathbf{V}) &= Q_k^{\mathrm{sp}}, & k &\in \mathcal{PQ} \quad (n_{PQ} \text{ equations})
\end{aligned}
```

where

- $P_k^{\mathrm{sp}}$, $Q_k^{\mathrm{sp}}$ are the specified net real and
  reactive power injection at bus $k$, in per unit: the bus's scheduled
  generation minus its load, read from the network file. $Q_k^{\mathrm{sp}}$
  is specified only at PQ buses, because at a PV bus $Q_k$ is unknown;
- $\mathcal{PV}$ and $\mathcal{PQ}$ are the sets of PV and PQ buses; the
  slack bus belongs to neither;
- $n_{PQ}$ is the number of PQ buses, the size of $\mathcal{PQ}$.

The unknowns are the $n - 1 + n_{PQ}$ quantities $\theta_k$ ($k \neq 1$)
and $V_k$ ($k \in \mathcal{PQ}$), one per equation.

A generator can supply reactive power only between its limits $Q^{\min}$
and $Q^{\max}$. If holding the set voltage would need more, the generator
is fixed at the limit and its bus becomes PQ (Kundur and Malik, sec.
6.4.2(b)). This **reactive-limit switching** changes the set of equations
during the solve.

The solved voltages determine the flow on every branch. A branch is
**overloaded** when its apparent power exceeds its rating; a bus violates its
**voltage limits** when $V_k$ lies outside a band such as 0.9–1.1 per unit.

### 5.2 Newton's method

Write the unknowns as a vector $\mathbf{x}$ and the equations as
$\mathbf{f}(\mathbf{x}) = \mathbf{b}$. Newton's method improves an estimate
$\mathbf{x}^0$ by looking for a correction $\Delta\mathbf{x}$ such that the
new point $\mathbf{x}^0 - \Delta\mathbf{x}$ satisfies every equation, and
expands each equation around $\mathbf{x}^0$ with Taylor's theorem (Kundur
and Malik, eqs. 6.93–6.96):

```math
f_i(\mathbf{x}^0 - \Delta\mathbf{x}) = f_i(\mathbf{x}^0) - \sum_{j=1}^{N_{\mathrm{eq}}} \left. \frac{\partial f_i}{\partial x_j} \right|_{\mathbf{x}^0} \Delta x_j + O\!\left(\lVert \Delta\mathbf{x} \rVert^2\right) = b_i, \qquad i = 1, \dots, N_{\mathrm{eq}},
```

where

- $N_{\mathrm{eq}}$ is the number of equations and of unknowns
  ($n - 1 + n_{PQ}$ for the power flow);
- $\mathbf{x} = (x_1, \dots, x_{N_{\mathrm{eq}}})$ is the vector of unknowns;
  for the power flow, the angles and voltage magnitudes of Section 5.1;
- $\mathbf{f}(\mathbf{x}) = (f_1(\mathbf{x}), \dots, f_{N_{\mathrm{eq}}}(\mathbf{x}))$
  holds the computed left-hand sides; for the power flow, the injections
  $P_k(\boldsymbol{\theta}, \mathbf{V})$ and
  $Q_k(\boldsymbol{\theta}, \mathbf{V})$;
- $\mathbf{b} = (b_1, \dots, b_{N_{\mathrm{eq}}})$ holds the specified
  right-hand sides; for the power flow, $P_k^{\mathrm{sp}}$ and
  $Q_k^{\mathrm{sp}}$;
- $\mathbf{x}^0$ is the current estimate;
- $\Delta\mathbf{x} = (\Delta x_1, \dots, \Delta x_{N_{\mathrm{eq}}})$ is the
  correction to be found;
- $\left. \partial f_i / \partial x_j \right|_{\mathbf{x}^0}$ is the partial
  derivative of equation $i$ with respect to unknown $j$, evaluated at
  $\mathbf{x}^0$;
- $O(\lVert \Delta\mathbf{x} \rVert^2)$ is the remainder of the expansion:
  the second- and higher-order terms, which are at most a constant times
  $\lVert \Delta\mathbf{x} \rVert^2$ when $\Delta\mathbf{x}$ is small;
- $\lVert \cdot \rVert$ is a vector norm.

If $\mathbf{x}^0$ is close to the solution, the remainder is much smaller
than the linear term, and Kundur and Malik drop it: "the terms of higher
powers can be neglected, if the initial estimate is close to the true
solution". Moving the sum to the right and $b_i$ to the left gives, for each
equation, $\sum_j J_{ij} \Delta x_j = F_i$. Stacking the $N_{\mathrm{eq}}$
equations as rows gives one linear system per step:

```math
\mathbf{J}\,\Delta\mathbf{x} = \mathbf{F}, \qquad
J_{ij} = \left. \frac{\partial f_i}{\partial x_j} \right|_{\mathbf{x}^0}, \qquad
F_i = f_i(\mathbf{x}^0) - b_i, \qquad
\mathbf{x}^1 = \mathbf{x}^0 - \Delta\mathbf{x} ,
```

where

- $\mathbf{J}$ is the **Jacobian**, the $N_{\mathrm{eq}} \times N_{\mathrm{eq}}$
  matrix of partial derivatives $J_{ij}$ at $\mathbf{x}^0$;
- $\mathbf{F}$ is the **mismatch vector**, with entries $F_i$: computed
  minus specified, at $\mathbf{x}^0$;
- $\Delta\mathbf{x}$ is the correction, now the solution of a linear system;
- $\mathbf{x}^1$ is the next estimate.

$\mathbf{J}$ and $\mathbf{F}$ are evaluated at the known point
$\mathbf{x}^0$, so the system is linear in $\Delta\mathbf{x}$. Because the
dropped remainder is of order $\lVert \Delta\mathbf{x} \rVert^2$,
$\mathbf{x}^1$ is not exact, but its error is roughly the square of that of
$\mathbf{x}^0$; repeating the step converges quadratically (Kundur and
Malik, sec. 6.4.3). Each step therefore:

1. evaluates the mismatch $\mathbf{F}$ at the current $\mathbf{x}$;
2. builds the Jacobian $\mathbf{J}$ at the current $\mathbf{x}$;
3. solves $`\mathbf{J}\,\Delta\mathbf{x} = \mathbf{F}`$ for the correction;
4. updates $\mathbf{x} \leftarrow \mathbf{x} - \Delta\mathbf{x}$.

The sign convention is GridPACK's (02-methodology §2.2): the mismatch is
computed minus specified. Kundur and Malik write specified minus computed
and add the correction; the two give the same step.

For the power flow, grouping the angle unknowns and the real-power
equations first gives the block form (Kundur and Malik, eq. 6.103; D'Orto et
al., eq. 3):

```math
\underbrace{\begin{bmatrix} \mathbf{H} & \mathbf{N} \\ \mathbf{M} & \mathbf{L} \end{bmatrix}}_{\mathbf{J}}
\underbrace{\begin{bmatrix} \Delta\boldsymbol{\theta} \\ \Delta\mathbf{V} \end{bmatrix}}_{\Delta\mathbf{x}}
=
\underbrace{\begin{bmatrix} \Delta\mathbf{P} \\ \Delta\mathbf{Q} \end{bmatrix}}_{\mathbf{F}}
```

where

- $\mathbf{H}$, $\mathbf{N}$, $\mathbf{M}$, $\mathbf{L}$ are the four blocks
  of $\mathbf{J}$, holding the derivatives of real power with respect to
  angles, real power with respect to voltage magnitudes, reactive power with
  respect to angles, and reactive power with respect to voltage magnitudes
  (entries below);
- $\Delta\boldsymbol{\theta}$ and $\Delta\mathbf{V}$ are the angle and
  voltage-magnitude parts of the correction $\Delta\mathbf{x}$;
- $\Delta\mathbf{P}$ and $\Delta\mathbf{Q}$ are the real- and reactive-power
  parts of the mismatch $\mathbf{F}$. Here $\Delta$ denotes a mismatch, not a
  change; the notation is Kundur and Malik's.

The four vectors are

```math
\Delta\boldsymbol{\theta} = \begin{bmatrix} \Delta\theta_2 \\ \vdots \\ \Delta\theta_n \end{bmatrix}, \quad
\Delta\mathbf{V} = \begin{bmatrix} \Delta V_{k_1} \\ \vdots \\ \Delta V_{k_{n_{PQ}}} \end{bmatrix}, \quad
\Delta\mathbf{P} = \begin{bmatrix} P_2(\boldsymbol{\theta}, \mathbf{V}) - P_2^{\mathrm{sp}} \\ \vdots \\ P_n(\boldsymbol{\theta}, \mathbf{V}) - P_n^{\mathrm{sp}} \end{bmatrix}, \quad
\Delta\mathbf{Q} = \begin{bmatrix} Q_{k_1}(\boldsymbol{\theta}, \mathbf{V}) - Q_{k_1}^{\mathrm{sp}} \\ \vdots \\ Q_{k_{n_{PQ}}}(\boldsymbol{\theta}, \mathbf{V}) - Q_{k_{n_{PQ}}}^{\mathrm{sp}} \end{bmatrix},
```

where

- $\Delta\theta_k$ and $\Delta V_k$ are the corrections to $\theta_k$ and
  $V_k$;
- $k_1 < k_2 < \dots < k_{n_{PQ}}$ are the PQ buses in increasing index
  order;
- $P_k(\boldsymbol{\theta}, \mathbf{V}) - P_k^{\mathrm{sp}}$ and
  $Q_k(\boldsymbol{\theta}, \mathbf{V}) - Q_k^{\mathrm{sp}}$ are the real-
  and reactive-power mismatches at bus $k$, computed at the current
  estimate.

$\Delta\boldsymbol{\theta}$ and $\Delta\mathbf{P}$ have $n - 1$ entries, one
per non-slack bus; $\Delta\mathbf{V}$ and $\Delta\mathbf{Q}$ have $n_{PQ}$,
one per PQ bus. $\mathbf{J}$ is therefore square, of order
$N_{\mathrm{eq}} = n - 1 + n_{PQ}$. Written out entry by entry,

```math
\mathbf{J} =
\left[\begin{array}{ccc|ccc}
\dfrac{\partial P_2}{\partial \theta_2} & \cdots & \dfrac{\partial P_2}{\partial \theta_n} & \dfrac{\partial P_2}{\partial V_{k_1}} & \cdots & \dfrac{\partial P_2}{\partial V_{k_{n_{PQ}}}} \\
\vdots & & \vdots & \vdots & & \vdots \\
\dfrac{\partial P_n}{\partial \theta_2} & \cdots & \dfrac{\partial P_n}{\partial \theta_n} & \dfrac{\partial P_n}{\partial V_{k_1}} & \cdots & \dfrac{\partial P_n}{\partial V_{k_{n_{PQ}}}} \\ \hline
\dfrac{\partial Q_{k_1}}{\partial \theta_2} & \cdots & \dfrac{\partial Q_{k_1}}{\partial \theta_n} & \dfrac{\partial Q_{k_1}}{\partial V_{k_1}} & \cdots & \dfrac{\partial Q_{k_1}}{\partial V_{k_{n_{PQ}}}} \\
\vdots & & \vdots & \vdots & & \vdots \\
\dfrac{\partial Q_{k_{n_{PQ}}}}{\partial \theta_2} & \cdots & \dfrac{\partial Q_{k_{n_{PQ}}}}{\partial \theta_n} & \dfrac{\partial Q_{k_{n_{PQ}}}}{\partial V_{k_1}} & \cdots & \dfrac{\partial Q_{k_{n_{PQ}}}}{\partial V_{k_{n_{PQ}}}}
\end{array}\right]
```

where

- each entry is a partial derivative of an injection function of Section
  5.1, evaluated at the current estimate; row $i$ belongs to one equation and
  column $j$ to one unknown;
- the lines separate the blocks: $\mathbf{H}$ top left
  ($(n-1) \times (n-1)$), $\mathbf{N}$ top right ($(n-1) \times n_{PQ}$),
  $\mathbf{M}$ bottom left ($n_{PQ} \times (n-1)$) and $\mathbf{L}$ bottom
  right ($n_{PQ} \times n_{PQ}$).

Kundur and Malik name $\mathbf{H}$ and $\mathbf{L}$ (sec. 6.4.4); D'Orto et
al. call the blocks $J_1$ to $J_4$. Differentiating the injection equations
of Section 5.1 gives every entry (Kundur and Malik, sec. 6.4.4; D'Orto et
al., eqs. 4–7):

| Entry | Off-diagonal ($m \neq k$, buses joined) | Diagonal ($m = k$) |
|---|---|---|
| $H_{km} = \partial P_k / \partial \theta_m$ | $V_k V_m (G_{km} \sin\theta_{km} - B_{km} \cos\theta_{km})$ | $-Q_k - B_{kk} V_k^2$ |
| $N_{km} = \partial P_k / \partial V_m$ | $V_k (G_{km} \cos\theta_{km} + B_{km} \sin\theta_{km})$ | $P_k / V_k + G_{kk} V_k$ |
| $M_{km} = \partial Q_k / \partial \theta_m$ | $-V_k V_m (G_{km} \cos\theta_{km} + B_{km} \sin\theta_{km})$ | $P_k - G_{kk} V_k^2$ |
| $L_{km} = \partial Q_k / \partial V_m$ | $V_k (G_{km} \sin\theta_{km} - B_{km} \cos\theta_{km})$ | $Q_k / V_k - B_{kk} V_k$ |

where

- $H_{km}$, $N_{km}$, $M_{km}$, $L_{km}$ are the entries of the four blocks
  in the row of bus $k$'s equation and the column of bus $m$'s unknown;
- $P_k$ and $Q_k$ are the injections $P_k(\boldsymbol{\theta}, \mathbf{V})$
  and $Q_k(\boldsymbol{\theta}, \mathbf{V})$ computed at the current
  estimate;
- $V_k$, $V_m$, $\theta_{km}$, $G_{km}$, $B_{km}$, $G_{kk}$ and $B_{kk}$ are
  as in Section 5.1.

The diagonal entries differentiate both the $m = k$ term and the sum of
Section 5.1 with respect to bus $k$'s own variables, and are then simplified
using $P_k$ and $Q_k$. D'Orto et al. print $V_m$ instead of $V_k$ as the
factor of $\partial P_k / \partial V_m$; the form above follows from
differentiating $P_k$. If buses $k$ and $m$ are not joined,
$G_{km} = B_{km} = 0$ and all four entries are zero, so each block has the
sparsity pattern of $\mathbf{Y}$ (Kundur and Malik, after eq. 6.103). A PV
bus has no $Q$ equation and no $V$ unknown, so it has one row and one column
instead of two.

*Example.* In a three-bus network where bus 1 is the slack, bus 2 is PV,
bus 3 is PQ and every pair is joined, the unknowns are $\theta_2$,
$\theta_3$ and $V_3$, and the equations are those of $P_2$, $P_3$ and
$Q_3$:

```math
\begin{bmatrix}
H_{22} & H_{23} & N_{23} \\
H_{32} & H_{33} & N_{33} \\
M_{32} & M_{33} & L_{33}
\end{bmatrix}
\begin{bmatrix} \Delta\theta_2 \\ \Delta\theta_3 \\ \Delta V_3 \end{bmatrix}
=
\begin{bmatrix}
P_2(\boldsymbol{\theta}, \mathbf{V}) - P_2^{\mathrm{sp}} \\
P_3(\boldsymbol{\theta}, \mathbf{V}) - P_3^{\mathrm{sp}} \\
Q_3(\boldsymbol{\theta}, \mathbf{V}) - Q_3^{\mathrm{sp}}
\end{bmatrix}
```

where

- $H_{km}$, $N_{km}$, $M_{km}$, $L_{km}$ are the Jacobian entries of the
  table above for buses $k, m \in \{2, 3\}$;
- $\Delta\theta_2$, $\Delta\theta_3$ and $\Delta V_3$ are the corrections to
  the three unknowns;
- $P_2(\boldsymbol{\theta}, \mathbf{V})$, $P_3(\boldsymbol{\theta}, \mathbf{V})$
  and $Q_3(\boldsymbol{\theta}, \mathbf{V})$ are the computed injections, and
  $P_2^{\mathrm{sp}}$, $P_3^{\mathrm{sp}}$ and $Q_3^{\mathrm{sp}}$ the
  specified ones.

From the table, the entries are

```math
\begin{aligned}
H_{22} &= -Q_2 - B_{22} V_2^2, & H_{23} &= V_2 V_3 (G_{23} \sin\theta_{23} - B_{23} \cos\theta_{23}), & N_{23} &= V_2 (G_{23} \cos\theta_{23} + B_{23} \sin\theta_{23}), \\
H_{32} &= V_3 V_2 (G_{32} \sin\theta_{32} - B_{32} \cos\theta_{32}), & H_{33} &= -Q_3 - B_{33} V_3^2, & N_{33} &= P_3 / V_3 + G_{33} V_3, \\
M_{32} &= -V_3 V_2 (G_{32} \cos\theta_{32} + B_{32} \sin\theta_{32}), & M_{33} &= P_3 - G_{33} V_3^2, & L_{33} &= Q_3 / V_3 - B_{33} V_3 ,
\end{aligned}
```

where

- $V_2$, $V_3$ are the voltage magnitudes of buses 2 and 3 at the current
  estimate ($V_2$ is fixed at its set point);
- $\theta_{23} = \theta_2 - \theta_3$ and $\theta_{32} = -\theta_{23}$;
- $G_{22}$, $B_{22}$, $G_{33}$, $B_{33}$ are diagonal entries of
  $\mathbf{Y}$, and $G_{23}$, $B_{23}$, $G_{32}$, $B_{32}$ off-diagonal ones
  ($Y_{23} = Y_{32}$ unless a phase-shifting transformer joins the two
  buses);
- $P_3$, $Q_2$, $Q_3$ are the injections computed at the current estimate.

The known quantities $V_1$, $\theta_1 = 0$ and $V_2$ do not appear in
$\Delta\mathbf{x}$, but they enter the entries through $P_k$, $Q_k$ and
$\theta_{km}$. $Q_2$ appears in $H_{22}$ as the value computed from the
current voltages, although bus 2 has no $Q$ equation.

The GPU paths of this work arrange the same equations so that the sparsity
pattern does not change when bus types do; 02-methodology §2.2 shows this
example in that layout.

*Convergence.* The iteration **converges** when the largest absolute
mismatch falls below a tolerance:

```math
\lVert \mathbf{F} \rVert_\infty = \max_{i} \lvert F_i \rvert < \varepsilon ,
```

where

- $\lVert \mathbf{F} \rVert_\infty$ is the infinity norm of the mismatch
  vector, its largest entry in absolute value;
- $F_i$ are the entries of $\mathbf{F}$, in per unit;
- $\varepsilon$ is the tolerance, $10^{-4}$ per unit (0.01 MW) here.

Because convergence is quadratic, two or three steps suffice from a good
starting point. A case **diverges** when it
reaches a step limit or its mismatch grows. Step 3 is the most expensive;
D'Orto et al. measured 81–93% of a single power flow's time in it (56611).

Newton's method stops at the first point whose mismatch is below the
tolerance. Two correct implementations that start from different points, or
round differently, can therefore stop at slightly different points, and
their flows can differ by about the tolerance (0.01 MW), more on a poorly
conditioned case.

### 5.3 Sparse direct solution

Every Newton step solves $\mathbf{J}\,\Delta\mathbf{x} = \mathbf{F}$, so the
cost of this solve sets the cost of a power flow. The system is large but
almost empty. For ACTIVSg10k the Jacobian used here has 20,000 rows and
137,736 nonzero entries, about seven per row (03-results Table 2). Dense
Gaussian elimination would take about $\tfrac{2}{3} N_{\mathrm{eq}}^3$, or
$5 \times 10^{12}$, multiply-adds and store $N_{\mathrm{eq}}^2$, or
$4 \times 10^8$, values. A **sparse direct solver** performs the same
elimination but stores and updates only nonzero entries; on ACTIVSg10k one
factorization takes 2.2 million multiply-adds. How much work remains
depends on how many new nonzeros the elimination creates, and that depends
on the order in which the unknowns are eliminated. The rest of this section
follows that chain: elimination, the fill it creates, the order that limits
the fill, how the order is written, how it is kept numerically safe, and why
it can be computed once and reused.

*Elimination produces two triangular factors.* Gaussian elimination removes
the unknowns one at a time. At step $j$ it uses row $j$, the **pivot row**,
to remove unknown $j$ from every later row $i$ that contains it:

```math
\ell_{ij} = \frac{a_{ij}}{a_{jj}}, \qquad
a_{ik} \leftarrow a_{ik} - \ell_{ij}\, a_{jk} \quad (k > j), \qquad
a_{ij} \leftarrow 0, \qquad
\text{for every } i > j \text{ with } a_{ij} \neq 0,
```

where

- $a_{ij}$ is the current value of entry $(i, j)$ of the matrix being
  factorized, updated in place as elimination proceeds;
- $j$ is the step, which is also the index of the pivot row and column, and
  $a_{jj}$ is the **pivot**;
- $i$ is a later row and $k$ a later column;
- $\ell_{ij}$ is the multiplier that cancels entry $(i, j)$.

After the last step the entries left on and above the diagonal form an
upper triangular matrix $\mathbf{U}$, and the multipliers form a lower
triangular matrix $\mathbf{L}$ ($L_{ij} = \ell_{ij}$, with ones on the
diagonal), so the matrix equals $\mathbf{L}\mathbf{U}$. This is **LU
factorization**; Kundur and Malik describe it as "triangular factorization
due to Gauss elimination" (after eq. 6.115). Once the factors exist, a
system with any right-hand side is solved by two cheap substitutions, shown
below with the ordering.

*Elimination creates fill.* The update
$a_{ik} \leftarrow a_{ik} - \ell_{ij} a_{jk}$ writes to position $(i, k)$
whenever row $i$ has a nonzero in column $j$ and row $j$ has a nonzero in
column $k$. If $a_{ik}$ was zero, it becomes nonzero. Such a new entry is
**fill**: it must be stored, and later steps update it like any other entry.
In graph terms, draw one node per unknown and an edge between $i$ and $k$
whenever $a_{ik} \neq 0$. Eliminating node $j$ joins every pair of its
remaining neighbours, because each such pair receives an update; when a
variable is eliminated "a clique is formed by its neighbours", and "each of
the edges within the clique contributes to the fill-in" (D'Orto et al.
56607). The growth is measured by

```math
\text{fill-in ratio} = \frac{\mathrm{nnz}(\mathbf{L}) + \mathrm{nnz}(\mathbf{U})}{\mathrm{nnz}(\mathbf{J})} ,
```

where $\mathrm{nnz}(\cdot)$ is the number of stored entries of a matrix (the
unit diagonal of $\mathbf{L}$ is not stored). At 2.0 the factors hold twice
as many entries as the Jacobian, half of them fill.

Fill sets three costs of every factorization:

1. **Memory.** Every fill entry is stored, at 8 bytes per value. A solver
   that factorizes many matrices at once stores one set of factors per
   matrix, so its memory grows with the fill times the number of matrices.
2. **Arithmetic.** Step $j$ performs one multiply-add for each pair of a
   nonzero below the pivot in column $j$ and a nonzero right of the pivot in
   row $j$, so a factorization costs

   ```math
   \sum_{j} c_j\, r_j \ \text{multiply-adds},
   ```

   where $c_j$ is the number of nonzeros of $\mathbf{L}$ below the diagonal
   in column $j$ and $r_j$ the number of nonzeros of $\mathbf{U}$ right of
   the diagonal in row $j$. Fill raises both counts. On the networks tested,
   fill-in ratios of 1.34 to 3.30 gave 0.15 to 10.1 million multiply-adds per
   factorization (03-results Table 2).
3. **Parallelism.** Fill entries in $\mathbf{U}$ make later columns depend
   on earlier ones (see the last paragraph of this section), which can
   lengthen the part of the factorization that must run in sequence.

*The elimination order decides the fill.* The same equations eliminated in a
different order give different fill. A star-shaped pattern shows the
extremes:

```math
\underbrace{\begin{bmatrix}
\times & \times & \times & \times \\
\times & \times & \cdot & \cdot \\
\times & \cdot & \times & \cdot \\
\times & \cdot & \cdot & \times
\end{bmatrix}}_{\text{centre first: } \mathbf{L}\mathbf{U} \text{ fills every } \cdot}
\qquad
\underbrace{\begin{bmatrix}
\times & \cdot & \cdot & \times \\
\cdot & \times & \cdot & \times \\
\cdot & \cdot & \times & \times \\
\times & \times & \times & \times
\end{bmatrix}}_{\text{centre last: no fill}}
```

where $\times$ marks a nonzero and $\cdot$ a zero; the left matrix has the
centre as unknown 1, the right matrix is the same matrix with the order
reversed. Eliminating the centre first uses its full row to update rows 2–4,
each of which has a nonzero in column 1, so every zero fills. Eliminating it
last, each of the other unknowns is joined only to the centre, so its step
updates only the centre's diagonal entry, which is already nonzero.

Finding the order with the least fill is NP-complete (Yannakakis), so
solvers use heuristics. Minimum degree eliminates, at each step, the unknown
with the fewest remaining neighbours, which keeps each clique small;
approximate minimum degree (AMD) does the same with cheaply estimated
degrees. Kundur and Malik call such "optimal ordering" "essential" for large
network equations (sec. 6.4.6). 01-literature §2.2 compares orderings
measured on power-flow Jacobians.

*An order is written as two permutations.* A solver records its choice as a
row order and a column order, each a **permutation matrix**: the identity
with its rows reordered, so that it has exactly one 1 in every row and
column. This report writes $\Pi_r$ for rows and $\Pi_c$ for columns (KLU's
documentation calls them $P$ and $Q$, which would clash with real and
reactive power here). Multiplying by one only reorders entries:

```math
(\Pi_r \mathbf{J})_{ij} = J_{p(i),\,j}, \qquad (\mathbf{J}\,\Pi_c)_{ij} = J_{i,\,q(j)},
\qquad\text{for example}\quad
\Pi_r = \begin{bmatrix} 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}
\ \text{moves rows } (3, 1, 2) \text{ to positions } (1, 2, 3),
```

where

- $p(i)$ is the row of $\mathbf{J}$ that becomes row $i$, and $q(j)$ the
  column of $\mathbf{J}$ that becomes column $j$;
- $J_{ij}$ is the entry of $\mathbf{J}$ in row $i$ and column $j$, and
  $(\cdot)_{ij}$ the same for a product.

$\Pi_r$ reorders the equations and $\Pi_c$ the unknowns, so $\Pi_c$ is the
elimination order. The solver factorizes the reordered matrix:

```math
\Pi_r\,\mathbf{J}\,\Pi_c = \mathbf{L}\mathbf{U} .
```

A permutation matrix is orthogonal, $\Pi^{-1} = \Pi^{\mathsf{T}}$, so
reordering does not change the solution. Inserting
$\Pi_c \Pi_c^{\mathsf{T}} = \mathbf{I}$ and multiplying on the left by
$\Pi_r$ turns the Newton system into one with the factors:

```math
\mathbf{J}\,\Delta\mathbf{x} = \mathbf{F}
\iff (\Pi_r \mathbf{J} \Pi_c)(\Pi_c^{\mathsf{T}} \Delta\mathbf{x}) = \Pi_r \mathbf{F}
\iff \mathbf{L}\mathbf{U}\,\mathbf{z} = \Pi_r \mathbf{F},
\qquad \mathbf{z} = \Pi_c^{\mathsf{T}} \Delta\mathbf{x},
```

where $\mathbf{I}$ is the identity, $\Pi^{\mathsf{T}}$ the transpose of
$\Pi$, $\mathbf{z}$ the correction with its entries in the permuted order,
and $\iff$ means that the systems have the same solutions. The system is
then solved in three steps (Kundur and Malik, eqs. 6.113–6.115):

1. **forward substitution**, $\mathbf{L}\mathbf{y} = \mathbf{c}$, from the
   first row down:
   ```math
   y_i = c_i - \sum_{j < i} L_{ij}\, y_j, \qquad i = 1, \dots, N_{\mathrm{eq}},
   ```
   where $\mathbf{c} = \Pi_r \mathbf{F}$ is the permuted right-hand side
   with entries $c_i$, $\mathbf{y}$ an intermediate vector with entries
   $y_i$, and $L_{ij}$ the entries of $\mathbf{L}$;
2. **backward substitution**, $\mathbf{U}\mathbf{z} = \mathbf{y}$, from the
   last row up:
   ```math
   z_i = \frac{1}{U_{ii}} \Bigl( y_i - \sum_{j > i} U_{ij}\, z_j \Bigr), \qquad i = N_{\mathrm{eq}}, \dots, 1,
   ```
   where $z_i$ are the entries of $\mathbf{z}$ and $U_{ij}$ those of
   $\mathbf{U}$;
3. undoing the column order: $`\Delta\mathbf{x} = \Pi_c\,\mathbf{z}`$.

Steps 1 and 2, the **triangular solves**, cost one multiply-add per stored
entry of the factors, so fill raises their cost too.

*The row order also protects accuracy.* Elimination divides by each pivot
$a_{jj}$, and a tiny pivot magnifies rounding errors. **Partial pivoting**
guards against this: when the diagonal entry is too small relative to the
rest of its column, another row is swapped into the pivot position, which
changes $\Pi_r$. A threshold rule keeps the diagonal entry as the pivot of
column $j$ when

```math
\lvert a_{jj} \rvert \ge \tau \max_{i \ge j} \lvert a_{ij} \rvert ,
```

where $a_{ij}$ are the entries of column $j$ of the partly eliminated matrix,
in the rows not yet used as pivots, and $\tau$, with $0 < \tau \le 1$, is
the pivot tolerance. A small $\tau$ keeps the sparsity-preserving order
wherever it is numerically safe and changes $\Pi_r$ only where the test
fails. The column order is thus chosen for sparsity and the row order
adjusted for stability; 01-literature §2.2 and 02-methodology §2.3 give the
settings used in practice and here.

*The order can be computed once.* Choosing $\Pi_c$ and finding where the
nonzeros of $\mathbf{L}$ and $\mathbf{U}$ will fall depends only on which
entries of $\mathbf{J}$ are nonzero, not on their values. A sparse solver
therefore splits its work into three phases:

1. **analysis** chooses the ordering and finds the structure of $\mathbf{L}$
   and $\mathbf{U}$; it depends only on the sparsity pattern;
2. **factorization** computes the values of $\mathbf{L}$ and $\mathbf{U}$;
3. **solve** runs the two triangular substitutions.

The Jacobian's pattern is the same at every Newton step, so analysis can be
done once and later steps only refactorize. **Refactorization** computes new
values while reusing $\Pi_r$, $\Pi_c$ and the pivot sequence, without
pivoting. The same reasoning extends from the steps of one power flow to
many power flows whose matrices share one pattern, which is what makes
batched factorization possible (01-literature §2.1–2.2). $\Pi_c$ stays valid
for any matrix with that pattern. $\Pi_r$ was chosen from one matrix's
values, so reusing it assumes the pivots stay large enough for new values;
this has to be checked.

*Dependencies limit the parallelism of one factorization.* Column $j$ of
the factors can be computed only after every earlier column $i$ with
$U_{ij} \neq 0$, because those columns update it. Columns that do not depend
on one another can be computed at the same time. Grouping them gives a
sequence of **levels**: the levels run one after another, and the columns
within a level run in parallel. The ACTIVSg10k Jacobian has 356 levels for
its 20,000 columns, about 56 columns per level on average and only a few in
the last levels (03-results Table 2). One factorization therefore offers
tens of independent column operations at a time, far fewer than a GPU can
run; Section 5.5 explains how batching supplies the rest.

### 5.4 An N-1 study

A study has five stages:

1. **read** the network file (here PSS/E RAW, the common exchange format for
   transmission models);
2. **solve the base case**, the intact network;
3. **build the case list**: one outage per in-service branch circuit, per
   generator and, optionally, per dc line pole;
4. **run the case loop**. For each case, the program:
   - **resets** the bus voltages to their starting values (the values in the
     network file, in the program used here);
   - **applies the outage**: sets the element out of service;
   - **updates the topology**: marks a bus left with no connection as
     isolated, moves the slack role to another generator bus if the slack
     generator was lost, and searches the network for islands. A case that
     leaves several multi-bus islands is reported as islanded and is not
     solved;
   - **updates the model**: recomputes the admittances and scheduled
     injections the outage changed;
   - **solves the power flow** by Newton's method (Section 5.2), with
     reactive-limit switching and the other enabled controls applied between
     Newton solves, and records whether it converged or diverged;
   - **checks the limits**: the slack generator's output against its maximum,
     every bus voltage against its band and every monitored branch's flow
     against its rating;
   - **writes the results**: one row per monitored branch, the case's
     outcome and any violations;
   - **restores** the network to the intact state for the next case;
5. **write the final tables**: per-case outcomes, violations, and one row per
   monitored branch per case (149 million rows for ACTIVSg10k).

Cases are independent of one another, so the case loop parallelizes across
processes. The CPU program runs as several cooperating operating-system
processes started by MPI, each called a **rank**. Each rank holds a full
copy of the network and takes the next case when it is free.

### 5.5 GPU execution

A GPU is built for throughput rather than for the speed of any one thread.
A CPU core runs a few heavyweight threads and is "designed to minimize
latency for a small number of threads at a time", whereas a GPU is designed
"to handle a large number of concurrent, lightweight threads in order to
maximize throughput" (NVIDIA, *CUDA C++ Best Practices Guide*, sec. 3.1). When
one group of threads waits for memory, the GPU runs another, at no cost,
because every active thread keeps its own registers. The consequence is a
requirement: "the software must use a large number (generally thousands or
tens of thousands) of concurrent threads" (sec. 3.2).

The guide's summary chapter reduces performance work to "three basic
strategies": "maximizing parallel execution", "optimizing memory usage to
achieve maximum memory bandwidth" and "optimizing instruction usage to
achieve maximum instruction throughput" (sec. 19.1). Its individual
recommendations appear as notes, each marked high, medium or low priority,
in the sections they concern (sec. 2.3). The three strategies explain why
one sparse factorization suits a GPU poorly and why many factorizations with
one sparsity pattern suit it well.

*Parallel execution.* GPU threads are grouped into **warps** of 32, warps
into **thread blocks**, and each block runs on one **multiprocessor**; the
GB10 GPU of the DGX Spark has 48. A **kernel** is one GPU program launched
over many blocks, every thread running the same code with its own thread
index. A multiprocessor hides memory latency by switching between its active
warps, so it needs many of them; **occupancy**, "the ratio of the number of
active warps per multiprocessor to the maximum number of possible active
warps", measures this, and low occupancy "always interferes with the ability
to hide memory latency" (sec. 11.1).

One sparse factorization does not offer that parallelism. It runs level by
level, with tens of columns per level on a 10,000-bus network and only a few
in the last levels (Section 5.3), so at any moment most of a GPU would idle.
Many factorizations of matrices with the same sparsity pattern supply the
missing parallelism if the matrices are independent: one analysis serves all
of them (Section 5.3), and every operation of the factorization then has one
independent copy per matrix. With $M$ matrices and $c$ columns in a level,
the level offers $cM$ independent column operations instead of $c$. An N-1
study has this structure, because its cases are independent outages of one
network. The batch is over cases, not over Newton iterations: the
iterations of one case depend on one another and must run in sequence.

*Memory bandwidth.* Each factorization update reads two values and writes
one for a single multiply-add, so its speed depends mainly on how fast
values move to and from the GPU's main (**global**) memory. The hardware
serves the 32 threads of a warp together: their accesses "coalesce into a
number of transactions equal to the number of 32-byte transactions
necessary to service all of the threads of the warp" (sec. 10.2.1). A warp
reading 32 adjacent 8-byte values needs eight transactions; a warp reading
32 scattered values needs 32, each moving 32 bytes to deliver 8, so only a
quarter of the traffic is useful. The guide's high-priority note is to
"ensure global memory accesses are coalesced whenever possible". The indices
of one sparse matrix give neighbouring threads unrelated positions to read.
With many matrices of one pattern, threads can instead read the same
position of consecutive matrices, which are adjacent if the values of one
position for all matrices are stored next to each other. The guide's other
memory notes are to "minimize data transfer between the host and the device"
(sec. 10.1), whose bandwidth is far lower than the GPU's own, and to prefer
on-chip **shared memory** where threads of a block reuse the same values
(sec. 12.2).

*Instruction throughput.* The 32 threads of a warp execute one instruction
at a time. When they take different branches of a conditional, "the
different execution paths must be executed separately" (sec. 13.1), so the
guide's high-priority note is to "avoid different execution paths within the
same warp". Threads assigned to the columns of one sparse matrix follow
columns of different lengths and diverge. Threads assigned one matrix each,
from a set of matrices with one pattern and one fixed sequence of
operations, run the same instructions on different numbers and do not
diverge.

These three properties are the basis of batched sparse factorization;
01-literature §2.1 reviews the published methods, and 02-methodology
§2.4–2.5 and §2.8 describe how this work applies them.

## 6. Conventions

- **Versions.** "Stock GridPACK" is GridPACK commit `b32969b0`. "Optimized
  CPU", "Alg 2" and "cuDSS" are the three ways the modified program runs
  (02-methodology §1.2).
- **Times** are wall-clock seconds of a whole study, from starting the MPI
  processes to their exit, unless a stage is named. A **speedup** is a ratio
  of median wall times on the same machine at the same process count.
- **Batch terms.** The GPU engine has $B$ **slots** ($B$ is the batch size,
  512 in the tests), each holding one case; an **engine step** advances every
  active case by one Newton step; a **chunk** is a group of cases submitted
  together, four batches' worth (02-methodology §2.4).
- **Citations** follow MLA 9: author and page or section in the text, full
  entries at the end of each part. "02-methodology §2.4" points to another
  part of this report.
- **File names and settings** are pointers for readers who want to check the
  code; the text does not depend on them. Appendix A of 02-methodology maps
  each section to its source files.
- **The private planning case** is Critical Energy/Electric Infrastructure
  Information (CEII). It is reported only in aggregate and is called
  "private case" everywhere.

## References

- Davis, T. A. *KLU User Guide*. SuiteSparse, KLU/Doc/KLU_UserGuide.
- Davis, T. A., and E. Palamadai Natarajan. "Algorithm 907: KLU, a Direct
  Sparse Solver for Circuit Simulation Problems." *ACM Transactions on
  Mathematical Software*, vol. 37, no. 3, 2010, article 36.
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
- Yannakakis, M. "Computing the Minimum Fill-In Is NP-Complete." *SIAM
  Journal on Algebraic and Discrete Methods*, vol. 2, no. 1, 1981, pp.
  77–79.
- Zhou, G., et al. "GPU-Based Batch LU-Factorization Solver for Concurrent
  Analysis of Massive Power Flows." *IEEE Transactions on Power Systems*,
  vol. 32, no. 6, 2017, pp. 4975–77.
