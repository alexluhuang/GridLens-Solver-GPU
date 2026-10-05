## 8. Cross-cutting Concepts

This chapter defines the physical, numerical, platform, and operational concepts that every building block relies on. Physics and modeling statements cite the Kundur excerpt first [K], then research papers, then other sources. Software-behavior statements cite GridPACK source first.

### 8.1 Domain Model: The AC Power-Flow Problem

#### 8.1.1 What a power flow computes

A power-flow (load-flow) analysis computes the voltages throughout a transmission network and the power flowing in each element, for specified conditions at each bus [K §6.4]. In this system, the computation is done once for the intact grid and once per contingency.

#### 8.1.2 Network representation

| Element | Representation | Basis |
|---|---|---|
| System | Balanced three-phase operation, represented by a single-phase (positive-sequence) equivalent. | [K §6.4] |
| Transmission line | Equivalent π circuit with lumped parameters: a series admittance between the two buses, plus shunt admittance to ground at each end. | [K §6.4, Representation of Network Elements] |
| Shunt capacitor or reactor | Simple admittance to ground. | [K §6.4] |
| Transformer with off-nominal turns ratio | Equivalent π circuit. | [K §6.4] |
| Phase shifts from winding connections | Usually not represented. In radial networks they do not affect the result; in meshed networks, utilities arrange windings so that no net shift arises around a loop. | [K §6.4] |
| Phase-shifting transformer (built to control flow) | Represented with its shift angle held fixed. | [K §6.4] |
| Load | Composite load at the bulk delivery point. Normally constant power; if tap-changer action is neglected, P and Q vary with voltage [K]. GridPACK's power-flow model adds the PSS/E constant-current and constant-admittance load components to its mismatch and Jacobian, and the GPU engine reproduces them (C-3, C-5). | [K §6.4, Bus Classification][GPK-PF] |

#### 8.1.3 Bus types

Each bus has four quantities: real power P, reactive power Q, voltage magnitude V, and voltage angle θ. At each bus, two are specified and two are solved for [K §6.4]:

| Bus type | Specified | Solved for | Notes |
|---|---|---|---|
| **PV** (voltage-controlled) | P, V | Q, θ | Buses with generators, synchronous condensers, or static var compensators. Reactive limits apply. |
| **PQ** (load) | P, Q | V, θ | Normally constant-power loads. |
| **Slack** (swing) | V, θ | P, Q | Needed because losses are not known in advance: at least one bus must have unspecified P and Q. It is the only bus with fully known voltage. Keeping slack Q within limits may be desirable, to avoid unrealistic solutions. |
| **Device** | Special conditions | — | Devices such as HVDC converters. Out of scope for version 1. |

#### 8.1.4 Network equations and power-flow equations

The network is described by node equations using the **node admittance matrix Y** [K §6.4.1]:

> **I = Y V**, where Iₖ is the current injected into the network at bus *k*, and Vₖ is the voltage phasor at bus *k*.

- **Y**ₖₖ (self admittance) is the sum of all admittances terminating at bus *k*.
- **Y**ₖₘ (mutual admittance) is the negative of the sum of all admittances between buses *k* and *m*.
- Constant-impedance loads are included in **Y**. Generators and other devices appear through the injected currents.

Properties of **Y** [K §6.4.1]:
- It is sparse, increasingly so as the network grows.
- It is singular if the network is floating, meaning there are no shunt branches to ground.
- It has weak diagonal dominance.
- It is symmetric if there are no phase-shifting transformers.

The equations are nonlinear because the injected currents are not known; they depend on the unknown voltages through Iₖ = (Pₖ − jQₖ)/Vₖ* [K §6.4.1.1]. Writing **Y**ₖₘ = Gₖₘ + jBₖₘ and θₖₘ = θₖ − θₘ, the power injections in polar form are [K Eq. 6.101]:

> Pₖ = Vₖ Σₘ Vₘ (Gₖₘ cos θₖₘ + Bₖₘ sin θₖₘ)
>
> Qₖ = Vₖ Σₘ Vₘ (Gₖₘ sin θₖₘ − Bₖₘ cos θₖₘ)

Each sum runs over bus *k* and its neighbors, because **Y**ₖₘ = 0 for unconnected buses. The **mismatches** are the specified values minus the computed values: ΔPₖ = Pₖ^spec − Pₖ(θ, V), and ΔQₖ = Qₖ^spec − Qₖ(θ, V) [K §6.4.3.1, §6.4.4].

#### 8.1.5 The Jacobian

Newton-Raphson relates mismatches to corrections through the Jacobian [K Eq. 6.103]:

> [ΔP; ΔQ] = **J** [Δθ; ΔV], with **J** = [∂P/∂θ, ∂P/∂V; ∂Q/∂θ, ∂Q/∂V]

**Rows and columns.** There is one P row and one θ column per non-slack bus, and one Q row and one V column per PQ bus. A PV bus has only one row and one column, because its Q is not specified and its V is fixed [K §6.4.3.1]. The dimension is therefore (n − 1) + n_PQ, which matches D'Orto's statement that the slack bus and the PV voltage magnitudes are excluded [D §II.B].

**Sparsity.** Each of the four sub-blocks has the same sparsity as **Y** [K §6.4.3.1]. This is the structural fact the whole architecture exploits.

**Entry formulas.** The entries follow by differentiating the injection equations [K Eq. 6.101]. For *m* ≠ *k*:

| Entry | Off-diagonal (m ≠ k) | Diagonal |
|---|---|---|
| ∂Pₖ/∂θₘ | VₖVₘ(Gₖₘ sin θₖₘ − Bₖₘ cos θₖₘ) [K §6.4.4, Hₖₘ] | −Qₖ − BₖₖVₖ² [K Hₖₖ][D Eq. 4] |
| ∂Pₖ/∂Vₘ | Vₖ(Gₖₘ cos θₖₘ + Bₖₘ sin θₖₘ) (derived from [K Eq. 6.101]) | Pₖ/Vₖ + GₖₖVₖ [D Eq. 5] |
| ∂Qₖ/∂θₘ | −VₖVₘ(Gₖₘ cos θₖₘ + Bₖₘ sin θₖₘ) (derived from [K Eq. 6.101]) | Pₖ − GₖₖVₖ² [D Eq. 6] |
| ∂Qₖ/∂Vₘ | Vₖ(Gₖₘ sin θₖₘ − Bₖₘ cos θₖₘ) [K §6.4.4, Lₖₘ] | Qₖ/Vₖ − BₖₖVₖ [K Lₖₖ][D Eq. 7] |

The diagonal forms reuse the just-computed Pₖ and Qₖ, as D'Orto et al. do to reduce work [D §II.B]. Implementers SHOULD derive all entries from [K Eq. 6.101] and verify them by finite differences in unit tests [P].

#### 8.1.6 Branch flows and loading

Once all bus voltages are known, the real and reactive flows in each element can be calculated [K §6.4][D §II.A]. Branch flows MUST be computed from the same π-equivalent parameters used to build **Y**, so that flows and bus injections are mutually consistent [P].

Loading is apparent power divided by the selected rating, times 100%. GridPACK computes apparent power as √(P² + Q²) for this purpose [GP].

#### 8.1.7 Contingencies

The Kundur excerpt does not treat contingency analysis, so the definitions below come from other textbook chapters, research papers, and online sources.

- **Security levels.** A system is secure (level 1) when all load is supplied, operating limits are enforced, and no limit violations occur in a contingency. Level 2 allows contingency violations that can be corrected by control action without loss of load [G]. This architecture checks level-1 security for single outages.

- **N-1 static security analysis** solves a full AC power flow for each single-element outage [Z16][Z].
- In North American planning practice, every single contingency is analyzed. Post-contingency results are checked for thermal violations (for example, loading above 100% of the emergency "rate B" in SPP criteria) and voltage violations (for example, outside 0.95–1.05 pu in SPP criteria) [SPP].
- GridPACK's contingency driver defaults to 0.9–1.1 pu voltage limits, rate A for loading with an A→B→C fallback when a rating is missing, and a severity threshold of 100% of the rating [GPK-CA].
- All limits and rating tiers MUST be configurable.

| Contingency type | Physical effect | Equation-level effect |
|---|---|---|
| **Branch outage** (line or transformer) | The element's π circuit is removed. | **Y** values at the two diagonal and two off-diagonal positions for that branch change (Section 8.3.3). |
| **Generator outage** | The unit's injection is removed. The slack absorbs the active-power imbalance. | Specified P (and, if the bus loses voltage control, the bus type) changes (Section 8.6). |

### 8.2 Numerical Solution Concept

#### 8.2.1 Newton-Raphson

Newton-Raphson solves the nonlinear equations by repeated linearization [K §6.4.3]:

1. Start from an estimate of all unknown angles and magnitudes.
2. Compute the mismatches.
3. Form the Jacobian at the current estimate.
4. Solve the linear system for the corrections.
5. Update the estimate.
6. Repeat until all mismatches are below a tolerance.

**Properties relevant to this architecture** [K §6.4.3, §6.4.5]:
- **Fast convergence:** convergence is quadratic once near the solution.
- **Repeated Jacobian work:** the Jacobian must be recomputed, and the linear system re-solved, at every iteration.
- **Scaling:** computation time grows only linearly with system size.
- **Sensitivity to the start:** convergence can fail when the initial voltages are far from the true values, and Kundur notes that the method is not suited to a "flat" start. A flat start sets all load-bus voltages to 1.0 pu at zero angle and generator buses to their specified magnitudes at zero angle; it is used when no previously solved case for similar conditions exists [K §6.4.7 footnote].

**Architectural consequence.** Every contingency case starts from the base-case solution (ADR-05). This is the "previously solved case for the same network with similar conditions" that Kundur identifies as preferable to a flat start.

#### 8.2.2 Alternatives considered

| Method | Characteristics [K §6.4.2, §6.4.4, §6.4.5] | Role in this architecture |
|---|---|---|
| **Gauss-Seidel** | Simple, reliable, tolerant of poor starting conditions, low memory. Slow convergence; time increases rapidly with size; struggles when the system is stressed by high power transfers. | Not used for large grids. |
| **Fast decoupled load flow (FDLF)** | Neglects the weak P–V and Q–θ coupling. Uses two constant real sparse matrices, B′ and B″, that are factorized once. Linear convergence, slightly more iterations, much less work per iteration, less sensitive to starting conditions. The XB variant is unsuited to high R/X systems; the BX variant suits them better. Full Newton-Raphson may be needed for very large angles or for special control devices. | Optional fast screening (B4.5) [P]. Not the primary solver, because post-contingency states can involve large angle differences (ADR-01). |
| **Full Newton-Raphson** | As in Section 8.2.1. | Primary solver for the base case and all contingencies. Used by both reference papers and by GridPACK [D][Z][GPK-PF]. |

#### 8.2.3 Sparse direct solution of the linear system

Each Newton step solves **A x = b**, with **A** = **J**. Kundur describes sparsity-oriented triangular factorization [K §6.4.6]:
- **A** is factored into lower-triangular, diagonal, and upper-triangular factors (LDU), which are themselves sparse.
- If **A** is symmetric, L is the transpose of U and need not be stored.
- **x** is found by forward substitution followed by back substitution: the last equation gives the last unknown, which is substituted into the equation above it, and so on.
- Optimal ordering is essential for large networks.

Production sparse solvers organize this into three phases [D §II.C, §IV]:

| Phase | What it does | Depends on |
|---|---|---|
| **Analysis** (symbolic) | Chooses a row and column ordering that limits fill-in (zero entries that become nonzero during elimination). Determines the nonzero structure of L and U. | Only the positions of nonzeros. |
| **Factorization** (numeric) | Computes the values of L and U, including pivoting for numerical safety on a full factorization. | Values. |
| **Solve** | Forward and backward substitution. | L, U, and the right-hand side. |

**Ordering methods evaluated by D'Orto et al.** [D §II.C]:

| Method | Approach |
|---|---|
| **AMD** (approximate minimum degree) | When a variable is eliminated, its neighbors become fully interconnected (a "clique"), and each new connection is fill-in. AMD picks, at each step, the elimination that forms the smallest possible clique. |
| **METIS** | Multilevel nested dissection: recursively finds small sets of nodes whose removal splits the graph. |
| **COLAMD** | Column ordering that does not require a symmetric pattern; suited to QR factorization. |
| **RCM** | Breadth-first ordering that minimizes the matrix bandwidth. |

**Results.** KLU with AMD produced the least fill-in at every tested size; for example, 696,576 factor entries for a 25,000-bus Jacobian with 314,922 nonzeros [D, Table 1].

**KLU** [D §II.C], designed for circuit-simulation-type matrices, works in four steps:
1. Permute the matrix to block triangular form.
2. Apply a fill-reducing ordering (here AMD) within each block.
3. Scale and factorize the diagonal blocks column by column with the Gilbert–Peierls left-looking method and partial pivoting.
4. Solve by block back substitution.

Because the Jacobian's pattern is identical at every Newton iteration, later iterations skip steps 1–2 and use a simplified step 3 without pivoting. The depth-first search that discovers each column's structure is also unnecessary, because the L and U patterns are already known [D §II.C].

**KLU in the GridPACK ecosystem.** GridPACK's linear solves go through PETSc, configured from the XML `LinearSolver` block [GPK-MATH][GPK-CA]. PETSc exposes KLU as a direct solver for sequential matrices (`-pc_type lu -pc_factor_mat_solver_type klu`). It is available when PETSc is configured with SuiteSparse, as GridPACK's Dockerfile does. Its options include the ordering (`-mat_klu_ordering`: AMD, COLAMD, or PETSc; default AMD), block-triangular preordering (`-mat_klu_use_btf`, default on), and the partial-pivoting tolerance (`-mat_klu_pivot_tol`, default 0.001) [PETSC]. The base case and the CPU path therefore get KLU + AMD with no new dependency. The batch planner (B6) calls the same SuiteSparse KLU library directly, because it needs the symbolic and numeric objects (permutations, L/U structure) that PETSc does not expose [P].

#### 8.2.4 Where the time goes

| Measurement | Source |
|---|---|
| Solving the linear equations took 81–93% of total CPU execution time across grids of 500–25,000 buses. | [D, Fig. 5] |
| An earlier study found about 85% for a 3,493-bus system. | Cited in [D §II.C] |

The architecture therefore concentrates its effort on the linear solve, and treats Jacobian assembly as secondary but still GPU-resident.

#### 8.2.5 Network reduction (not used by default)

Kron reduction eliminates passive buses (those with zero injection). Kundur cautions that reduced systems generally become denser, and recommends eliminating only nodes whose removal does not add branches [K §6.4.7]. Because density directly increases fill-in and batch memory, network reduction is NOT part of the default pipeline. It MAY be evaluated as a pre-processing option [P].

### 8.3 Superset Pattern Concept

#### 8.3.1 Why a shared pattern is possible

Three facts make one structure usable across all cases:
- Each Jacobian sub-block has the same sparsity as **Y** [K §6.4.3.1].
- Within one case, the pattern is the same at every Newton iteration [D §II.C].
- Across cases, keeping removed elements as explicit zeros gives every outage case the base case's layout [Z §III].

What remains is the effect of **bus types** (PV, PQ, slack, isolated) on the Jacobian's rows and columns. Kundur notes that a PV bus has only one row and one column [K §6.4.3.1].

#### 8.3.2 GridPACK's two formulations

| Aspect | GridPACK default build | GridPACK `LARGE_MATRIX` build (compile-time option, off by default) |
|---|---|---|
| Bus diagonal block | 2×2 for PQ buses; 1×1 for PV buses (remote-regulating PV buses keep 2×2); none for the reference bus and isolated buses | 2×2 for every non-isolated bus |
| PV bus | Q equation and V unknown removed | Q row replaced by a fixed-voltage row (1 on its own V, 0 elsewhere) |
| Reference bus | Removed | Identity 2×2 block |
| Branch block | Present only if the branch is active and neither end is the reference bus or isolated; sized by end types | Always 2×2 under the same conditions; entries zeroed where an end is PV |
| Consequence | Pattern changes with outages, bus types, slack placement, and islands; the Jacobian mapping is rebuilt inside each solve | Pattern independent of PV/PQ assignment, but still changes with outages, slack placement, and islands |

All rows in this table come from GridPACK's power-flow bus and branch components and its power-flow module [GPK-PF].

In GridPACK today, `LARGE_MATRIX` is a preprocessor switch that is commented out by default in the power-flow components, so choosing it means recompiling [GPK-PF]. Extension E6 proposes making it a run-time choice (`Powerflow/jacobianFormulation`), set per component by the factory in the same way GridPACK already sets each component's matrix mode, so no new global state is introduced ([CG I.2]; RT-1). The GPU path does not depend on E6; GridPACK's large formulation is used only as a test oracle (Section 8.10).

#### 8.3.3 The superset pattern used by the batch path

The batch path extends the `LARGE_MATRIX` idea so that **nothing** in an N-1 case changes the pattern [P]:

| Element | Superset rule |
|---|---|
| Bus diagonal | 2×2 block for every bus, including the reference bus and any bus that may become isolated. |
| Branch off-diagonal | 2×2 block for every bus pair connected in the base case, kept even when the branch is out, an end is isolated, or an end is the slack. |
| PQ bus rows | GridPACK's P and Q equations. |
| PV bus rows | GridPACK's P equation and fixed-voltage row (as in `LARGE_MATRIX`). |
| Slack bus rows | Identity rows (as in `LARGE_MATRIX`). |
| Isolated bus rows | Identity rows, with their coupling entries set to zero. This is equivalent to GridPACK removing them. |
| Branch outage | Off-diagonal values become zero, and diagonal values change by the branch's own contributions, taken from GridPACK's per-circuit accessors (`getLineElements`, `getRvrsLineElements`), so **Y** stays identical to GridPACK's [GPK-YM]. |
| Generator outage | Injection change. If the bus loses its last voltage-controlling unit, its rows switch from PV to PQ equations. If the slack is lost, the slack role moves to GridPACK's chosen bus. All of these are value changes. |
| Reactive-limit switching | PV→PQ row switch: a value change, so the GPU outer loop needs no re-planning. |

**Consequences:**
- **Coverage:** one plan serves essentially every N-1 case, including the cases v0.1 routed to the CPU.
- **Cost:** the Jacobian has 2n rows instead of (n − 1) + n_PQ, plus placeholder entries, so it has more nonzeros and possibly more fill. The size of this overhead MUST be measured on target grids against the minimal form (Section 8.14) [P].
- **Numerics:** identity rows are trivially well conditioned. The pivot order, however, is fixed from the base case, and a row that changes from fixed-voltage to full PQ form may meet a poor pivot. Health checks and the GridPACK fallback cover this (Section 8.5) [P].
- **Remote-regulating PV buses** (which GridPACK gives two equations) MUST follow GridPACK's equations exactly, or the case is routed to the CPU path [GPK-PF] + [P].

#### 8.3.4 Value deltas

B4.4 converts each classified case into value deltas, using B5's index maps:

| Delta | Content |
|---|---|
| **Y** delta | Per outaged branch, the per-end contributions to subtract. |
| Bus-type vector | PV, PQ, slack, or isolated, per bus. |
| Isolation mask | Buses outside the kept island. |
| Injection deltas | Changes in specified P and Q. |
| Slack index | The bus holding the slack role. |

These are small: tens of numbers for a branch outage, plus bus-type changes [P].

### 8.4 Batched Data Layout and Parallelism Concept

#### 8.4.1 Case-interleaved layout

Zhou et al. obtain fast GPU memory access in two ways [Z §III]:
- One team of GPU workers (a thread block) factorizes the same column for every batch member, one worker per member.
- Data for the same column of every member is stored contiguously in memory.

Neighboring workers then read neighboring addresses, and, because every member has the same structure, workers in a 32-wide group never take different branches. Every shared-pattern array (**Y** values, Jacobian values, L/U values, per-bus vectors) SHOULD therefore store the values of all members side by side for each structural position. Blocked variants MAY be used if profiling favors them [P].

#### 8.4.2 Dependency levels

Column *j* depends on every earlier column *i* with U(i,j) ≠ 0. Columns form levels: levels run in order, and columns within a level run concurrently. Batching multiplies the work available in every level by the batch size [Z §II–III]. Levels depend only on the U pattern, so they are computed once (B6.4).

#### 8.4.3 Kernel overheads and data movement

D'Orto et al. report [D §IV]:
- 128 workers per thread block performed best for Jacobian assembly.
- Overlapping the four Jacobian sub-blocks on four streams halved assembly time on smaller grids.
- Kernel-launch overhead is about 5–10 µs, which matters for tiny kernels.
- Overlapping transfers with computation requires pinned host memory.

Batching amortizes launch overhead [P]. Data movement depends on the platform profile:
- **Profiles U and C:** batch inputs and outputs MAY be shared without copies, using the allocation kinds of Section 8.8 [SPK-PG][CUDA-UM].
- **Profile D:** explicit asynchronous copies through pinned staging buffers at batch start and end [D §IV] + [P].

### 8.5 Numerical Robustness Concept

| Concern | Mechanism | Basis |
|---|---|---|
| Reused pivot order poor for a case | Refactorization requires unchanged ordering and pivoting [NV-RF]. Batched refactorization presumes stable fixed permutations [W21]. Health checks plus GridPACK fallback. | [NV-RF][W21][EX] |
| Zero or tiny pivots; non-finite values | Per-member checks after refactorization and solve. | [NV-RF][NV-DSS] + [P] |
| Inaccurate solve | Residual checks; optional iterative refinement (cuDSS). | [NV-DSS] |
| Divergence or stagnation | Iteration limit (GridPACK default 50) and divergence detection. GridPACK's own Newton loop also detects stagnation; the GPU monitor SHOULD offer the same behavior. Diverged GPU cases are re-run on the GridPACK CPU path before being reported as diverged. | [GPK-PF] + [P] |
| Superset row switches | PV↔PQ and slack moves change row values under a fixed pivot order; monitored by the same checks. | [P] |
| Reference factorization | Superset Jacobian at the base-case solution, with the base case's bus types. | [P] |
| Accuracy target | Linear solves agree with KLU to rounding level ([Z] reports differences below 10⁻¹⁴); case states agree with GridPACK's CPU solution within the tolerances of Section 8.14. | [Z] + [P] |

### 8.6 Contingency Semantics (GridPACK-Equivalent)

#### 8.6.1 Case classes

| Class | Definition | Path |
|---|---|---|
| **G — GPU-eligible** | Any branch or generator N-1 case whose effects the superset formulation expresses: outages, island isolation with the slack in the kept island, slack transfer, PV→PQ changes, reactive-limit switching. Requires that no CPU-only feature is enabled. | GPU batch |
| **C — CPU-only** | Cases in a study that enables switched shunts, transformer tap control, or area interchange; buses or devices the GPU engine does not reproduce; cases where the kept island has no slack candidate. | GridPACK CPU path, or `NO_SLACK` per GridPACK |
| **F — Flagged at run time** | GPU members that fail health checks or diverge. | GridPACK CPU path |

#### 8.6.2 Islands

When a contingency is applied, GridPACK runs a breadth-first search over active branches. It keeps the **largest island by bus count** as the main network, marks every bus in other islands as isolated (to avoid a singular Jacobian), records the island count, and separately detects lone buses [GPK-PF].

The batch path reproduces this through the isolation mask. If the kept island lacks the slack, the case follows GridPACK's outcome; in GridPACK, slack checking and transfer run before island detection [GPK-PF] + [P].

Kundur's observation that **Y** is singular for a floating network explains why isolated parts must be removed or decoupled [K §6.4.1].

#### 8.6.3 Slack transfer and capacity

- **Transfer:** if the slack bus no longer has an online generator, GridPACK moves the slack role to the bus with the largest online generating capacity, and restores the original slack when the case is removed [GPK-PF].
- **Capacity:** after solving, the contingency driver checks whether the slack's output exceeds its maximum; if so, the case is reported as `SLACK_OVERLOAD` [GPK-CA][GP].
- **Physical role:** Kundur describes the slack as the bus that absorbs the losses not known in advance, and notes that keeping its reactive output within limits may be desirable [K §6.4].

#### 8.6.4 Loss of voltage control and reactive limits

A bus is PV because it has voltage-controlling equipment [K §6.4]. Kundur's procedure for a generator whose computed Q exceeds a limit is to fix Q at the limit and treat the bus as PQ [K §6.4.2(b)].

GridPACK's settings:
- **`qlim`:** the contingency driver reads it with a code default of **true**, while the application's README table lists **false** [GPK-CA][GP]. Studies MUST therefore set `qlim` explicitly.
- **`qlimDeadband`:** default 0.1 [GPK-CA].
- **`maxQlimIterations`:** the power-flow module bounds the outer loop, default 3 [GPK-PF].

B8.7 mirrors these semantics.

#### 8.6.5 Ratings, thresholds, and monitoring

GridPACK's contingency driver provides [GPK-CA]:
- **Rating tier:** `contingencyRating` A/B/C, default A, with A→B→C fallback.
- **Severity threshold:** `violationSeverityThreshold`, default 1.0, meaning 100% of rating.
- **Voltage limits:** `minVoltage` / `maxVoltage`, defaults 0.9 / 1.1 pu.
- **Monitor filters:** areas, kV range, branch list. Filters affect reporting only.
- **Summary controls:** `topN`; performance-index weights.

The reporter reuses all of these unchanged. How GridPACK's v35/v36 parsers map PSS/E's extended rating sets onto tiers A/B/C MUST be checked on the target RAW files [P].

### 8.7 Memory and Batch-Sizing Concept

**Per-member working set:** Jacobian values, L/U values, and a few bus-length vectors. With D'Orto's 25,000-bus *minimal-form* figures (314,922 Jacobian nonzeros and 696,576 factor entries with KLU + AMD [D, Table 1]), this is about 8.1 MB per member at 8 bytes per value. The superset form is larger by an amount that MUST be measured (Section 8.3.3).

**Budget, on unified memory (Profile U)** [P]:

> Available ≈ MemAvailable − headroom − (GridPACK ranks × per-rank replica size) − superset model − backend workspace
>
> Batch size ≈ min(saturation size, Available ÷ (per-member working set × batches in flight), validated backend cap)

**Inputs to the budget:**
- **Reclaimable memory:** `cudaMemGetInfo` does not count memory the OS could reclaim, so the budget starts from OS-reported available memory [SPK-PG].
- **Headroom:** protects against the host-stall behavior reported under memory pressure [OGKM].
- **Saturation size:** found by sweeping. Zhou et al. saw per-system time flatten near batch 128 on a 9,241-bus grid (K40) [Z].
- **Validated cap:** reflects backend validation results (Risk R-2).

On Profiles C and D, the GPU-only working set is budgeted against GPU memory instead, and host memory budgets the GridPACK replicas [P].

### 8.8 Memory Placement Policy (Platform Abstraction)

#### 8.8.1 Capability probe

At start-up, B9.1 MUST query and log the following [CUDA-UM][SPK-PG][TEGRA]:

| Attribute | What it indicates |
|---|---|
| Compute capability | Which compiled code path to use. |
| Integrated GPU | Whether CPU and GPU share one physical memory. |
| Concurrent managed access | Full vs. limited unified-memory support. |
| Pageable memory access | Whether the GPU can use ordinary system allocations. |
| Pageable access uses host page tables | Hardware (ATS) vs. software (HMM) coherence. |
| Host-native atomics | Whether GPU atomics on host memory are hardware-supported. |
| GPUDirect RDMA / dma-buf support | Must be queried; unsupported on DGX Spark. |

#### 8.8.2 Buffer classes and placement

| Buffer class | Examples | Profile U (integrated) | Profile C (coherent C2C) | Profile D (discrete) |
|---|---|---|---|---|
| **M** — model, read-only after setup | Superset pattern, index maps, plan | Device allocation | GPU memory | GPU memory |
| **W** — per-batch working | **Y**/Jacobian/L/U values, Newton vectors | Device allocation. On DGX Spark it is not CPU-coherent, which is acceptable because only the GPU touches it [SPK-PG]. | GPU memory | GPU memory |
| **X** — exchange | Deltas, warm starts, masks in; states, statuses out | Pinned or registered host memory. Pageable memory where pageable access is supported (Thor onward on Tegra) [TEGRA]. | System allocation (ATS); large pages for large regions [CUDA-UM] | Pinned staging + async copies |
| **F** — factors for a host-side solve (only if ADR-04's alternative is chosen) | L/U values | Host-coherent allocation | System allocation | Copy to host |

**Tegra-specific guidance** [TEGRA]:
- Device memory is recommended for buffers only the integrated GPU uses.
- On I/O-coherent Tegra devices (compute capability 7.2 and up), pinned memory is cached on the CPU.
- Thor adds full system-memory coherence, with registered host memory cached in the GPU's L2.

Managed memory MAY be used where full unified-memory support exists, but explicit placement is the default, for predictability [P].

### 8.9 CPU Topology and Placement Concept

| Topic | Rule | Basis |
|---|---|---|
| Heterogeneous cores | DGX Spark mixes Cortex-X925 performance cores (2 MB L2) and Cortex-A725 efficiency cores (512 KB L2) in two clusters with different L3 sizes. Other ARM systems may be homogeneous. Placement MUST use run-time discovery. | [SPK-PG][SR] |
| Rank and thread binding | GridPACK ranks and accelerator threads are bound by role (Section 7.1.2) through the MPI launcher's binding options or an equivalent mechanism. | [P] |
| Bandwidth sharing | On unified memory, CPU-path and reporter ranks compete with GPU batches for the same bandwidth. B12 MUST expose throughput, so the number of active CPU ranks can be tuned. | [SPK-PG] + [P] |
| Memory model | Code shared between threads MUST use standard atomics and fences, because Arm's memory model is more relaxed than x86's. | [SPK-PG] |

### 8.10 GridPACK Parity Concept

- **Equation source:** GridPACK's power-flow bus and branch components define mismatches and Jacobian blocks; the `LARGE_MATRIX` code paths define the superset rows [GPK-PF].
- **Kernel tests:** for randomized states and cases, B8.3/B8.4 outputs are compared with values produced by GridPACK components in their large formulation: selected at run time once extension E6 is available, or through a separately compiled test build until then [P]. The element formulas are host-and-device functions, so the same code is also tested on the CPU [CUDA-BP §7.1.2].
- **Case tests:** final states and statuses are compared with the stock GridPACK contingency-analysis application run on the same configuration, with the same `qlim` and tolerance settings (FR-11) [GPK-CA] + [P].
- **Known intentional differences** are recorded: the warm-start source (ADR-05) and the fixed-pivot refactorization [P].

### 8.11 Status Taxonomy

| Status | Meaning | GridPACK equivalent [GPK-CA] |
|---|---|---|
| `OK` | Converged | OK |
| `ISLANDED` | Converged on the kept island; other buses isolated | ISLANDED |
| `NO_SLACK` | No usable slack in the kept island | NO_SLACK |
| `SLACK_OVERLOAD` | Converged, but the slack exceeds its maximum | SLACK_OVERLOAD |
| `DIVERGED` | Not converged on either path | DIVERGED |
| `NUMERICAL_FAILURE` | Failed health checks on both paths | — |
| `MISSING` | No outcome (a defect), detected during reconciliation | — |

Every outcome also records its **path** (GPU batch or GridPACK CPU), iterations, final mismatch, final PV/PQ count, and health events. Violations are reported separately by GridPACK's checks.

### 8.12 Observability Concept

| Metric | Source |
|---|---|
| CPU phase timings | GridPACK timer module [GPK] |
| GPU phase timings (mismatch, Jacobian, refactorization, solve, control loop, exchange) | Accelerator, via GPU events and profiler ranges [NV-DSS] + [P] |
| Batch occupancy over time; effective memory bandwidth per kernel, computed as bytes read plus written divided by time (the CUDA guide's primary metric, and the saturation indicator used by [Z]) | Accelerator [CUDA-BP §9.2.2][Z] + [P] |
| Path mix and fallback causes; iteration histograms | B7, B8 [P] |
| Memory-pressure indicators (OS available memory, swap activity) on unified memory | B9 [SPK-PG] + [P] |

NVIDIA recommends Nsight tools and `perf` for profiling on DGX Spark [SPK-PG].

### 8.13 Configuration Concept

The configuration stays in GridPACK's XML format, and `configuration.xml` remains the only file the user writes besides the RAW case. Existing blocks and keys keep their meaning [GPK-CA]. Two optional blocks are added inside `Contingency_analysis`: `GPUBatch` for accelerator behavior, and `Execution` for process-level behavior [P]. Every new key is a run-time setting (Section 8.16); the complete list, with defaults and permitted values, is in Appendix G. Illustrative only:

```xml
<Configuration>
  <Contingency_analysis>
    <FullBranchN1>true</FullBranchN1>
    <FullGeneratorN1>true</FullGeneratorN1>
    <qlim>true</qlim>                          <!-- set explicitly (Section 8.6.4) -->
    <minVoltage>0.9</minVoltage>
    <maxVoltage>1.1</maxVoltage>
    <contingencyRating>A</contingencyRating>
    <outputFormat>csv_delta</outputFormat>
    <GPUBatch>                                 <!-- new, optional; absent = stock GridPACK -->
      <enabled>auto</enabled>                  <!-- auto | on | off -->
      <onUnavailable>fallback</onUnavailable>  <!-- fallback | error -->
      <backend>auto</backend>                  <!-- auto | cudss | alg2 | cpu_reference -->
      <batchSize>auto</batchSize>
      <maxValidatedBatch>128</maxValidatedBatch>
      <threadsPerBlock>auto</threadsPerBlock>  <!-- auto, or a multiple of 32 -->
      <memoryProfile>auto</memoryProfile>      <!-- auto | unified | coherent | discrete -->
      <memoryHeadroomGB>16</memoryHeadroomGB>
      <solvePlacement>gpu</solvePlacement>     <!-- gpu | host -->
      <warmStart>base_case</warmStart>         <!-- base_case | raw -->
      <shadowFraction>0.0</shadowFraction>     <!-- share of GPU cases re-solved on the CPU -->
      <telemetry>summary</telemetry>           <!-- off | summary | detailed -->
    </GPUBatch>
    <Execution>                                <!-- new, optional -->
      <acceleratorRanks>auto</acceleratorRanks>
      <cpuBinding>auto</cpuBinding>
      <logLevel>info</logLevel>
    </Execution>
  </Contingency_analysis>
  <Powerflow>
    <networkConfiguration>network.raw</networkConfiguration>  <!-- version auto-detected -->
    <tolerance>1.0e-6</tolerance>
    <maxIteration>50</maxIteration>
    <LinearSolver>
      <PETScOptions>-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type klu</PETScOptions>
    </LinearSolver>
  </Powerflow>
</Configuration>
```

| Key | Default | Basis |
|---|---|---|
| `networkConfiguration` (auto-detect) or `networkConfiguration_v33` … `_v36` | — | [GPK-PF] |
| `tolerance`, `maxIteration` | 10⁻⁶, 50 | [GPK-PF] |
| `qlim`, `qlimDeadband`, `maxQlimIterations` | code default true (set explicitly), 0.1, 3 | [GPK-CA][GPK-PF] |
| `FullBranchN1`, `FullGeneratorN1`, `contingencyList` | — | [GPK-CA] |
| `minVoltage`, `maxVoltage`, `contingencyRating`, `violationSeverityThreshold`, monitor filters, `outputFormat`, `outputFile` | 0.9, 1.1, A, 1.0, none, text, `ca_results` | [GPK-CA] |
| `LinearSolver/PETScOptions` | GridPACK's example uses SuperLU_DIST; this design uses KLU | [GPK-CA][PETSC] |
| `GPUBatch/*`, `Execution/*` | Appendix G | [P] |

The example's `memoryHeadroomGB` of 16 and `maxValidatedBatch` of 128 are placeholders, to be replaced by measured values [P].

### 8.14 Verification and Validation Concept

| Level | Check | Reference |
|---|---|---|
| Parser | RAW files of each supported version produce identical GridPACK networks with and without the accelerator build. | [GPK-PF] |
| Kernel parity | Mismatch and Jacobian values vs. GridPACK components (`LARGE_MATRIX` test build). | [GPK-PF] + [P] |
| Linear algebra | Backend refactorization and solve vs. KLU, near rounding level. | [Z] |
| Case parity | Every GPU-path case vs. stock GridPACK on the CPU: voltages and angles within tolerance (proposed 10⁻⁶ pu and 10⁻⁶ rad); identical status; identical final PV/PQ sets, or differences explained. | [GPK-CA] + [P] |
| Output parity | GridPACK output files (csv_delta, convergence, summary) are equal after sorting, within numeric tolerance. | [GPK-CA] + [P] |
| Superset overhead | Nonzeros and fill of the superset vs. the minimal form, per grid. | [P] |
| Platform matrix | At least the DGX Spark plus one other Arm + NVIDIA system from a different profile (C or D) when available; GPU-less build. | [P] |
| Performance | Throughput vs. the stock GridPACK contingency-analysis application on the same machine and grid. | [GPK-CA] + [P] |
| Element functions | Host-and-device element functions give the same results on CPU and GPU within tolerance; differences are expected from fused multiply-add and reordered summation, so comparisons are never bitwise. | [CUDA-BP §7.1.2, §7.3] |
| Configurability | Each run-time setting in Appendix G changes behavior without a rebuild; absent blocks reproduce stock GridPACK; missing GPU or plugins fall back. | FR-14; CONF-1 to CONF-4 |
| Standards | Static analysis, CUDA review checklist, Docker build checks, and CMake review per Appendix F. | Section 2.5; STD-1 to STD-4 |

### 8.15 Performance Model

**Stock GridPACK baseline** [GPK-CA] + [P]. Each rank solves whole cases sequentially:

> T_GridPACK ≈ (N_cases ÷ N_ranks) × T_case,CPU

where T_case,CPU is about the number of iterations × (Jacobian assembly + KLU refactorization + solve) on one core, and N_ranks is limited by cores and memory bandwidth.

**Composite pipeline.** This extends the timing model of D'Orto et al. [D §IV]; see Appendix A.7:

> T_study ≈ T_parse + T_base + T_classify + T_export + T_plan + T_setup + Σ_batches T_batch + T_fallback + T_report
>
> T_batch ≈ T_exchange,in + Σ_{outer} Σ_{k} (T_mis + T_jac + T_refact + T_solve + T_upd + T_check) + T_exchange,out

On Profile U, the exchange terms approach zero with zero-copy buffers [SPK-PG] + [P]. T_fallback and T_report run on CPU ranks while GPU batches run.

**Bandwidth bound on DGX Spark** [P]. Batched sparse refactorization is memory-bound; Zhou et al. reached about 70% of peak bandwidth [Z]. A lower bound per member-iteration is:

> t ≥ (bytes read and written per member-iteration) ÷ (effective bandwidth)

Illustrative, using D'Orto's minimal-form 25,000-bus counts [D, Table 1]:

| Step | Approximate traffic per member-iteration |
|---|---|
| Refactorization: read about 2.5 MB of Jacobian values; read and write about 5.6 MB of factors | ≈ 14 MB |
| Solve: read the factors | ≈ 6 MB |
| Jacobian and mismatch evaluation | ≈ 3–4 MB |
| **Total** | **≈ 23 MB** |

At 200–273 GB/s (measured to peak [CT][SPK-HW]), this gives roughly 85–115 µs per member-iteration, or about 0.4–0.6 ms per case at 5 iterations. That is a ceiling of very roughly 1,700–2,400 cases per second, before superset overhead, CPU bandwidth sharing, and imperfect efficiency. These figures are for capacity planning only and are not predictions. Measured values (Chapter 10) replace them.

### 8.16 Runtime Configuration Concept

This section is the decision register for the runtime-first principle (Section 2.6). Every row is a run-time setting, except the build-time residuals listed at the end.

**Resolution (B14):** for each setting, the effective value comes from `configuration.xml` if present, else from an environment variable where one is defined, else from automatic detection, else from the documented default (RT-2). The effective value and its source are logged (RT-3). Invalid values stop the run at start-up (RT-5).

| # | Decision | Run-time control | Default | Basis |
|---|---|---|---|---|
| 1 | Use the accelerator at all | `GPUBatch/enabled` (`auto`/`on`/`off`), plus detection of a usable GPU | `off` when the block is absent (stock GridPACK); `auto` when present | [CUDA-BP §17.1]; RT-4 |
| 2 | Reaction when the accelerator is unavailable | `GPUBatch/onUnavailable` (`fallback`/`error`) | `fallback` | [P] |
| 3 | Plugin location | `GPUBatch/pluginPath`; environment variable; path relative to the installed `ca.x` | Relative path | [P] |
| 4 | Batch backend | `GPUBatch/backend` (`auto`/`cudss`/`alg2`/`cpu_reference`) | `auto`: cuDSS if its plugin loads and the batch size is within its validated cap, otherwise custom Algorithm 2 | [NV-DSS][Z] |
| 5 | GPU device | `CUDA_VISIBLE_DEVICES`; `GPUBatch/device` | First visible device | [CUDA-BP §18.5] |
| 6 | Which ranks drive GPUs | `Execution/acceleratorRanks` (`auto` or a list) | One rank per visible GPU | [P] |
| 7 | CPU placement of ranks and threads | `Execution/cpuBinding` (`auto`/`none`/`performance_first`), together with the MPI launcher's own binding options | `auto` (topology-aware, Section 8.9) | [SPK-PG] + [P] |
| 8 | Memory paradigm and buffer placement | `GPUBatch/memoryProfile` (`auto`/`unified`/`coherent`/`discrete`) | `auto` (probe, Section 8.8) | [CUDA-UM][TEGRA] |
| 9 | Memory budget | `GPUBatch/memoryHeadroomGB`, `GPUBatch/maxMemoryGB` | Headroom from validation; no cap | [SPK-PG] |
| 10 | Batch size | `GPUBatch/batchSize` (`auto` or integer); `GPUBatch/maxValidatedBatch` | `auto` (saturation sweep within budget) | [Z] |
| 11 | Kernel launch configuration | `GPUBatch/threadsPerBlock` (`auto` or a multiple of 32) | `auto` (occupancy-based choice) | [CUDA-BP §11.1, §11.3] |
| 12 | Jacobian formulation on the GPU | `GPUBatch/formulation` (`superset`; `grouped` reserved for sub-batches by bus-type signature) | `superset` | ADR-08 |
| 13 | Planner ordering | `GPUBatch/plannerOrdering` (`amd`/`colamd`) | `amd` | [D][PETSC] |
| 14 | Reference-factorization pivot tolerance | `GPUBatch/pivotTolerance` | KLU's default, which PETSc documents as 0.001 | [PETSC] |
| 15 | Solve placement | `GPUBatch/solvePlacement` (`gpu`/`host`) | `gpu` | ADR-04 |
| 16 | Warm start | `GPUBatch/warmStart` (`base_case`/`raw`) | `base_case` | ADR-05 |
| 17 | Health thresholds | `GPUBatch/health/*` (residual limit, pivot magnitude, non-finite check) | From validation | [P] |
| 18 | Iterative refinement | `GPUBatch/refinementSteps` | 0 | [NV-DSS] |
| 19 | Newton tolerance, iteration limit, reactive limits, controls | Existing GridPACK keys | GridPACK defaults | [GPK-PF][GPK-CA] |
| 20 | Shadow validation | `GPUBatch/shadowFraction` (share of GPU-solved cases re-solved by GridPACK and compared) | 0 in production; 1 in validation runs | [CUDA-BP §7.1.1] |
| 21 | Telemetry and profiler ranges | `GPUBatch/telemetry` (`off`/`summary`/`detailed`) | `summary` | [CUDA-BP §4.1][SPK-PG] |
| 22 | Output content and format | Existing GridPACK keys | GridPACK defaults | [GPK-CA] |
| 23 | GridPACK's own Jacobian formulation, for oracle tests | `Powerflow/jacobianFormulation` (`standard`/`large`), proposed extension E6 replacing the compile-time `LARGE_MATRIX` switch | `standard` | [GPK-PF] + [P] |
| 24 | JIT cache location and size | CUDA's cache environment variables | Driver defaults | [CUDA-BP §18.4] |
| 25 | GPU code variant | Automatic: the driver selects an embedded native binary, or JIT-compiles PTX | — | [CUDA-BP §17.3] |
| 26 | Log verbosity | `Execution/logLevel` | `info` | [P] |

**Build-time residuals and why they cannot be run-time settings:**

| Residual | Why it is build-time | Mitigation |
|---|---|---|
| Whether plugins exist | Compiling CUDA code needs a CUDA compiler at build time. | `ca.x` is identical either way; the plugins are optional files. |
| Native GPU architectures | Native GPU code is produced by the compiler. | `all` real architectures plus PTX; the driver chooses or JIT-compiles at run time [CUDA-BP §17.3][CMAKE]. |
| CUDA toolkit and cuDSS versions in an image | Libraries are installed into the image. | CUDA's minor-version compatibility [CUDA-BP §16.4]; build arguments select versions. |
| Jetson Orin profile | Separate CUDA packaging [CUDA-RN]. | Same source tree and settings. |
| Language standard, build type | Properties of compilation. | One documented choice per image. |
| Double precision | Fidelity requirement (T-5), not a policy choice. | — |
| Compiled kernel variants (for example tile sizes) | Templates or specializations are compiled. | A fixed variant set, selected at run time (RT-6). |

### 8.17 C++ Coding Standard Concept (S-1)

All new and modified C++ follows the *C++ Core Guidelines* [CG]. The rules below are the ones that most shape this design; the full mapping is in Appendix F.

| Area | Application in this system | Rules |
|---|---|---|
| Language level and support library | ISO C++17 for all new code, matching GridPACK, which sets no standard and builds with the compiler default. The Guidelines Support Library supplies `not_null`, `span`, `Expects`/`Ensures`, and `narrow`. | [CG P.2][CG P.13][CG GSL.view][CG GSL.assert] |
| Resource management | Every CUDA stream, event, and allocation; every cuDSS and KLU object; every dynamically loaded library handle; and every MPI-adjacent resource is owned by an RAII handle. No naked `new`/`delete` or `malloc`/`free`. Ownership is expressed with `unique_ptr` (custom deleters for C APIs), never with raw pointers. | [CG R.1][CG R.3][CG R.10][CG R.11][CG R.20][CG R.21][CG E.6] |
| Classes | Rule of zero where possible; otherwise define or delete all five special members. Backend and plugin adapters are pure abstract interfaces with protected or virtual destructors, suppressed copying, and `override` on every overrider. | [CG C.20][CG C.21][CG C.35][CG C.67][CG C.121][CG C.128] |
| Interfaces | Strongly typed indices (bus, branch, case, batch member) instead of bare integers. Preconditions stated with `Expects`. Sequences passed as spans (pointer plus size), never as a single pointer. No ownership transfer by raw pointer or reference. | [CG I.4][CG I.6][CG I.11][CG I.13][CG F.24] |
| Functions | Small parameter lists; "in" parameters by value or reference to const; results returned rather than written through output parameters; `noexcept` where a function cannot throw. | [CG I.23][CG F.16][CG F.20][CG F.21][CG F.6] |
| Error handling | The strategy is fixed early: host code throws purpose-designed exception types when it cannot perform its task. Non-convergence or a flagged batch member is an expected **outcome**, returned as a status value, not an exception. Plugin boundaries and device code use status codes systematically. | [CG E.1][CG E.2][CG E.3][CG E.14][CG E.27][CG I.10] |
| Arithmetic and indexing | Signed types for arithmetic and subscripts; no lossy conversions; named casts only where a cast is unavoidable. | [CG ES.102][CG ES.106][CG ES.107][CG ES.46][CG ES.48][CG ES.49] |
| Constants and macros | Immutable by default; `constexpr` for compile-time values; enumerations (`enum class`) instead of macros. | [CG Con.1][CG Con.4][CG Con.5][CG Enum.1][CG Enum.3][CG ES.31] |
| Global state | No new non-const globals or singletons. Accelerator state lives in session objects owned by `ca.x`'s driver. | [CG I.2][CG I.3] |
| Concurrency | Code assumes it runs multithreaded; no data races; minimal shared writable data; RAII locks only; tasks rather than raw threads; no lock-free code unless measured necessary. Standard atomics with explicit memory ordering suit Arm's relaxed memory model [SPK-PG]. | [CG CP.1][CG CP.2][CG CP.3][CG CP.4][CG CP.20][CG CP.100] |
| Performance | No optimization without measurement; design that permits optimization; predictable memory access (the case-interleaved layout). | [CG Per.1][CG Per.6][CG Per.7][CG Per.19] |
| Source files | `#include` guards; no `using namespace` at global scope in headers; quoted includes for local files. GridPACK's `.cpp`/`.hpp` convention is kept. | [CG SF.1][CG SF.7][CG SF.8][CG SF.12] |
| Tools | clang-tidy's `cppcoreguidelines-*` checks on all new C++. | [CG P.12][CG App. D] |

**Where GridPACK's interfaces force deviations.** New code must call GridPACK APIs that return shared pointers and raw component pointers obtained from them, must downcast generic components to power-flow component types (done with `dynamic_cast`, as [CG C.146]–[CG C.148] direct, though [CG C.153] prefers virtual functions), and must use GridPACK's existing static configuration flags. These uses are confined to one adapter layer between GridPACK and the accelerator interfaces ([CG I.30]: encapsulate rule violations) and recorded in Appendix F. The rest of the new code sees only the guideline-conforming types of that layer.

### 8.18 CUDA Coding Standard Concept (S-2)

All CUDA code follows the *CUDA C++ Best Practices Guide* [CUDA-BP], applied by its priorities (Section 2.5.2). The work follows the guide's Assess, Parallelize, Optimize, Deploy cycle [CUDA-BP §2.2]: profiling and benchmark gates (Chapter 10) precede each optimization.

| Priority | Recommendation | Application in this system |
|---|---|---|
| High | Profile to find hotspots and bottlenecks [CUDA-BP §4.1] | Phase timers and Nsight profiling before any optimization (Section 8.12). |
| High | Use effective bandwidth as the performance metric [CUDA-BP §9.2] | Bytes read plus written per kernel, divided by time, is a first-class telemetry metric; the batch kernels are bandwidth-bound (Section 8.15). |
| High | Minimize host–device data transfer [CUDA-BP §10.1] | Exchange only at batch boundaries; zero-copy exchange buffers on integrated GPUs (Section 8.8). |
| High | Coalesce global memory accesses [CUDA-BP §10.2.1] | Case-interleaved layout, one worker per batch member (Section 8.4.1). |
| High | Minimize global memory use; prefer shared memory where possible [CUDA-BP §12.2] | Per-row bus data reused by many nonzeros is staged in shared memory where profiling shows a benefit [P]. |
| High | Avoid different execution paths within a warp [CUDA-BP §13.1] | The shared pattern keeps members' control flow identical. Superset rows differ between members only at buses their own contingency affects; such rows are written branch-free or predicated where profiling shows divergence [P]. |
| Medium | Use shared memory to avoid redundant global loads [CUDA-BP §10.2.3]; keep enough occupancy [CUDA-BP §11.1] | Applied after profiling. |
| Medium | Threads per block a multiple of 32 [CUDA-BP §11.3] | Enforced when `threadsPerBlock` is set manually; the automatic choice uses occupancy information. |
| Medium | Signed integers as loop counters [CUDA-BP §12.1.5] | Matches the guidelines' signed-index rules [CG ES.102][CG ES.107]. |
| Medium | Prefer faster, specialized math functions [CUDA-BP §12.1] | Used only where precision is unchanged (for example, computing sine and cosine of the same angle together). |
| Medium | Fast-math library "whenever speed trumps precision" [CUDA-BP §12.1] | **Not applied.** Precision outranks speed here (fidelity, T-5). Fast-math and reduced-precision compiler switches (`-use_fast_math`, `-ftz=true`, `-prec-div=false`, `-prec-sqrt=false`) are prohibited [CUDA-BP §20.1]. |
| Low | Zero-copy on integrated GPUs [CUDA-BP §10.1.3] | Profile U exchange buffers. |
| Low | Avoid automatic double-to-float conversion [CUDA-BP §12.1] | All arithmetic is double precision. |
| Low | Branch predication instead of branches where possible [CUDA-BP §13.2] | Per-row type selection in superset rows. |

**Practices without a priority label, all applied:**

| Practice | Application |
|---|---|
| Reference comparison and unit testing [CUDA-BP §7.1] | Element formulas (mismatch terms, Jacobian blocks) are written once as host-and-device functions. They are unit-tested on both the CPU and the GPU, as the guide suggests, and shared by the CPU reference backend, which avoids duplicate code. |
| Numerical accuracy [CUDA-BP §7.3] | Results are compared within tolerances, not bitwise. The guide notes that floating-point addition is not associative under parallel reordering and that fused multiply-add changes results slightly. |
| Stream-ordered pool allocation [CUDA-BP §10.3] | Batch buffers come from stream-ordered pools rather than repeated allocate/free calls. |
| NUMA tuning [CUDA-BP §10.4] | Where a platform has several NUMA nodes, memory binding is set through the launcher or `Execution/cpuBinding`, and automatic NUMA balancing SHOULD be disabled. |
| Device availability [CUDA-BP §17.1] | `cudaGetDeviceCount` first; any failure leads to the CPU path. |
| Error handling [CUDA-BP §17.2] | Every CUDA and CUDA-library call's status is checked, and `cudaGetLastError` is checked immediately after each kernel launch, because errors may otherwise surface only at a later synchronization. |
| Building for compatibility [CUDA-BP §17.3] | All real architectures plus PTX (Section 7.3.2). |
| Runtime distribution [CUDA-BP §17.4] | Static CUDA runtime; dynamic-only libraries (cuDSS, CUDA math libraries) shipped in the image. |
| Minor-version compatibility [CUDA-BP §16.4.1.4, §16.4.1.5] | Features newer than the minimum driver are queried and used conditionally; plugin interface records carry their sizes (Section 5.3.6). |
| Register and resource use [CUDA-BP §20.1] | Per-kernel register and memory reports in CI; register limits or launch bounds only when profiling shows register pressure. |

### 8.19 Build-System Concept (S-4)

All new CMake follows *Effective Modern CMake* [EMC].

**Structure.** A new directory in GridPACK's module tree, for example `src/applications/modules/batch_pf/` [P]:

| Target (namespaced alias) | Kind | Contents | Links |
|---|---|---|---|
| `gridpack::batchpf_interface` | `INTERFACE` | C-style plugin interface headers (I-10, I-11) | — |
| `gridpack::batchpf_host` | `STATIC` | B13, B14, B9-H, and the GridPACK adapter layer (B1, B5, B7, B11 extensions) | `PUBLIC` interface target; `PRIVATE` GridPACK libraries |
| `gridpack_batchpf_core` | `MODULE` | B6, B8, B9-D, custom Algorithm 2 and CPU reference backends | `PRIVATE` interface target, the static CUDA runtime from `FindCUDAToolkit`, KLU |
| `gridpack_batchpf_cudss` | `MODULE` | cuDSS backend | `PRIVATE` interface target, the `cudss` package target |
| Test executables | Executables | Unit, parity, platform, and performance tests | `PRIVATE` |

| Rule from [EMC] | How it is met |
|---|---|
| Think in targets and properties; keep internal properties `PRIVATE`; always state `PUBLIC`/`PRIVATE`/`INTERFACE` | Every `target_*` call states its scope. Only the interface target publishes include directories. |
| Forget `include_directories`, `add_compile_options`, `link_libraries`, and `add_definitions`; hands off `CMAKE_CXX_FLAGS` | None are used in new code. The language level is set with `target_compile_features` (`cxx_std_17`, `cuda_std_17`). |
| Define project properties globally; do not set ABI-affecting options per target | Warning settings live in one internal `INTERFACE` target linked `PRIVATE` by every new target. The language standard is identical across all new targets. |
| Use modern find modules and exported targets of external packages | CUDA through `FindCUDAToolkit` (NVIDIA notes that the older `FindCUDA` module is deprecated [CUDA-IG]); cuDSS through its CMake package, which exports a `cudss` target [CRS]. Where a dependency exports no targets (possibly SuiteSparse as installed by PETSc), a small find module that exports an imported target is written, as [EMC] recommends. |
| No `file(GLOB)`; avoid custom variables in target definitions | Sources are listed explicitly in each `add_library` call. |
| A library in the same tree looks like an external one | Namespaced `ALIAS` targets; exported interface target with `BUILD_INTERFACE`/`INSTALL_INTERFACE` filtering. |
| Every header has a source file that includes it first | Applied to `.hpp`/`.cpp` and `.cuh`/`.cu` pairs. |
| Use more than one analyzer, through `<LANG>_CLANG_TIDY` and related properties; put CI settings in CTest scripts | The CI's CTest script sets clang-tidy (with `cppcoreguidelines-*`) and include-what-you-use. The project itself does not hard-code analyzer settings. |
| Never pass `-Werror`; treat new warnings as errors | CI compares warning counts against the previous build and rejects increases. |
| Naming convention for tests | `batchpf.unit.*`, `batchpf.parity.*`, `batchpf.platform.*`, `batchpf.perf.*`. End-to-end runs of `ca.x` reuse GridPACK's existing `gridpack_add_run_test` helper [GPK-CA]. |
| Toolchain files for cross-compiling, one simple file per platform | Used only if Jetson images are cross-built. |

**Optional CUDA at configure time.** The directory includes CMake's `CheckLanguage` module and calls `check_language(CUDA)`. Only if a CUDA compiler is found, and the cache option `GRIDPACK_ENABLE_GPU_BATCH` is `AUTO` or `ON`, does it call `enable_language(CUDA)` and add the plugin targets [CMAKE]. `ON` without a CUDA compiler stops configuration with a clear message. `CUDA_ARCHITECTURES` defaults to `all` unless `CMAKE_CUDA_ARCHITECTURES` or the `CUDAARCHS` environment variable supplies a value [CMAKE]. Plugins hide all symbols except their entry points [CUDA-BP §16.4.1.4] + [P].

**Existing GridPACK CMake (exception S-4).** GridPACK's top-level file uses `include_directories` and `add_definitions`, and the contingency-analysis directory links dependencies through variables such as `${Boost_LIBRARIES}` with a plain-signature `target_link_libraries` call [GPK-CA]. These existing lines are left unchanged and recorded in Appendix F. One existing line is changed: because CMake forbids mixing plain and keyword `target_link_libraries` signatures on one target (policy CMP0023), the `ca.x` link line gains the `PRIVATE` keyword so the new host library can be linked with an explicit scope. For an executable this does not change behavior [CMAKE] + [P].

### 8.20 Container Concept (S-3)

| Principle | Application | Basis |
|---|---|---|
| One Dockerfile | GridPACK's Dockerfile builds both the stock and the GPU image; default build arguments reproduce today's image. | [GPK]; T-10 |
| Pinned, verified inputs | Base images pinned to full version tags (optionally digests); remote downloads verified by checksum; build arguments, not edits, select versions. | [DF][NV-IMG] |
| No secrets in build arguments | Build arguments appear in image history; secrets, if ever needed, use secret mounts. | [DF] |
| Cache-friendly package installation | `apt-get` steps for new packages use cache mounts with `sharing=locked`. | [DF] |
| Multi-architecture | One build serves arm64 and amd64 through `--platform` and the automatic `TARGETARCH` arguments, as GridPACK's images already do. | [DF][GPK] |
| No GPU needed to build | Compiling CUDA code needs no GPU. GPU tests run when containers are run, not during `docker build`; `RUN --device` exists but needs a special builder entitlement and is not used. | [DF] + [P] |
| User command unchanged | `ca.x` on the path; explicit `CMD ["/bin/bash"]`; no `ENTRYPOINT`, so `docker run … bash` and `docker run … ca.x configuration.xml` both work. | [DF][GPK]; FR-12 |
| Existing conflicts left in place | Recorded in Appendix F (for example, the persistent `DEBIAN_FRONTEND` setting). | S-3 |

### 8.21 Standards Exception Process

| Step | Rule |
|---|---|
| 1. Propose | The author shows that the functionality cannot be implemented compliantly (S-1 to S-4), or, for Dockerfile and CMake only, that the conflicting line already existed. |
| 2. Contain | The deviation is confined to the smallest unit: one adapter class, one function, or one Dockerfile or CMake line ([CG I.30], [CG P.11]). |
| 3. Record | An entry is added to the Appendix F register: ID, standard and rule, location, required functionality, why no compliant alternative exists, containment, reviewer, revisit trigger. |
| 4. Review | Merge requires reviewer approval of the register entry. Static-analysis suppressions refer to the entry ID. |
| 5. Revisit | Entries are re-checked when their trigger occurs, for example a move to C++20 (which would lift the concepts exception) or an upstream GridPACK modernization (which would lift CMake exceptions). |

---

