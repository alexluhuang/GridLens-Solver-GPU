# PSS/E v34+ Compatibility: Parser Fixes, Distributed Generation and Two-Terminal DC Lines

This note documents the changes made on the `feature/wecc-gridpack-compat`
branch relative to GridPACK `develop` (commit `b32969b0`), why each change was
needed, and how distributed generation (DG) and two-terminal dc lines are
modeled.

The work started from a full-size interconnection planning case in PSS/E v34
format (about 28,000 buses, 11,000 transformers, 1,800 loads with distributed
generation, six in-service line-commutated dc lines and a system switching
device section). The case is CEII and is not part of the repository; results
for it are given here only in aggregate. With `develop`, the power flow failed
on the first Newton iteration: the Jacobian was singular, and after the
singularities were removed the iteration still diverged. Six parser and
error-handling defects and two unmodeled data types were responsible. Each is
described below with its fix.

## Summary

| Change | Files | Effect |
|---|---|---|
| Comment stripping respects every quoted field | `parser/block_parsers/base_block_parser.cpp`, `parser/base_pti_parser.hpp` | A `/` in a transformer or branch name no longer truncates the record |
| System switching devices parsed | `parser/block_parsers/sys_switch_parser34.*`, `PTI34/35/36_parser.hpp` | Breakers and switches become branch elements instead of being dropped |
| Winding ratios in pu of bus base voltage | `transformer_parser33/34/35.cpp`, `base_block_parser.*` | CW=2 and CW=3 taps of 2- and 3-winding transformers are correct; LTC tap limits in consistent units; CZ=1 3-winding impedances not rescaled |
| Orientation of reversed parallel transformers | `transformer_parser33/34/35.cpp` | Tap and phase shift apply at the correct end |
| Solver error messages | `modules/powerflow/pf_app_module.cpp` | A solver failure is reported instead of crashing the run |
| Distributed generation | `load_parser33.cpp`, `load_defs.hpp`, `pf_components.*` | DGENP/DGENQ offset the load in the power flow |
| Two-terminal dc lines | `two_term_parser33.*`, `misc_defs.hpp`, `pf_hvdc.*`, `pf_factory_module.*`, `pf_app_module.*` | Line-commutated converters solved with the sequential ac/dc method, including control mode changes |
| DC line contingencies | `contingency_analysis/ca_driver.*`, `pf_app_module.*` | Pole and bipole outages in contingency analysis |

The changes are in these commits on top of `develop`:

| Commit | Change |
|---|---|
| `6836d546` | Distributed generation, two-terminal dc lines, dc line contingencies (sections 2–4) |
| `28029969` | Comment stripping (section 1.1) |
| `2dd7f412` | System switching devices (section 1.2) |
| `63c2b328` | Transformer winding ratios, LTC tap limits, 3-winding CZ=1 (section 1.3) |
| `ff0637bc` | Orientation of reversed parallel transformers (section 1.4) |
| `2e8ae395` | Power flow error messages (section 1.5) |

This note and the change log entry are in the commit that follows them.

On `feature/wecc-gpu-merged` the same changes are replayed on top of the GPU
batch contingency branch (`feature/gpu-batch-n1`), with the first commit
split in five; section 8 describes how the GPU path models DG and dc lines.

## 1. Parser and error-handling fixes

### 1.1 Comment stripping with several quoted fields

PSS/E RAW files use `/` to start a comment. `BaseBlockParser::cleanComment()`
(and the identical copy in `BasePTIParser`) only checked whether the first `/`
lay inside the *first* quoted field of the line. A transformer record's first
quoted field is the circuit ID, so a `/` in its name (e.g. `'GSU 500/230 KV'`)
was taken as a comment. The rest of the line, including the status field, was
discarded, and the transformer was loaded out of service. Branch records lost
their ratings and line shunts the same way, although their status defaulted to
in service.

In the large case, 85 in-service 2-winding transformers and one 3-winding
transformer were lost, leaving their buses (mostly generator terminals) with
no connection and the Jacobian singular.

**Fix.** `cleanComment()` scans the line once and tracks the quote state
(single or double quotes) across all fields; a `/` starts a comment only
outside quotes. Lines with no quoted `/` are stripped exactly as before.

### 1.2 System switching devices (v34–v36)

`SysSwitchParser34::parse()` read past the section without storing anything,
so closed breakers and switches were missing from the network. Buses connected
only through a closed device became isolated (all-zero Jacobian rows).

**Fix.** Each device record (`I, J, 'CKT', X, RATE1–12, STAT, ...`) is added
as a branch element: R = 0, X from the record (0.0001 pu, the PSS/E default,
when X is 0), no charging or line shunts, ratings RATE1–12 (RATE1–3 also as
ratings A–C), status STAT and tap 0 (not a transformer). A device between the
same two buses as an existing branch becomes a parallel element of it, with
the usual orientation bookkeeping. The v34, v35 and v36 parsers pass the
branch data to the device parser; the fields used are the same in all three
versions. Switching devices are branch elements like any other, so
`FullBranchN1` includes them.

### 1.3 Transformer winding ratios (CW codes)

PSS/E writes winding voltages according to the CW code: CW=1 in per unit of
the winding's bus base voltage, CW=2 in kV, CW=3 in per unit of the nominal
winding voltage NOMV (NOMV = 0 means the bus base voltage). NOMV itself only
matters for CW=3 and for magnetizing data with CM=2. Two defects followed:

* **3-winding transformers** stored WINDV unconverted, so CW=2 windings got a
  tap ratio equal to their kV rating (about 230 for a 230 kV winding). With
  small star-leg reactances this produced starting mismatches of order 10^4 pu
  at the star buses.
* **2-winding transformers** divided CW=2 winding voltages by NOMV instead of
  the bus base voltage. Where NOMV differs from the bus base voltage the ratio
  is wrong by NOMV/BASKV. In the large case 1,923 windings are affected; at
  the warm start the median mismatch at their buses was 0.107 pu with the
  NOMV convention and 0.0003 pu with the bus base voltage.

**Fix.** `BaseBlockParser::windingRatio(cw, windv, nomv, basekv)` returns the
ratio in per unit of the bus base voltage:

| CW | Ratio |
|---|---|
| 1 | `windv` |
| 2 | `windv / basekv` |
| 3 | `windv * nomv / basekv` (`nomv = basekv` if 0) |

`BaseBlockParser::busBaseKV()` supplies the base voltage. Both the 2-winding
path (tap `t = w1/w2`, series impedance referred through `w2^2` as before) and
each leg of a 3-winding transformer use it, in the v33, v34 and v35 parsers.
Two related corrections are in the same change:

* **LTC tap limits.** For voltage or reactive power control (|COD1| = 1 or 2),
  RMA1/RMI1 are winding 1 ratio limits in the units of WINDV1, but the LTC
  control in `PFBranch` compares them with the branch tap ratio `w1/w2`. They
  are now stored as `windingRatio(cw, RMA1, NOMV1, BASKV1) / w2`. Phase shift
  control limits (|COD1| = 3) are angles and are left unchanged.
* **3-winding impedance base.** The pairwise impedances of a 3-winding
  transformer were rescaled from the winding MVA base for every CZ code; with
  CZ=1 they are already on the system base and are no longer rescaled.

The RAW exporters write transformers with CW=1 and the stored WINDV1/WINDV2,
which are now in per unit of the bus base voltage as CW=1 requires.

### 1.4 Transformers merged into a reversed parallel branch

When a 2-winding transformer is defined between the same buses as an existing
branch or transformer but in the opposite direction, the parser adds it as an
element of the existing branch and computes `switched = true`, but never
stored it. The Y-matrix code (`YMBranch`) already handles `BRANCH_SWITCHED` by
applying the off-nominal ratio and phase shift at the branch's bus 2, but
without the flag it applied them at the wrong end, which inverts the tap
(about 1/t) and the sign of the phase shift.

**Fix.** The 2-winding paths of the v33, v34 and v35 transformer parsers store
`BRANCH_SWITCHED` for each element. In the large case this corrects two
phase-shifting transformers merged into reversed out-of-service jumpers and 15
pairs of parallel transformers defined in opposite directions.

### 1.5 Power flow error messages

`PFAppModule::solve()` formatted solver exceptions into a 128-byte buffer with
`sprintf`. PETSc error messages are far longer, so any linear solver failure
ended in a buffer overflow abort instead of a reported non-convergence.
`readNetwork()` had a format string with three conversions and two arguments.
Both now build the message as a `std::string`, and exceptions are caught by
reference.

## 2. Distributed generation

PSS/E v34 and later load records carry distributed generation behind the load:
`DGENP` (MW), `DGENQ` (MVar) and `DGENF` (1 = in service).

**Parsing.** `LoadParser33` (used by the v33–v36 readers) stores the three
fields as `LOAD_DGENP`, `LOAD_DGENQ` and `LOAD_DGENF` when a record has them.
Records from older formats end at INTRPT and are unaffected. The parser prints
the number of in-service units and their total output.

**Model.** `PFBus` keeps DG separate from the load (`p_dgp`, `p_dgq`,
`p_dgstatus`, one entry per load). DG is in service when DGENF is 1 and the
load is in service. It is a constant-power injection, independent of the
voltage-dependent (ZIP) part of the load:

```
P_demand = sum over in-service loads (PL - DGENP)  + P_dc
Q_demand = sum over in-service loads (QL - DGENQ)  + Q_dc
```

`PFBus::getFixedPowerDemand()` evaluates this (P_dc and Q_dc are the dc
converter terms, section 3) and is used everywhere the bus's fixed demand
enters: the specified injection (`setSBus`), the reactive power required from
generators (`chkQlim`), slack generation (`getTotalGenOutput`), generator
output reporting and the generator values saved to the data collection. Load
scaling (`scaleLoadPower`), load setting and load reporting act on the load
only. `PFBus::setDGStatus()` changes the status of the DG on a load, for use
by future DG tripping logic.

**Rationale.** This is the constant-power operating mode of inverter-based DG
in the steady-state models reviewed by Meena et al. (2024). The voltage
control (PV) mode needs a setpoint and reactive limits that the PSS/E load
record does not carry, so it is not modeled. In the large case the DG totals
11.4 GW with 32 MVar, effectively unity power factor. Ignoring it left the
warm start about 10.6 GW out of balance, which caused the remaining
divergence.

## 3. Two-terminal dc lines

### 3.1 Data

`TwoTermParser33` (used by the v33–v36 readers) previously read past the
section. It now stores each three-line record in the network-level data
collection, which every PTI parser broadcasts to all processes (fields
`HVDC_LINE_*`, `HVDC_RECT_*` and `HVDC_INV_*` in `misc_defs.hpp`):

* line: name, MDC, RDC, SETVL, VSCHD, VCMOD, RCOMP, DELTI
* each converter: ac bus, NB, ANMX, ANMN, RC, XC, EBAS, TR, TAP, TMX, TMN, STP

Fields after STP differ between PSS/E versions and are not used. Records whose
converter buses are not in the network are skipped.

### 3.2 Converter model

`pf_hvdc.cpp` models each converter as NB six-pulse bridges in series with
commutating reactance Xc and resistance Rc per bridge (standard line-commutated
converter equations; see e.g. Kundur, *Power System Stability and Control*,
Ch. 10). For ac voltage V (pu) at the converter bus and tap T:

```
E      = V * EBAS * TR / T                         bridge ac voltage (kV)
Vd'    = Vd + 2 NB Rc Id   (rectifier)              dc voltage behind Rc
Vd'    = Vd - 2 NB Rc Id   (inverter)
cos a  = (Vd'/NB + (3/pi) Xc Id) / ((3 sqrt2 / pi) E)
                                                    firing angle a (rectifier)
                                                    or extinction angle g (inverter)
cos a - cos(a + mu) = sqrt2 Xc Id / E               overlap angle mu
tan phi = (2 mu + sin 2a - sin 2(a + mu)) / (cos 2a - cos 2(a + mu))
P = Vd' Id,   Q = P tan phi
```

P is the ac active power drawn by the rectifier or delivered by the inverter
(the commutating resistance loss is on the ac side) and Q is the reactive power
absorbed by both. The exact power factor expression is used instead of the
common approximation `cos phi ~ (cos a + cos(a + mu))/2`; against the stored
solution of the large case it reproduces converter reactive power within about
1.5%, while the approximation is off by 4.6% at a converter with large overlap.

The converter tap starts at TAP and moves continuously within [TMN, TMX] only
when the angle would otherwise leave [ANMN, ANMX]. Limits given in decreasing
order are used in increasing order, with a warning.

### 3.3 Control modes

`solveTwoTerminalDC(line, Vr, Vi)` finds the operating point for given ac
voltages at the rectifier and inverter buses:

1. **Normal.** The inverter holds the compounded dc voltage,
   `Vdci + Id RCOMP = VSCHD`, and `Vdcr = Vdci + Id RDC`. For power control
   (MDC=1) SETVL > 0 is the dc power at the rectifier and SETVL < 0 the power
   at the inverter, which gives a quadratic in Id. For current control (MDC=2)
   `Id = SETVL/1000` (amps to kA).
2. **Inverter at minimum extinction angle.** If the inverter cannot hold the
   voltage with gamma >= ANMN even with its tap at a limit, it runs at gamma
   min and sets the dc voltage (`Vdci = a + b Id`, linear in Id). A power order
   is still met by raising the current. If Vdci then falls below VCMOD the line
   switches to the current that gives the order at VSCHD (`Id = |SETVL|/VSCHD`).
3. **Rectifier at minimum firing angle.** If the rectifier cannot supply Vdcr
   with alpha >= ANMN even with its tap at a limit, it runs at alpha min and the
   inverter takes current control with the order reduced by the margin:
   `Id = (1 - DELTI) Iorder`. The inverter angle is then solved for the
   resulting dc voltage.

If no operating point inside the angle limits exists (for example the power
order exceeds the maximum transfer at gamma min), the line runs at the nearest
feasible point and is reported as limited. Lines with MDC = 0 are blocked.

### 3.4 Sequential ac/dc solution

The ac network sees each converter as a constant P/Q injection at its bus.
The injections are recomputed in the power flow controller loop of
`PFAppModule::solve()`, alongside the IREG, Q-limit, switched shunt and LTC
checks:

1. At the start of a solve, `PFFactoryModule::startHVDC()` sets the converter
   injections: from the reference operating point if one has been set (see
   below), otherwise by solving the lines at the starting voltages.
2. After each converged Newton solution, `updateHVDC()` gathers the voltage
   magnitude and isolation flag of every converter bus (an `MPI_Allreduce`
   over the processes that own them, so every process solves every line
   identically), re-solves each line and updates the injections.
3. If any converter P or Q changed by more than `hvdcTolerance` (per unit on
   the system base; default equal to `tolerance`), the controller loop repeats.

This is the sequential method of AC/DC power flow (Khan and Bhowmick, *Power
Flow Modelling of HVDC Transmission Systems*, Sec. 1.7): the ac and dc
systems are solved separately and coupled through equivalent injections at
the converter buses. It converges linearly; the largest converter change fell
by a factor of 4 to 20 per pass in the tests. A unified formulation (dc
variables in the Jacobian) was not needed.

A line is blocked, with zero injection at both ends, if it is scheduled
blocked (MDC = 0), taken out of service by a contingency, or has an isolated
converter bus.

`PFAppModule::setHVDCReference()` makes the current operating point the
starting point of later solves. Contingency analysis calls it after the base
case, so each contingency starts its dc lines from the base-case operating
point (as its ac solution starts from the base voltages). This avoids
re-converging the base-case operating point in every contingency, and keeps
results independent of the order in which contingencies run.

`PFAppModule::write()` and contingency analysis (base case) print a table of
the dc lines: control mode, current, and P, Q and angle at each converter.

## 4. Contingency analysis

* **Contingency type `DCLine`** (or `HVDC`) blocks one or more dc lines. Each
  PSS/E two-terminal record is one pole, so a bipole outage lists both poles.
  Names can contain blanks, so several names are separated by `;` or `,`, and
  names are matched after trimming and collapsing white space:

  ```xml
  <Contingency>
    <contingencyType>DCLine</contingencyType>
    <contingencyName>BIPOLE_1</contingencyName>
    <contingencyDCLines>POLE_1; POLE_2</contingencyDCLines>
  </Contingency>
  ```

* **`FullHVDCN1`** adds one contingency per in-service dc line, named
  `DC_<line name>`. Duplicates from a contingency file are skipped.
* **Output.** The contingency `type` is `hvdc` in every output, and
  `_contingencies.csv` has a trailing `dc_line` column with the blocked line
  names (`;`-separated).

## 5. New configuration options

| Block | Option | Default | Meaning |
|---|---|---|---|
| `Powerflow` | `hvdcTolerance` | `tolerance` | Largest change (pu) in dc converter injections between controller passes |
| `Contingency_analysis` | `FullHVDCN1` | `false` | Auto-generate dc line (pole) outages |
| contingency file | `contingencyDCLines` | | Lines blocked by a `DCLine` contingency |

## 6. Validation

* **Regression.** 16 data sets in `src/applications/data_sets/raw` (PSS/E v23,
  v33, v34 and v36) give byte-identical power flow output before and after all
  changes. The v33–v36 data sets use only CW=1 and contain no DG, dc lines,
  switching devices, quoted `/` in names or reversed parallel transformers; the
  v23 data sets are read by the v23 parser, where only comment stripping
  changed.
* **Parser fixes.** Pairs of physically identical cases written two ways, built
  from `240busWECC_2018_PSS_fixedshunt.raw`:

  | Pair | Original code (max dV) | Fixed code (max dV) |
  |---|---|---|
  | Transformer name with and without `/` | 1.6e-2 pu | 0 |
  | Tie as branch record vs as switching device | 1.07 pu | 0 |
  | Windings in CW=1 vs CW=2/CW=3 (NOMV different from bus base voltage) | 8.9e-2 pu | 0 |
  | Phase-shifting transformer written I->J vs J->I next to a parallel line | 3.4e-2 pu | 0 |

  Forcing a linear solver failure crashed the original code with a buffer
  overflow on every process; the fixed code reports the exception and returns
  a failed solve.
* **Distributed generation.** Loads with DG give a bit-identical solution to
  the same loads with DG subtracted from PL and QL.
* **DC lines.** The converter solver agrees with an independent implementation
  of the same model on 6,008 random operating points covering all control
  modes (largest relative difference 5e-12). Every mode satisfies its
  invariants (power order met, compounded voltage held, current reduced by the
  margin at alpha min, angles inside their limits). On the 240-bus case with
  added dc lines, the sequential solution is a fixed point (re-solving the
  converters at the converged voltages and applying the result as fixed loads
  reproduces the solution), forced alpha min, gamma min and VCMOD cases switch
  mode as specified, and 1- and 4-process runs give the same solution.
* **Large planning case.** The unmodified case now converges in 3 Newton
  iterations from its stored solution (plus 5 sequential dc passes). Bus
  voltages agree with the PSS/E solution stored in the case to a median of
  2e-4 pu and a 99th percentile of 4e-3 pu. A contingency analysis of 659
  contingencies (651 branch and generator outages plus 6 pole and 2 bipole
  outages) gave 553 solved, 106 radial islands and no divergence.

## 7. Known remaining gaps

* FACTS devices (SVCs and STATCOMs) are not modeled; their buses account for
  the largest remaining voltage differences from PSS/E in the large case.
* VSC dc lines and multi-terminal dc lines are not modeled.
* Converter taps move continuously (STP is not applied). The firing-angle
  measuring bus (ICR) and capacitor-commutated converters (XCAP) are not
  modeled.
* Converter P and Q are held constant within each Newton solution and updated
  between solutions; a unified formulation would make them part of the
  Jacobian.
* Zero-impedance branches (|Z| < THRSHZ) are modeled with their small
  impedance rather than merged; this was not needed for convergence.
* Only the power flow and contingency analysis use DG and dc converter
  injections; dynamic simulation reads loads directly.
* KLU is a serial solver; multi-process runs need SuperLU_DIST or MUMPS.

## 8. GPU batch contingency path (`feature/wecc-gpu-merged`)

The GPU branch solves eligible contingency cases together on the GPU with
one shared Jacobian pattern (its `docs/gpu_n1` and
`modules/batch_pf/README.md`). Merging the two branches needed seven small
textual resolutions and four changes for cases the GPU path did not know
about.

### 8.1 Textual resolutions

| Where | Resolution |
|---|---|
| `powerflow/CMakeLists.txt`, `pf_factory_module.hpp` | Install and include both `pf_hvdc.hpp` and `pf_superset_model.hpp` (full `gridpack/...` include paths) |
| `pf_factory_module.cpp` | Union of the standard includes |
| `pf_components.cpp` (`serialWrite`, `saveData`, `saveDataAlsotoOrg`) | The GPU branch's run-time Jacobian layout switch (`if (!p_largeMatrix)` instead of `#ifndef LARGE_MATRIX`) with this branch's `getFixedPowerDemand()`, so reported generator output includes DG and converters |
| `ca_driver.cpp` (`csv_delta` rows) | The GPU branch's direct component reads (`readBuses()`) with `contingencyTypeName()` |

### 8.2 Two-terminal dc lines on the GPU

A dc line enters the ac equations only through the power its converters
draw, so the GPU path keeps one Jacobian pattern for every case (Zhou et al.
2017) and solves the dc lines sequentially (Khan and Bhowmick, ch. 3.3.2),
as `PFAppModule::solve()` does:

* The numeric converter model moved into `pf_hvdc.hpp` as inline functions
  on plain data that compile as GPU device functions too
  (`gridpack::hvdc_model` interface target). CPU results on the large case
  are byte-identical to before the move.
* Each case starts its lines at the base-case operating point, blocked if
  out of service or if a converter bus is isolated, and re-solves them after
  each converged Newton loop, with the reactive-limit check and before the
  last step. A change above `hvdcTolerance` repeats the controller
  iteration; the contingency driver's second solve starts the lines again
  (`core/dc_kernels.cuh`, `core/engine_control.cpp`).
* A dc pole outage changes neither admittances nor topology. The classifier
  sends it to the GPU with the indices of the blocked lines; batches carry
  each case's line statuses.
* The final dc operating points return with each result and are set in
  GridPACK before its reports, so slack output, flows and checks see the
  converter injections the solution was found with.
* Removing a contingency now also restores the converter injections, as it
  restores switched shunts and taps; before, a solve left its converter
  state in the network for the next caller.

### 8.3 Reactive-limit demand

GridPACK's `chkQlim()` compares the generators' limits with the bus demand
of `getFixedPowerDemand()`: loads less their DG plus dc converter draw
(Kundur 6.4: the limits belong to the generators; other devices at the bus
are demand, and the constant-power DG model of Meena et al. is a negative
load). The GPU check used raw loads. It now receives DG Q (`dg_q`) and the
converter Q of the case as separate fields and forms the same demand.

### 8.4 Labels and interface version

`contingencyTypeName()` labels every row, including the convergence rows
written for missing GPU outcomes. The plugin interface is version 1.1: DG,
dc line and dc state records are appended to the model, solver, batch and
results records, and callers built for 1.0 still work. With a plugin older
than 1.1, networks with DG or dc lines use the CPU loop.

### 8.5 Results

* **Large planning case** (2,010 cases: every branch and generator outage
  within three buses of a converter, 1,500 sampled elsewhere, and every pole
  outage; tolerance 1e-4): 1,655 cases on the GPU and 355 radial islands on
  the CPU path, identical on Algorithm 2 and cuDSS. Every shadow re-solve
  agrees in status, PV/PQ sets and classification (largest voltage
  difference 3e-7 pu). Against the CPU path, 54 million flow rows and
  240,000 violation rows agree within 1e-3 or one printed digit except in
  two cases, where the dc change at the deciding controller iteration was
  1.000e-04 pu, the `hvdcTolerance` itself: one path took one more
  sequential step, which moves converter-bus flows by 0.01 MVA (the
  tolerance) and the final Newton count by one. A tighter `hvdcTolerance`
  than the Newton tolerance (for example 1e-5 with 1e-4) makes such ties
  rare. Without shadow re-solves (the production setting) the study takes
  188.2 s on the CPU path, 25.7 s with Algorithm 2 and 45.4 s with cuDSS at
  16 processes, with the same statuses and summary counts.
* **Every input file.** Full N-1 parity with every dc pole outage on all
  31 RAW files in `data_sets/raw` (v23 to v36), the public test network and
  three external validation grids up to 10,000 buses passes on Algorithm 2
  and cuDSS (`docs/gpu_n1/validation.md`).
* **Public test network** (`batch_pf/test/make_dc_case.py`: the 240-bus
  WECC case with a power-order, a current-order and a blocked dc line, DG
  on 27 loads and one DG unit out of service): full N-1 with both pole
  outages matches the CPU path on the CPU reference, Algorithm 2 and cuDSS
  backends (largest shadow voltage difference 4e-14 pu);
  `ctest -R ca_dc_dg`.

## References

* P. Kundur, *Power System Stability and Control*, McGraw-Hill, Sec. 6.4
  (power flow) and Ch. 10 (HVDC).
* S. Khan and S. Bhowmick, *Power-Flow Modelling of HVDC Transmission
  Systems*, CRC Press, Ch. 1.
* G. Meena, V. P. Singh, A. Dixit, A. Mathur and A. Bhatt, "Steady-State
  Models of AC Microgrid for Load Flow Solutions," ISTEMS 2024,
  doi:10.1109/ISTEMS60181.2024.10560356.
* Siemens PTI, *PSS/E Program Operation Manual* (RAW file format, versions 33
  to 36).
* G. Zhou, R. Bo, L. Chien, X. Zhang, F. Shi, C. Xu and Y. Feng, "GPU-Based
  Batch LU-Factorization Solver for Concurrent Analysis of Massive Power
  Flows," *IEEE Trans. Power Systems*, 32(6), 2017,
  doi:10.1109/TPWRS.2017.2662322.
* M. D'Orto, S. Sjöblom, L. S. Chien, L. Axner and J. Gong, "Comparing
  Different Approaches for Solving Large Scale Power-Flow Problems With the
  Newton-Raphson Method," *IEEE Access*, 9, 2021,
  doi:10.1109/ACCESS.2021.3072338.
