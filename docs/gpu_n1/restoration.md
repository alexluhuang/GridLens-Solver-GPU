# Case restoration and remote voltage regulation

The guide builds each batch member from a prepared network (B8.2) and
describes restoration after CPU fallbacks (B10.1). The implementation treats
these as separate outage checks. Full 10k and Texas7k validation exposed
state carrying over between checks.

A bus can have both a locally regulating generator and a remotely
regulating generator. Tripping the local unit activates GridPACK's CPU
remote controller. `PFBus::adjustVoltageForRemoteReg` changes both the
present voltage and the reference used by `resetVoltage`. Contingency
cleanup restores the generator and reactive limits but leaves that
reference changed.

The batch path schedules its CPU-only cases before reporting GPU cases.
Consequently, the reporting rank that handled this outage subsequently
started shadow solves from a different voltage. In the 10k reproduction,
bus 74332's reference changed from 1.029 to 1.00047 pu. Later shadow
solves lost one PV bus and differed by about 0.022 pu. Both backends
showed the problem, at different event indices because work assignment
differs between runs.

Initially, the optional batch driver saved the voltage references once and
restored references changed by a case after the existing cleanup. The
controller retains its adjustments while solving and reporting that
case. The saved reference agrees with the model used by all GPU members.
That initial correction left the ordinary CPU path unchanged (guide RT-4).
The approved correction below now applies to both paths.

`test/data/IEEE14_remote.raw` gives bus 2 two generators: unit 1 regulates
locally and unit 2 regulates bus 5. The two-case list reports a branch
outage on the GPU and processes the generator outage on the CPU first.
The old implementation fails its shadow comparison by 0.0181 pu with
different PV counts; the correction agrees below 1e-15. The test runs
through the CPU reference, Alg2 and cuDSS backends.

A 65-case 10k sample also passed after the correction. Both full Texas and 10k
backends now pass every GPU shadow's voltage, angle, status and exact
bus-set checks. The default-order stock output still differs for some
late generator cases. The exact iteration comparison remains a failure.

A separate one-rank Texas reproduction proves the order dependence:
`GN_240278_1`, then `GN_250371_1`, then `GN_270208_1`. Stock takes two
iterations on the latter two cases after the remote-control outage, while
the restored batch path takes three. Their delta tables differ by up to
40.7604. Running only those latter two cases makes stock take three
iterations and pass all output comparisons against the batch path.

The unchanged stock application can therefore retain controller state
for later cases on the same rank. The batch model describes independent
outages from the prepared network. Preserving this independence conflicts
with FR-11's promise to match the original application and FR-14/RT-4's
promise that configurations without acceleration behave exactly as before.
These compatibility promises are explicit; interpreting B8.2/B10.1 as
requiring independence is an architectural conclusion. Do not hide the
original comparison failures with relaxed
tolerances or an ignored iteration column. An additional oracle study
may put the controller-activating outage last, using the same complete
case list for both paths, but must report that ordering and retain the
default-order failure evidence.

The exact small lists are preserved as
`reproductions/texas-controller-sequence.xml` and
`reproductions/texas-independent-cases.xml`. Run the existing
`batch_pf/test/run_ca_test.py` harness against `Texas7k_20210804.RAW` with
`--mode parity --backend alg2 --ranks 1 --contingency-list <list>` and a
separate unmodified `--stock-cax`. The three-case sequence is expected to
fail strict stock comparison; the two independent cases are expected to
pass. These lists support diagnosis, not a waived CI failure or a bundled
copy of the external RAW model.

## Approved correction to ordinary CPU behavior

On 2026-10-05 the user chose: "Restore settings in both paths; accept the
documented correction to existing behavior (recommended)." This overrides
the guide's compatibility promises only for this cleanup defect. The guide
itself remains unchanged. The driver now saves the prepared network's reset
references and restores changed references after reporting every case,
including when acceleration is absent or disabled.

Kundur and Malik §6.4 distinguishes a specified generator voltage from a
starting estimate for an unknown voltage. The retained value is not merely
an estimate: GridPACK resets a bus to it and can hold it fixed after the
local generator returns. A different fixed voltage can change the solved
flows and whether a generator reaches its supply limit. Keep controller
adjustments during the current outage; restore saved settings between the
independent outages described by the guide. This excerpt does not itself
define contingency cleanup; the study definition comes from guide 8.1.7.

`batchpf.parity.cpu_case_restoration` runs the existing IEEE14 fixture with
the controller outage first, then the branch outage, and runs that branch
outage alone in another process. It compares the later case's convergence,
delta, violations and identity tables. Before this correction it fails by
eight calculation rounds and up to 31.7952 in the delta table. After the
correction it passes in both CPU-only and CUDA-enabled builds. The three
existing backend restoration tests and absent/disabled path checks pass.

The CPU references are GridPACK's own ordinary calculation, run again after
resetting the case, and a separate unmodified executable at `b32969b0`.
Both use PETSc KLU. These are separate CPU executions, not a second modeling
package; a modeling error common to GridPACK and the accelerator can escape
them. The GPU's final state is not used as the CPU solve's starting state.

## Additional ordered-study discrepancy

The full Texas Alg2 CSV study with the controller outage last passes.
The corresponding cuDSS study has one strict failure: event 7822,
`BR_210326_210331_1`, reports zero CPU rounds and two GPU rounds. All other
tables and shadows pass; that shadow differs by 1.266e-14 pu and 1.732e-14
rad. Running the outage alone gives two rounds in both paths and passes
every comparison, including final PV/PQ counts. The full CPU log records a
limit check on disconnected generator bus 210331 followed by another
calculation with zero rounds. This is separate from the voltage-reference
defect. A two-case reproduction at `f2b6cfa4` now proves the cause:
`BR_210279_210278_1` diverges, leaving calculated injection data at bus
210331. `BR_210326_210331_1` then disconnects that bus. `rhsValues` skips
the disconnected bus, but `chkQlim` still checks its retained injection,
converts it and repeats the controller calculation. That final repeat needs
zero rounds; the connected grid's result is unchanged. The same latter
outage alone takes two rounds. The CPU restoration test reports only a
count discrepancy; its delta, violations and identity tables agree.

The two-case list is `reproductions/texas-isolated-injection-sequence.xml`.
The full comparison remains failed and this correction is not implemented.
Work stopped at the user's request. The independently prepared original-source
voltage-corrected reference patch is
`reproductions/cpu-voltage-cleanup-reference.patch`; it is not yet built.
