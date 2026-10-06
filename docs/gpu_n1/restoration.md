# Case restoration and remote voltage regulation

The guide requires each batch member to start from one prepared network
(B8.2) and CPU fallbacks to restore it after reporting (B10.1). Full 10k
and Texas7k validation exposed a violation of that assumption.

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

The optional batch driver now saves the voltage references once and
restores references changed by a case after the existing cleanup. The
controller retains its adjustments while solving and reporting that
case. The saved reference agrees with the model used by all GPU members.
Configurations without an active batch path keep their existing behavior
(guide RT-4).

`test/data/IEEE14_remote.raw` gives bus 2 two generators: unit 1 regulates
locally and unit 2 regulates bus 5. The two-case list reports a branch
outage on the GPU and processes the generator outage on the CPU first.
The old implementation fails its shadow comparison by 0.0181 pu with
different PV counts; the correction agrees below 1e-15. The test runs
through the CPU reference, Alg2 and cuDSS backends.

A 65-case 10k sample also passed after the correction. Full studies are
still being checked. The unchanged stock application can retain this
controller state for later cases on the same rank; differences after
such an outage must be investigated and reported, not hidden by relaxed
tolerances or an ignored iteration column.
