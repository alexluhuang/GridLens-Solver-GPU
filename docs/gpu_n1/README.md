# GPU N-1 implementation

The supplied architecture guide is preserved without edits in five files.
Read them in this order:

1. [Requirements and building blocks](architecture/01-requirements-and-blocks.md)
2. [Runtime and deployment](architecture/02-runtime-and-deployment.md)
3. [Equations, memory, settings, and standards](architecture/03-concepts.md)
4. [Decisions, validation gates, and reference methods](architecture/04-decisions-and-methods.md)
5. [Standards register, settings reference, and sources](architecture/05-standards-settings-and-references.md)

Concatenating these files reproduces all 2,637 lines of
`N1_Contingency_Solver_Architecture_Guide_v0.3.md`. Its SHA256 is
`cbfce273ea1d2a192de51677d869b6707ee93cad2c57f4c5d16927456df6d524`.
The split permits reviewable commits of at most 1,000 changed lines.

The guide describes proposals as well as requirements. Its platform and
software version claims are retained as supplied; they are not evidence of
testing on those platforms. The implementation's measured results and open
gates belong in the validation report, separately from this design record.

For troubleshooting, consult the relevant guide section first. For physics
and modeling, then consult Kundur and Malik, section 6.4 (`PF.pdf`), before
the research papers. For software behavior, read the GridPACK implementation
before vendor documentation. The two algorithm papers are D'Orto et al.
(2021, KLU planning) and Zhou et al. (2017, batch factorization).

The original stock baseline is GridPACK commit `b32969b0`. The user approved
correcting voltage cleanup in both CPU and GPU paths; comparisons with this
original baseline retain the resulting known differences. A second, separate
correction stops the reactive-limit check from using a stale injection at a
disconnected bus. The corrected CPU reference is `b32969b0` plus the two
patches in `reproductions/`, built separately from the original. Work
continues on `feature/gpu-batch-n1`; commits are local and are not pushed.

Implementation evidence is in [validation.md](validation.md) and the
[structured status](validation-status.json). The
[restoration diagnosis](restoration.md) explains the approved correction and
the remaining comparison failures. [standards.md](standards.md) records analysis scope,
tool limitations and exceptions awaiting maintainer review.
The repeated single-Spark timings and kernel-counter limits are documented
in [performance.md](performance.md).

The [methodology report](report/01-methodology.md) explains every change
from stock GridPACK to the optimized CPU path and to the Alg 2 and cuDSS
paths, how this work differs from the published work it builds on
([report/02-literature.md](report/02-literature.md)), and the full test
matrix of seven networks, three process counts and four versions
([report/03-results.md](report/03-results.md)).
