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

The stock baseline is GridPACK commit `b32969b0`. Work continues on
`feature/gpu-batch-n1`; commits are local and are not pushed.
