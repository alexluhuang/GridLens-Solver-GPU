# Contingency analysis test matrix

Runs full N-1 studies over several networks, solve paths and rank counts,
and reports two things for each run, both against the optimized CPU path of
the same network and rank count:

- **accuracy**: do its result tables match?
- **runtime**: how long each step of the pipeline took, and that time as a
  multiple of the optimized CPU time.

The four solve paths:

| Path | Program | `--program` / `--paths` |
|---|---|---|
| Stock GridPACK | `b32969b0` plus timing lines only | `stock` / `cpu` |
| Optimized CPU | this repository, no GPU | `ours` / `cpu` |
| Alg 2 | this repository, GPU | `ours` / `alg2` |
| cuDSS | this repository, GPU | `ours` / `cudss` |

Everything is kept in `matrix/` at the top of the repository, which git
ignores: `networks/`, `builds/`, `out/` (one folder per run) and
`results.jsonl` (one line per run).

## Steps

Run these from the top of the repository.

1. **Build both programs** (once, and again after changing the source):

   ```sh
   tools/ca_matrix/build.sh
   ```

   Both are release builds made in the same container, so they use the same
   compilers and libraries.

2. **Add networks**: copy PSS/E `.raw` files into `matrix/networks/`.
   Networks above 10,000 buses have not been tested.

3. **Run the matrix.** Repeat for each rank count. `--repeat 2` runs each
   study twice for steadier times; `--keep` keeps the first run's tables
   for step 4.

   ```sh
   N=MemphisCase2026_Mar7.RAW,Texas7k_20210804.RAW,ACTIVSg10k.RAW
   for R in 4 8 16; do
     tools/ca_matrix/run.sh "python3 /src/tools/ca_matrix/run_matrix.py --program stock --networks $N --paths cpu --ranks $R --repeat 2 --keep"
     tools/ca_matrix/run.sh "python3 /src/tools/ca_matrix/run_matrix.py --program ours --networks $N --paths cpu,alg2,cudss --ranks $R --repeat 2 --keep"
   done
   ```

   Leaving out `--ranks` uses the cores rule of `ca_run.sh` (16 on a
   20-core DGX Spark). Run nothing else on the machine meanwhile.

   GPU paths normally start each case from the solved base case (the
   production setting, so the times are what a user would see). The CPU
   paths start from the network file's voltages, so the number of solver
   steps per case differs. To compare those counts too, add a run with
   `--start file`; it is kept and reported separately.

4. **Accuracy against optimized CPU:**

   ```sh
   tools/ca_matrix/run.sh "python3 /src/tools/ca_matrix/compare_runs.py"
   ```

   One line per kept run: PASS if every table matches the optimized CPU
   run within 0.001 (same cases, same convergence, and the same number of
   solver steps unless the run started from the base case), or FAIL with
   the first differences. Stock GridPACK fails on some networks: it
   carries a voltage setting from one case into later cases on the same
   rank, so a few results change from run to run (see
   `docs/gpu_n1/restoration.md`).

5. **Runtime against optimized CPU:**

   ```sh
   python3 tools/ca_matrix/matrix_report.py --csv matrix/report.csv
   ```

   For every network, rank count and path: the total time and each step
   (read network, base case, case setup, all cases, merge files, and inside
   the cases: apply outage, solve, inject GPU result, check and report,
   write rows, restore), each followed by its multiple of the optimized CPU
   time. `matrix/report.csv` holds the same table for a spreadsheet.

6. **Free disk space** once step 4 is done (a 10k-bus run writes about
   17 GB of tables):

   ```sh
   find matrix/out -name 'ca_results_*.csv' -size +1M -delete
   ```

   `results.jsonl` and the run logs stay, so step 5 still works.

To start over, delete `matrix/out` and `matrix/results.jsonl`.
