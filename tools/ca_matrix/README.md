# Contingency analysis test matrix

Runs full N-1 studies over several networks, solve paths and rank counts,
and reports two things, both against the optimized CPU path of the same
network:

- **accuracy**, tested at 16 ranks: absolute and percentage differences of
  utilization, power and bus values over all result rows, and the share of
  rows more than 1% off;
- **runtime**: each step of the pipeline at every rank count, with its
  absolute and percentage difference from optimized CPU at the same rank
  count.

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
     K=$([ $R = 16 ] && echo --keep)   # tables are needed only at 16 ranks (step 4)
     tools/ca_matrix/run.sh "python3 /src/tools/ca_matrix/run_matrix.py --program stock --networks $N --paths cpu --ranks $R --repeat 2 $K"
     tools/ca_matrix/run.sh "python3 /src/tools/ca_matrix/run_matrix.py --program ours --networks $N --paths cpu,alg2,cudss --ranks $R --repeat 2 $K"
   done
   ```

   Leaving out `--ranks` uses the cores rule of `ca_run.sh` (16 on a
   20-core DGX Spark). Run nothing else on the machine meanwhile: before
   each run the script waits up to 10 minutes for other work on the machine
   to finish, and marks a run made while it was still busy; the runtime
   report flags such runs.

   `run.sh` caps the container's memory at 64 GB (set
   `CA_MATRIX_MEMORY_GB` to change it) and refuses to start if the machine
   has less than that available, so a run cannot use up the memory of a
   shared machine.

   GPU paths normally start each case from the solved base case (the
   production setting, so the times are what a user would see). The CPU
   paths start from the network file's voltages, so the number of solver
   steps per case differs. To compare those counts too, add a run with
   `--start file`; it is kept and reported separately.

4. **Accuracy against optimized CPU:**

   ```sh
   tools/ca_matrix/run.sh "python3 /src/tools/ca_matrix/compare_runs.py"
   ```

   Accuracy is tested at 16 ranks only (`--ranks` to choose another
   count): a case's results do not depend on the rank count. Each kept
   16-rank run's result table is matched row by row with the optimized CPU
   run of the same network. For utilization, complex
   power, real power, reactive power, bus voltage and bus angle it prints
   the mean and largest absolute difference, the mean and largest
   percentage difference, and the percentage of rows more than 1% away from
   the CPU value, all to four decimal places, and writes them to
   `matrix/accuracy.csv`. It also counts rows found in only one of the two
   runs. Solver steps are not compared: the GPU paths take different steps
   by design.

   Add `--rows` to also save every row's values and differences
   (`matrix/accuracy_rows/`, about 9 GB for a 10,000-bus network).
   `--engine gpu` runs the comparison on the GPU; the default CPU engine
   was faster on the DGX Spark (9 s against 12 s for 150 million rows).
   The first comparison of a run converts its table to a compact copy
   (`flat.parquet/` in the run folder, about 20 s for 17 GB),
   which later comparisons reuse.

5. **Runtime against optimized CPU:**

   ```sh
   python3 tools/ca_matrix/matrix_report.py --csv matrix/report.csv
   ```

   For every network, rank count and path, and every step (total, read
   network, base case, case setup, all cases, merge files, and inside the
   cases: apply outage, solve, inject GPU result, check and report, write
   rows, restore): the seconds, the difference from optimized CPU in
   seconds, and the difference in percent (negative means faster), to four
   decimal places. `matrix/report.csv` holds the same table for a
   spreadsheet. With `--repeat 2` or more, each value is the middle value
   of the repeats; check that repeats agree, since a busy machine slows a
   run.

6. **Trends** across the whole matrix:

   ```sh
   python3 tools/ca_matrix/analyze_matrix.py
   ```

   Writes tables and figures to `matrix/analysis/`: speedups against stock
   and optimized CPU, every pipeline part's seconds and share of wall time,
   which parts account for each version's saving, scaling with ranks and
   with network size, GPU engine figures from the logs (cases on the GPU,
   retries, occupancy, busy share of the case loop), Alg 2 against cuDSS
   with the fill of the factors, case outcomes and Newton steps, agreement
   of the violation tables, and peak memory, with `summary.md` holding them
   all as Markdown. It needs only the Python standard library. Networks not
   in its list of public networks are called "private network" in every
   output.

7. **Free disk space** once step 4 is done (a 10,000-bus run writes about
   17 GB of tables):

   ```sh
   find matrix/out -name '*_flat.csv' -size +1M -delete
   rm -rf matrix/accuracy_rows
   ```

   `results.jsonl`, the run logs and the compact copies stay, so steps 4
   to 6 still work.

To start over, delete `matrix/out` and `matrix/results.jsonl`.
