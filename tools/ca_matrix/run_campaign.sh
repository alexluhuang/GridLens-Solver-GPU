#!/bin/sh
#
#     Copyright (c) 2013 Battelle Memorial Institute
#     Licensed under modified BSD License. A copy of this license can be
#     found in the LICENSE file in the top level directory of this
#     distribution.
#
# The full test matrix: every network in matrix/networks, smallest file
# first, at 8, 16 and 20 ranks, on stock GridPACK and on the optimized CPU,
# Alg 2 and cuDSS paths, one study at a time. Every input is the template
# (default test_runs/input.xml) with only the solve path, outputFile and
# networkConfiguration changed. Result tables are kept, as compact copies, at
# 16 ranks only, for compare_runs.py. Already finished runs are skipped, so
# the script can be started again after an interruption.
#
#   tools/ca_matrix/run_campaign.sh [template, as seen in the container]
#
# REPEATS (default 2) runs each study that many times; networks over
# LARGE_MB megabytes (default 10) run LARGE_REPEATS times (default 1).

REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUN="$REPO/tools/ca_matrix/run.sh"
TEMPLATE=${1:-/src/test_runs/input.xml}
REPEATS=${REPEATS:-2}
LARGE_REPEATS=${LARGE_REPEATS:-1}
LARGE_MB=${LARGE_MB:-10}
TOOL=/src/tools/ca_matrix/run_matrix.py

for net in $(ls -Sr "$REPO/matrix/networks"); do
  size_mb=$(( $(stat -c %s "$REPO/matrix/networks/$net") / 1000000 ))
  n=$REPEATS
  [ "$size_mb" -gt "$LARGE_MB" ] && n=$LARGE_REPEATS
  for ranks in 8 16 20; do
    keep=""
    [ "$ranks" = 16 ] && keep="--keep"
    echo "== $(date '+%F %T') $net, $ranks ranks, $n repeat(s)"
    "$RUN" "python3 $TOOL --program stock --template $TEMPLATE --networks $net \
            --ranks $ranks --repeat $n $keep --resume" \
      || echo "stock runs of $net at $ranks ranks did not finish"
    "$RUN" "python3 $TOOL --program ours --template $TEMPLATE --networks $net \
            --paths cpu,alg2,cudss --ranks $ranks --repeat $n $keep --resume" \
      || echo "optimized runs of $net at $ranks ranks did not finish"
  done
done
echo "== $(date '+%F %T') campaign finished"
