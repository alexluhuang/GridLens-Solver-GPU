#!/bin/sh
#
#     Copyright (c) 2013 Battelle Memorial Institute
#     Licensed under modified BSD License. A copy of this license can be
#     found in the LICENSE file in the top level directory of this
#     distribution.
#
# Run a command in the measurement container, with this repository at /src.
# Every build and run of the matrix goes through here, so all of them use
# the same compilers, libraries and MPI.
#
#   tools/ca_matrix/run.sh [--nogpu] '<command>'
#
# CA_MATRIX_IMAGE selects another image (default alh360/gridpack-n1-tools:1.2,
# built by Dockerfile.tools: image 1.1 plus Polars and its GPU engine).
# CA_MATRIX_MEMORY_GB sets the container's memory cap (default 64).

REPO=$(cd "$(dirname "$0")/../.." && pwd)
GPU="--gpus all"
if [ "$1" = "--nogpu" ]; then GPU=""; shift; fi

# Hard memory cap for the container (CA_MATRIX_MEMORY_GB, default 64), and
# a check that the machine has that much available first. Running a shared
# machine out of memory stops other people's work too.
CAP_GB=${CA_MATRIX_MEMORY_GB:-64}
AVAIL_GB=$(awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
if [ "$AVAIL_GB" -lt "$CAP_GB" ]; then
  echo "run.sh: only $AVAIL_GB GB of memory available, less than the ${CAP_GB} GB cap;" \
       "wait for other work to finish or set CA_MATRIX_MEMORY_GB lower" >&2
  exit 1
fi
exec docker run --rm $GPU --ipc=host -u "$(id -u):$(id -g)" \
  --memory="${CAP_GB}g" --memory-swap="${CAP_GB}g" \
  -v "$REPO":/src -w /src --entrypoint bash \
  "${CA_MATRIX_IMAGE:-alh360/gridpack-n1-tools:1.2}" -lc "$*"
