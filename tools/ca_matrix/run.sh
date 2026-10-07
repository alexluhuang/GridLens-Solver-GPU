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
# CA_MATRIX_IMAGE selects another image (default alh360/gridpack-n1-tools:1.1).

REPO=$(cd "$(dirname "$0")/../.." && pwd)
GPU="--gpus all"
if [ "$1" = "--nogpu" ]; then GPU=""; shift; fi
exec docker run --rm $GPU --ipc=host -u "$(id -u):$(id -g)" \
  -v "$REPO":/src -w /src --entrypoint bash \
  "${CA_MATRIX_IMAGE:-alh360/gridpack-n1-tools:1.1}" -lc "$*"
