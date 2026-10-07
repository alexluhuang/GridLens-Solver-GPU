#!/bin/sh
#
#     Copyright (c) 2013 Battelle Memorial Institute
#     Licensed under modified BSD License. A copy of this license can be
#     found in the LICENSE file in the top level directory of this
#     distribution.
#
# Build the two programs of the matrix inside this repository:
#
#   matrix/builds/optimized   this source tree (optimized CPU, Alg 2, cuDSS)
#   matrix/builds/stock       stock GridPACK b32969b0 plus timing lines only
#                             (docs/gpu_n1/reproductions/stock-timer-only.patch)
#
# Both are release builds made in the same container. Run from anywhere:
#   tools/ca_matrix/build.sh
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUN="$REPO/tools/ca_matrix/run.sh"
mkdir -p "$REPO/matrix/builds" "$REPO/matrix/networks"
STOCK_SRC="$REPO/matrix/builds/stock-src"
if [ ! -d "$STOCK_SRC" ]; then
  mkdir "$STOCK_SRC"
  git -C "$REPO" archive b32969b0 | tar -x -C "$STOCK_SRC"
  patch -d "$STOCK_SRC" -p1 < "$REPO/docs/gpu_n1/reproductions/stock-timer-only.patch"
fi
JOBS=$(nproc)
"$RUN" "sh /src/tools/ca_matrix/configure.sh /src/src /src/matrix/builds/optimized \
          -D CMAKE_CUDA_ARCHITECTURES=native \
        && cmake --build /src/matrix/builds/optimized -j $JOBS --target ca.x gridpack_batchpf_core \
        && (cmake --build /src/matrix/builds/optimized -j $JOBS --target gridpack_batchpf_cudss \
            || echo 'cuDSS backend not built (cuDSS not found)')"
"$RUN" --nogpu "sh /src/tools/ca_matrix/configure.sh /src/matrix/builds/stock-src/src /src/matrix/builds/stock \
        && cmake --build /src/matrix/builds/stock -j $JOBS --target ca.x"
echo "built matrix/builds/optimized and matrix/builds/stock"
