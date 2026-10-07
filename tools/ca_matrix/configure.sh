#!/bin/sh
#
#     Copyright (c) 2013 Battelle Memorial Institute
#     Licensed under modified BSD License. A copy of this license can be
#     found in the LICENSE file in the top level directory of this
#     distribution.
#
# Configure a release build of GridPACK (run inside the container).
#   configure.sh <source dir> <build dir> [extra cmake arguments]
set -e
SRC=$1; BUILD=$2; shift 2
cmake -S "$SRC" -B "$BUILD" -Wno-dev \
  -D GA_DIR:STRING="${ga_gp_dir}" \
  -D Boost_ROOT:STRING="${boost_gp_dir}" \
  -D Boost_DIR:STRING="${boost_gp_dir}/lib/cmake/Boost-1.81.0" \
  -D PETSC_DIR:PATH="${petsc_gp_dir}" \
  -D MPI_CXX_COMPILER:STRING=mpicxx \
  -D MPI_C_COMPILER:STRING=mpicc \
  -D MPIEXEC:STRING=mpiexec \
  -D ENABLE_ENVIRONMENT_FROM_COMM:BOOL=YES \
  -D CMAKE_BUILD_TYPE:STRING=Release \
  -D BUILD_SHARED_LIBS=true \
  -D GRIDPACK_ENABLE_TESTS:BOOL=OFF \
  "$@"
