#!/bin/sh
#
#     Copyright (c) 2013 Battelle Memorial Institute
#     Licensed under modified BSD License. A copy of this license can be
#     found in the LICENSE file in the top level directory of this
#     distribution.
#
# Run contingency analysis with an MPI rank count chosen from the cores this
# process may use, leaving some cores free for the operating system, MPI
# progress and the GPU worker thread of the batch path:
#
#   8 or more cores      cores - 4
#   fewer than 8 cores   cores - 2, and at least 1
#
# GRIDPACK_CA_RANKS overrides the count; GRIDPACK_MPIEXEC names the launcher
# (default mpiexec). Arguments are passed to ca.x, normally the input file:
#
#   ca_run.sh input.xml

set -e

cores=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)
if [ -n "${GRIDPACK_CA_RANKS}" ]; then
  ranks=${GRIDPACK_CA_RANKS}
elif [ "${cores}" -ge 8 ]; then
  ranks=$((cores - 4))
else
  ranks=$((cores - 2))
  [ "${ranks}" -ge 1 ] || ranks=1
fi

here=$(dirname "$0")
cax=${here}/ca.x
[ -x "${cax}" ] || cax=ca.x
exec "${GRIDPACK_MPIEXEC:-mpiexec}" --bind-to none -n "${ranks}" "${cax}" "$@"
