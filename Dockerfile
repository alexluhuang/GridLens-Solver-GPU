# syntax=docker/dockerfile:1.19
# check=error=true

# Base image. The default builds today's CPU-only image; a GPU image uses an
# NVIDIA CUDA development image with a full version tag, for example
#   --build-arg BASE_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04
ARG BASE_IMAGE=ubuntu:questing
FROM ${BASE_IMAGE}

# Configure dependency versions
ARG boost_version=1.81.0
ARG ga_version=5.9.1
ARG petsc_version=3.24.2

# GPU batch contingency path (build-time choices only; everything else is a
# run-time setting in configuration.xml). The defaults reproduce the
# CPU-only image, including its Debug build type; GPU images should use
# GRIDPACK_BUILD_TYPE=Release.
#   GRIDPACK_ENABLE_GPU_BATCH  OFF, AUTO or ON (needs a CUDA base image)
#   CUDA_ARCHITECTURES         GPU code targets; "all" = every supported GPU
#                              plus PTX for newer ones
#   CUDSS_APT_PACKAGE          cuDSS package from NVIDIA's repository
#   GRIDPACK_ENABLE_PARQUET    OFF or ON: outputFormat=parquet, using Apache
#                              Arrow's Parquet library from Apache's apt
#                              repository (Ubuntu releases it publishes,
#                              e.g. 24.04; not the default questing base)
ARG GRIDPACK_ENABLE_GPU_BATCH=OFF
ARG GRIDPACK_ENABLE_PARQUET=OFF
ARG CUDA_ARCHITECTURES=all
ARG CUDSS_APT_PACKAGE=cudss
ARG GRIDPACK_BUILD_TYPE=Debug
ARG GRIDPACK_IMAGE_VERSION=dev

LABEL org.opencontainers.image.title="GridPACK" \
      org.opencontainers.image.description="GridPACK power grid simulation framework; optional GPU batch N-1 contingency analysis" \
      org.opencontainers.image.source="https://github.com/GridOPTICS/GridPACK" \
      org.opencontainers.image.version="${GRIDPACK_IMAGE_VERSION}" \
      org.opencontainers.image.licenses="BSD-2-Clause"

# Setup environment variables used throughout installation
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC GNUMAKEFLAGS=--no-print-directory
ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

ENV GRIDPACK_ROOT_DIR=/app GP_EXT_DEPS=/deps
ENV GRIDPACK_INSTALL_DIR=${GRIDPACK_ROOT_DIR}/src/install GRIDPACK_BUILD_DIR=${GRIDPACK_ROOT_DIR}/src/build
ENV GRIDPACK_DIR=${GRIDPACK_INSTALL_DIR}

ENV boost_dir=${GP_EXT_DEPS}/boost-${boost_version} \
    ga_dir=${GP_EXT_DEPS}/ga-${ga_version} \
    petsc_dir=${GP_EXT_DEPS}/petsc

ENV boost_gp_dir=${boost_dir}/install_for_gridpack \
    ga_gp_dir=${ga_dir}/install_for_gridpack \
    petsc_gp_dir=${petsc_dir}/install_for_gridpack

ENV PETSC_DIR=${petsc_dir} PETSC_ARCH=build-dir

ENV LD_LIBRARY_PATH=${boost_gp_dir}/lib:${ga_gp_dir}/lib:${petsc_gp_dir}/lib
ENV DYLD_LIBRARY_PATH=${LD_LIBRARY_PATH}

# Install required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends cmake make wget tzdata git gfortran build-essential pkg-config \
    python3 python3-pip python3-venv python3-dev python-is-python3 \
    openmpi-bin openmpi-common openmpi-doc libopenmpi-dev && \
    apt-get clean

# cuDSS for the GPU batch path, only in GPU builds. NVIDIA's CUDA images
# carry NVIDIA's package repository; the package also installs a CMake
# package exporting the cudss target, which the build finds on its own.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    if [ "${GRIDPACK_ENABLE_GPU_BATCH}" != "OFF" ]; then \
      apt-get update && \
      apt-get install -y --no-install-recommends ${CUDSS_APT_PACKAGE} libmsgsl-dev; \
    fi

# Apache Arrow's Parquet library for outputFormat=parquet, only if requested.
# Apache's repository package adds its signed apt source for this release.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    if [ "${GRIDPACK_ENABLE_PARQUET}" = "ON" ]; then \
      apt-get update && \
      apt-get install -y --no-install-recommends ca-certificates lsb-release && \
      release="$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb" && \
      wget -q -O /tmp/apache-arrow.deb "https://packages.apache.org/artifactory/arrow/${release}" || \
        { echo "Apache publishes no Arrow packages for this base image (${release})"; exit 1; } && \
      apt-get install -y --no-install-recommends /tmp/apache-arrow.deb && \
      rm /tmp/apache-arrow.deb && \
      apt-get update && \
      apt-get install -y --no-install-recommends libparquet-dev; \
    fi

# Compile/Install Boost
WORKDIR ${GP_EXT_DEPS}
RUN wget "https://github.com/boostorg/boost/releases/download/boost-${boost_version}/boost-${boost_version}.tar.gz"
RUN tar -xf "boost-${boost_version}.tar.gz"
WORKDIR ${boost_dir}
RUN ./bootstrap.sh --prefix=install_for_gridpack --with-libraries=mpi,serialization,random,filesystem,system
RUN echo 'using mpi : mpicxx ; ' >> project-config.jam
RUN ./b2 -a -d+2 link="shared" stage
RUN ./b2 -a -d+2 link="shared" install

# Compile/Install GA (Global Arrays)
WORKDIR ${GP_EXT_DEPS}
RUN wget "https://github.com/GlobalArrays/ga/releases/download/v${ga_version}/ga-${ga_version}.tar.gz"
RUN tar -xf "ga-${ga_version}.tar.gz"
WORKDIR ${ga_dir}
RUN ./configure --with-mpi-ts --disable-f77 \
    --without-blas --without-lapack --without-scalapack \
    --enable-cxx --enable-i4 \
    --prefix=${ga_gp_dir} \
    CFLAGS="-Wno-implicit-function-declaration -Wno-incompatible-pointer-types -Wno-old-style-definition" \
    CXXFLAGS="-Wno-incompatible-pointer-types" \
    --enable-shared=yes --enable-static=no
RUN make -j 10 install
WORKDIR ${GP_EXT_DEPS}

# Compile/Install PETSc
RUN git clone https://gitlab.com/petsc/petsc.git
WORKDIR ${petsc_dir}
RUN git checkout "tags/v${petsc_version}" -b "v${petsc_version}"
RUN ./configure \
    --prefix=${PWD}/install_for_gridpack \
    --scalar-type=real  \
    --with-fortran-bindings=0 \
    --download-superlu_dist \
    --download-metis \
    --download-parmetis \
    --download-suitesparse \
    --download-f2cblaslapack \
    --download-scalapack \
    --download-mumps \
    --download-cmake=0 \
    --with-sowing=0 \
    --with-debugging=0 \
    --with-shared-libraries=1
RUN make all
RUN make install
#RUN make PETSC_DIR=${petsc_gp_dir} PETSC_ARCH="" check

# Copy in GridPACK source code from repository
COPY README.md .gitignore .gitmodules ${GRIDPACK_ROOT_DIR}/
COPY .git ${GRIDPACK_ROOT_DIR}/.git
COPY docs ${GRIDPACK_ROOT_DIR}/docs
COPY python ${GRIDPACK_ROOT_DIR}/python
COPY src ${GRIDPACK_ROOT_DIR}/src

# Build GridPACK
WORKDIR ${GRIDPACK_BUILD_DIR}
RUN cmake -Wdev -D GA_DIR:STRING=${ga_gp_dir} \
    -D Boost_ROOT:STRING=${boost_gp_dir} \
    -D Boost_DIR:string=${boost_gp_dir}/lib/cmake/Boost-${boost_version} \
    -D PETSC_DIR:PATH=${petsc_gp_dir} \
    -D MPI_CXX_COMPILER:STRING='mpicxx' \
    -D MPI_C_COMPILER:STRING='mpicc' \
    -D MPIEXEC:STRING='mpiexec' \
    -D MPIEXEC_MAX_NUMPROCS:STRING=2 \
    -D GRIDPACK_TEST_TIMEOUT:STRING=120 \
    -D ENABLE_ENVIRONMENT_FROM_COMM:BOOL=YES \
    -D CMAKE_INSTALL_PREFIX:PATH=${GRIDPACK_INSTALL_DIR} \
    -D CMAKE_BUILD_TYPE:STRING=${GRIDPACK_BUILD_TYPE} \
    -D BUILD_SHARED_LIBS=true \
    -D CMAKE_CXX_FLAGS_DEBUG:STRING="-D_GLIBCXX_NO_ASSERTIONS" \
    -D GRIDPACK_ENABLE_GPU_BATCH:STRING=${GRIDPACK_ENABLE_GPU_BATCH} \
    -D GRIDPACK_ENABLE_PARQUET:STRING=${GRIDPACK_ENABLE_PARQUET} \
    -D CMAKE_CUDA_ARCHITECTURES:STRING=${CUDA_ARCHITECTURES} \
    ..
RUN make install

# Install Python bindings
WORKDIR ${GRIDPACK_ROOT_DIR}
RUN git submodule update --init
RUN pip config --global set global.break-system-packages true
WORKDIR ${GRIDPACK_ROOT_DIR}/python
RUN pip install --upgrade --prefix=${GRIDPACK_INSTALL_DIR} .

# Configure Python module search path using .pth file (no environment variables needed)
# Python automatically reads .pth files from its site-packages directories
RUN pyvnum=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")') && \
    system_site_packages=$(python3 -c 'import site; print(site.getsitepackages()[0])') && \
    echo "${GRIDPACK_INSTALL_DIR}/lib/python${pyvnum}/site-packages" > ${system_site_packages}/gridpack.pth && \
    echo "${GRIDPACK_INSTALL_DIR}/local/lib/python${pyvnum}/dist-packages" >> ${system_site_packages}/gridpack.pth && \
    echo "Configured Python ${pyvnum} module search path via ${system_site_packages}/gridpack.pth"

WORKDIR ${GRIDPACK_ROOT_DIR}/workspace
ENV PATH=${GRIDPACK_INSTALL_DIR}/bin:${GRIDPACK_INSTALL_DIR}/local/bin:${PATH}

# Default command: a shell, as in GridPACK's documented usage
# (docker run ... bash). No ENTRYPOINT, so "docker run ... ca.x
# configuration.xml" works the same way.
CMD ["/bin/bash"]
