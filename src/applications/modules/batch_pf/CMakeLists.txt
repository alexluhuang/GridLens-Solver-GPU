#
#     Copyright (c) 2013 Battelle Memorial Institute
#     Licensed under modified BSD License. A copy of this license can be
#     found in the LICENSE file in the top level directory of this
#     distribution.
#
# -------------------------------------------------------------
# GPU batch power flow for contingency analysis
# -------------------------------------------------------------
# ca.x links only the CUDA-free host library defined here. The GPU code is
# built as plugins that ca.x loads at run time, so CUDA is enabled in this
# directory only, and only when a CUDA compiler is found (architecture guide
# sections 7.3.2 and 8.19, ADR-16, ADR-21). Without a CUDA toolkit the build
# is exactly today's GridPACK plus a small host library that falls back to
# the stock CPU path.
#
# Targets:
#   gridpack::batchpf_interface  C plugin interface headers (I-10, I-11)
#   gridpack::batchpf_host       host side, linked into ca.x (no CUDA)
#   gridpack_batchpf_core        core GPU plugin (MODULE)
#   gridpack_batchpf_cudss       cuDSS backend plugin (MODULE)
# -------------------------------------------------------------

include(CheckLanguage)

# GridPACK's top-level CMakeLists.txt passes GCC warning flags (-pedantic,
# -Wall, ...) with add_definitions(), which applies them to every language
# in every directory. nvcc does not accept them and -pedantic makes the host
# compiler warn about every line marker nvcc emits. The existing lines stay
# as they are (exception EX-CM-01); this directory removes the same flags
# and sets its own through gridpack_batchpf_warnings (exception EX-CM-04).
if (CMAKE_COMPILER_IS_GNUCXX)
  remove_definitions(
    -pedantic
    -Wno-write-strings
    -Wno-long-long
    -Wno-sign-compare
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-maybe-uninitialized
    -Wall
  )
endif()

set(GRIDPACK_ENABLE_GPU_BATCH "AUTO" CACHE STRING
  "Build the GPU batch plugins: AUTO (if a CUDA compiler is found), ON, OFF")
set_property(CACHE GRIDPACK_ENABLE_GPU_BATCH PROPERTY STRINGS AUTO ON OFF)

# Plugins are installed next to each other; the host loader also looks here
# relative to the installed ca.x.
set(GRIDPACK_BATCHPF_PLUGIN_SUBDIR "lib/gridpack/plugins")

# -------------------------------------------------------------
# Plugin interface headers
# -------------------------------------------------------------
add_library(gridpack_batchpf_interface INTERFACE)
add_library(gridpack::batchpf_interface ALIAS gridpack_batchpf_interface)
target_include_directories(gridpack_batchpf_interface INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# -------------------------------------------------------------
# Warning settings shared by every new target, in one place. New warnings
# are treated as errors by review, never with -Werror.
# -------------------------------------------------------------
add_library(gridpack_batchpf_warnings INTERFACE)
target_compile_options(gridpack_batchpf_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

# -------------------------------------------------------------
# GPU plugins: only with a CUDA compiler
# -------------------------------------------------------------
set(GRIDPACK_BATCHPF_HAVE_CUDA OFF)
if (NOT GRIDPACK_ENABLE_GPU_BATCH STREQUAL "OFF")
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    # The "all" architecture list needs CMake 3.23 (constraint T-11).
    if (CMAKE_VERSION VERSION_LESS 3.23)
      message(FATAL_ERROR
        "The GPU batch plugins need CMake 3.23 or newer (found ${CMAKE_VERSION}). "
        "Set GRIDPACK_ENABLE_GPU_BATCH=OFF to build without them.")
    endif()
    # Native code for every supported GPU plus PTX for newer ones, unless
    # the user chose a list (CMAKE_CUDA_ARCHITECTURES or CUDAARCHS).
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
      set(CMAKE_CUDA_ARCHITECTURES all)
    endif()
    enable_language(CUDA)
    set(GRIDPACK_BATCHPF_HAVE_CUDA ON)
  elseif (GRIDPACK_ENABLE_GPU_BATCH STREQUAL "ON")
    message(FATAL_ERROR
      "GRIDPACK_ENABLE_GPU_BATCH=ON but no CUDA compiler was found. "
      "Install the CUDA toolkit or use AUTO/OFF.")
  endif()
endif()
if (GRIDPACK_BATCHPF_HAVE_CUDA)
  message(STATUS "GPU batch plugins: enabled (CUDA ${CMAKE_CUDA_COMPILER_VERSION}, "
    "architectures ${CMAKE_CUDA_ARCHITECTURES})")
else()
  message(STATUS "GPU batch plugins: not built; ca.x uses the CPU path")
endif()

# -------------------------------------------------------------
# Core GPU plugin: planner (B6), engine (B8), device platform layer (B9-D),
# built-in backends (Algorithm 2, CPU reference KLU)
# -------------------------------------------------------------
if (GRIDPACK_BATCHPF_HAVE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(Threads REQUIRED)
  # SuiteSparse as installed by GridPACK's PETSc build exports CMake
  # targets; its config files find their own dependencies (AMD, COLAMD,
  # BTF, SuiteSparse_config) through the prefix path, so add PETSc's
  # prefix in this directory only.
  list(PREPEND CMAKE_PREFIX_PATH ${PETSC_DIR})
  find_package(KLU CONFIG REQUIRED)

  add_library(gridpack_batchpf_core MODULE
    core/backend_alg2.cu
    core/backend_cpu.cpp
    core/backend_plugin.cpp
    core/engine.cu
    core/engine_setup.cu
    core/engine_control.cpp
    core/model.cpp
    core/planner.cpp
    core/platform.cu
    core/plugin_api.cpp
    core/session.cpp
  )
  target_compile_features(gridpack_batchpf_core PRIVATE cxx_std_17 cuda_std_17)
  target_link_libraries(gridpack_batchpf_core PRIVATE
    gridpack::batchpf_interface
    gridpack_batchpf_warnings
    CUDA::cudart_static
    SuiteSparse::KLU
    Threads::Threads
    ${CMAKE_DL_LIBS}
  )
  # Static CUDA runtime (CUDA Best Practices Guide 17.4); only the plugin
  # entry point is visible (16.4.1.4)
  set_target_properties(gridpack_batchpf_core PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${GRIDPACK_BATCHPF_PLUGIN_SUBDIR}
  )
  install(TARGETS gridpack_batchpf_core
    LIBRARY DESTINATION ${GRIDPACK_BATCHPF_PLUGIN_SUBDIR})
endif()

# -------------------------------------------------------------
# installation
# -------------------------------------------------------------
install(FILES
  include/gridpack/batchpf/batchpf_plugin.h
  include/gridpack/batchpf/batchpf_backend.h
  DESTINATION include/gridpack/batchpf
)
