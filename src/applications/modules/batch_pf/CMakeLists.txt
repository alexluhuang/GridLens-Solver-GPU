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
# installation
# -------------------------------------------------------------
install(FILES
  include/gridpack/batchpf/batchpf_plugin.h
  include/gridpack/batchpf/batchpf_backend.h
  DESTINATION include/gridpack/batchpf
)
