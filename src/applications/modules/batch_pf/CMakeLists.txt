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
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--resource-usage>
)

# -------------------------------------------------------------
# Host library linked into ca.x: settings resolver (B14), accelerator
# loader (B13), host platform layer (B9-H) and the GridPACK adapter layer
# (classifier, export, batch scheduling, state injection). No CUDA.
# -------------------------------------------------------------
add_library(gridpack_batchpf_host STATIC
  host/accelerator.cpp
  host/batch_path.cpp
  host/batch_run.cpp
  host/classifier.cpp
  host/host_platform.cpp
  host/reconcile.cpp
  host/settings.cpp
)
add_library(gridpack::batchpf_host ALIAS gridpack_batchpf_host)
target_compile_features(gridpack_batchpf_host PUBLIC cxx_std_17)
# GridPACK's own headers are treated as system headers here so that the
# warning settings above apply to the new code only (risk R-24)
target_include_directories(gridpack_batchpf_host SYSTEM PRIVATE
  ${GridPACK_SOURCE_DIR} ${GA_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
# GridPACK exposes its dependencies through variables, not targets, so the
# host library links what ca.x links (exception EX-CM-02)
target_link_libraries(gridpack_batchpf_host
  PUBLIC
    gridpack::batchpf_interface
  PRIVATE
    gridpack_batchpf_warnings
    gridpack_powerflow_module
    gridpack_configuration
    gridpack_parallel
    gridpack_timer
    ${GA_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${CMAKE_DL_LIBS}
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
  find_package(Microsoft.GSL 4 CONFIG REQUIRED)

  # The plugin's code is an object library so the unit tests can link the
  # same objects; the plugin adds its C entry points.
  add_library(gridpack_batchpf_core_objects OBJECT
    core/backend_alg2.cu
    core/backend_cpu.cpp
    core/backend_plugin.cpp
    core/engine.cu
    core/engine_setup.cu
    core/engine_control.cpp
    core/model.cpp
    core/planner.cpp
    core/platform.cu
    core/session.cpp
  )
  target_compile_features(gridpack_batchpf_core_objects PUBLIC cxx_std_17 cuda_std_17)
  target_link_libraries(gridpack_batchpf_core_objects
    PUBLIC
      gridpack::batchpf_interface
      CUDA::cudart_static
      SuiteSparse::KLU
      Microsoft.GSL::GSL
      Threads::Threads
      ${CMAKE_DL_LIBS}
    PRIVATE
      gridpack_batchpf_warnings
  )
  # Static CUDA runtime (CUDA Best Practices Guide 17.4); only the plugin
  # entry point is visible (16.4.1.4)
  set_target_properties(gridpack_batchpf_core_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Static
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
  add_library(gridpack_batchpf_core MODULE core/plugin_api.cpp)
  target_link_libraries(gridpack_batchpf_core PRIVATE
    gridpack_batchpf_core_objects
    gridpack_batchpf_warnings
  )
  set_target_properties(gridpack_batchpf_core PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${GRIDPACK_BATCHPF_PLUGIN_SUBDIR}
  )
  install(TARGETS gridpack_batchpf_core
    LIBRARY DESTINATION ${GRIDPACK_BATCHPF_PLUGIN_SUBDIR})

  # cuDSS backend plugin (I-11), only if cuDSS is installed. Its CMake
  # package exports the cudss target. cuDSS and the CUDA math libraries it
  # uses are shared libraries that must be present at run time; without them
  # only this backend is unavailable.
  find_package(cudss CONFIG QUIET HINTS $ENV{CUDSS_ROOT} /opt/cudss)
  if (cudss_FOUND)
    add_library(gridpack_batchpf_cudss MODULE cudss/backend_cudss.cu)
    target_compile_features(gridpack_batchpf_cudss PRIVATE cxx_std_17 cuda_std_17)
    target_link_libraries(gridpack_batchpf_cudss PRIVATE
      gridpack::batchpf_interface
      gridpack_batchpf_warnings
      cudss
      CUDA::cudart_static
      Microsoft.GSL::GSL
    )
    set_target_properties(gridpack_batchpf_cudss PROPERTIES
      CUDA_RUNTIME_LIBRARY Static
      CXX_VISIBILITY_PRESET hidden
      CUDA_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN ON
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${GRIDPACK_BATCHPF_PLUGIN_SUBDIR}
      INSTALL_RPATH "${cudss_LIBRARY_DIR}"
      BUILD_RPATH "${cudss_LIBRARY_DIR}"
    )
    install(TARGETS gridpack_batchpf_cudss
      LIBRARY DESTINATION ${GRIDPACK_BATCHPF_PLUGIN_SUBDIR})
    message(STATUS "GPU batch plugins: cuDSS backend enabled (cuDSS ${cudss_VERSION})")
  else()
    message(STATUS "GPU batch plugins: cuDSS not found; cuDSS backend not built")
  endif()
endif()

# -------------------------------------------------------------
# Tests (guide 8.14; names follow batchpf.unit.*, batchpf.parity.*,
# batchpf.platform.*). GridPACK's convention: a test passes when it prints
# "No errors detected".
# -------------------------------------------------------------
if (GRIDPACK_ENABLE_TESTS)
  function(batchpf_add_test name)
    add_test(NAME ${name} COMMAND ${ARGN})
    set_tests_properties(${name} PROPERTIES
      PASS_REGULAR_EXPRESSION "No errors detected"
      FAIL_REGULAR_EXPRESSION "failure detected"
      TIMEOUT 1800)
  endfunction()

  # Control logic: plain C++, runs without CUDA
  add_executable(batchpf_test_control test/test_control.cpp core/engine_control.cpp)
  target_compile_features(batchpf_test_control PRIVATE cxx_std_17)
  target_link_libraries(batchpf_test_control PRIVATE
    gridpack::batchpf_interface gridpack_batchpf_warnings)
  batchpf_add_test(batchpf.unit.control $<TARGET_FILE:batchpf_test_control>)

  add_executable(batchpf_test_reconcile test/test_reconcile.cpp host/reconcile.cpp)
  target_compile_features(batchpf_test_reconcile PRIVATE cxx_std_17)
  target_link_libraries(batchpf_test_reconcile PRIVATE gridpack_batchpf_warnings)
  batchpf_add_test(batchpf.unit.reconcile $<TARGET_FILE:batchpf_test_reconcile>)

  set(batchpf_data ${GRIDPACK_DATA_DIR}/raw)
  find_package(Python3 COMPONENTS Interpreter QUIET)

  if (GRIDPACK_BATCHPF_HAVE_CUDA)
    add_executable(batchpf_test_model test/test_model.cpp)
    target_link_libraries(batchpf_test_model PRIVATE
      gridpack_batchpf_core_objects gridpack_batchpf_warnings)
    batchpf_add_test(batchpf.unit.model $<TARGET_FILE:batchpf_test_model>)

    # Element functions on GPU and CPU, Algorithm 2 against KLU
    add_executable(batchpf_test_kernels test/test_kernels.cu)
    target_link_libraries(batchpf_test_kernels PRIVATE
      gridpack_batchpf_core_objects gridpack_batchpf_warnings)
    set_target_properties(batchpf_test_kernels PROPERTIES CUDA_RUNTIME_LIBRARY Static)
    batchpf_add_test(batchpf.unit.kernels $<TARGET_FILE:batchpf_test_kernels>)

    # Kernel parity with GridPACK's large Jacobian layout, and fast-path
    # classification against GridPACK's contingency routine
    add_executable(batchpf_test_gridpack test/test_gridpack.cpp host/classifier.cpp)
    target_include_directories(batchpf_test_gridpack SYSTEM PRIVATE
      ${GridPACK_SOURCE_DIR} ${GA_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
    target_link_libraries(batchpf_test_gridpack PRIVATE
      gridpack_batchpf_core_objects
      gridpack_batchpf_warnings
      gridpack_powerflow_module
      gridpack_pfmatrix_components
      gridpack_ymatrix_components
      gridpack_components
      gridpack_stream
      gridpack_partition
      gridpack_environment
      gridpack_math
      gridpack_configuration
      gridpack_timer
      gridpack_parallel
      gridpack_block_parsers
      ${PARMETIS_LIBRARY} ${METIS_LIBRARY}
      ${Boost_LIBRARIES}
      ${GA_LIBRARIES}
      ${PETSC_LIBRARIES}
      ${MPI_CXX_LIBRARIES}
    )
    # The WECC grid has remote voltage regulation (PV buses without
    # generators, which GridPACK never limits)
    foreach(grid IEEE14_ca IEEE118 240busWECC_2018_PSS_fixedshunt)
      set(dir ${CMAKE_CURRENT_BINARY_DIR}/parity_${grid})
      file(MAKE_DIRECTORY ${dir})
      configure_file(${batchpf_data}/${grid}.raw ${dir}/${grid}.raw COPYONLY)
      file(WRITE ${dir}/input.xml
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<Configuration><Powerflow>"
        "<networkConfiguration>${grid}.raw</networkConfiguration>"
        "<jacobianFormulation>large</jacobianFormulation>"
        "<maxIteration>50</maxIteration><tolerance>1.0e-6</tolerance>"
        "<LinearSolver><PETScOptions>-ksp_type preonly -pc_type lu "
        "-pc_factor_mat_solver_type klu</PETScOptions>"
        "</LinearSolver></Powerflow></Configuration>\n")
      add_test(NAME batchpf.parity.components_${grid}
        COMMAND $<TARGET_FILE:batchpf_test_gridpack> input.xml
        WORKING_DIRECTORY ${dir})
      set_tests_properties(batchpf.parity.components_${grid} PROPERTIES
        PASS_REGULAR_EXPRESSION "No errors detected"
        FAIL_REGULAR_EXPRESSION "failure detected")
    endforeach()
  endif()

  # End-to-end runs of ca.x against the same ca.x without the batch path
  if (Python3_Interpreter_FOUND)
    set(runner ${CMAKE_CURRENT_SOURCE_DIR}/test/run_ca_test.py)
    set(raw118 ${batchpf_data}/IEEE118.raw)
    set(raw14 ${batchpf_data}/IEEE14_ca.raw)
    set(work ${CMAKE_CURRENT_BINARY_DIR}/ca_tests)
    set(lu klu)   # KLU, as the plan uses for the CPU path (GridPACK builds PETSc with it)
    foreach(mode no_block disabled no_device no_plugin invalid required)
      batchpf_add_test(batchpf.platform.${mode} ${Python3_EXECUTABLE} ${runner}
        --cax $<TARGET_FILE:ca.x> --raw ${raw14} --workdir ${work}/${mode}
        --mode ${mode} --solver ${lu})
    endforeach()
    set(backends)
    if (GRIDPACK_BATCHPF_HAVE_CUDA)
      list(APPEND backends cpu_reference alg2)
      if (cudss_FOUND)
        list(APPEND backends cudss)
      endif()
    endif()
    foreach(b ${backends})
      batchpf_add_test(batchpf.parity.ca_ieee118_${b} ${Python3_EXECUTABLE} ${runner}
        --cax $<TARGET_FILE:ca.x> --raw ${raw118} --workdir ${work}/parity_${b}
        --mode parity --require-shadow-sets --backend ${b} --solver ${lu})
    endforeach()
    if (NOT GRIDPACK_BATCHPF_HAVE_CUDA)
      batchpf_add_test(batchpf.platform.cpu_only_ieee118 ${Python3_EXECUTABLE} ${runner}
        --cax $<TARGET_FILE:ca.x> --raw ${raw118} --workdir ${work}/cpu_only_118
        --mode no_block --solver ${lu})
    endif()

    if (GRIDPACK_BATCHPF_HAVE_CUDA AND MPIEXEC)
      foreach(ranks 2 4)
        batchpf_add_test(batchpf.parity.mpi_${ranks} ${Python3_EXECUTABLE} ${runner}
          --cax $<TARGET_FILE:ca.x> --raw ${raw118} --workdir ${work}/mpi_${ranks}
          --mode parity --require-shadow-sets --backend alg2 --solver ${lu} --ranks ${ranks}
          --mpiexec ${MPIEXEC} --gpu-setting "<batchSize>32</batchSize>")
      endforeach()
      batchpf_add_test(batchpf.parity.mpi_two_accelerators ${Python3_EXECUTABLE} ${runner}
        --cax $<TARGET_FILE:ca.x> --raw ${raw118} --workdir ${work}/mpi_two_accelerators
        --mode parity --require-shadow-sets --backend alg2 --solver ${lu} --ranks 4 --mpiexec ${MPIEXEC}
        --accelerator-ranks 0,1 --gpu-setting "<batchSize>32</batchSize>")
    endif()
    if (GRIDPACK_BATCHPF_HAVE_CUDA)
      batchpf_add_test(batchpf.platform.validated_cap ${Python3_EXECUTABLE} ${runner}
        --cax $<TARGET_FILE:ca.x> --raw ${raw118} --workdir ${work}/validated_cap
        --mode parity --require-shadow-sets --backend alg2 --solver ${lu}
        --gpu-setting "<batchSize>32</batchSize>"
        --gpu-setting "<maxValidatedBatch>8</maxValidatedBatch>" --expect-batch-at-most 8)
      batchpf_add_test(batchpf.platform.memory_cap ${Python3_EXECUTABLE} ${runner}
        --cax $<TARGET_FILE:ca.x> --raw ${raw118} --workdir ${work}/memory_cap
        --mode parity --require-shadow-sets --backend alg2 --solver ${lu}
        --gpu-setting "<batchSize>64</batchSize>"
        --gpu-setting "<maxMemoryGB>0.001</maxMemoryGB>" --expect-batch-at-most 32)
    endif()

    # External RAW files are not redistributed. Point this list at locally
    # available validation cases (at most 10k buses, per the handoff).
    set(GRIDPACK_BATCHPF_TEST_RAW_FILES "" CACHE STRING
      "Optional RAW files for full N-1 parity tests; semicolon-separated")
    foreach(raw IN LISTS GRIDPACK_BATCHPF_TEST_RAW_FILES)
      if (NOT EXISTS "${raw}")
        message(FATAL_ERROR "Batch parity RAW file does not exist: ${raw}")
      endif()
      get_filename_component(case_name "${raw}" NAME_WE)
      foreach(b IN LISTS backends)
        if (b STREQUAL "cpu_reference")
          continue()
        endif()
        batchpf_add_test(batchpf.parity.external_${case_name}_${b}
          ${Python3_EXECUTABLE} ${runner} --cax $<TARGET_FILE:ca.x> --raw ${raw}
          --workdir ${work}/${case_name}_${b} --mode parity --require-shadow-sets --backend ${b}
          --solver ${lu} --ranks 8 --mpiexec ${MPIEXEC})
        # Full csv_delta studies can write tens of GB. Allow comparison
        # time as well as two solves; each solver still has its own timeout.
        set_tests_properties(batchpf.parity.external_${case_name}_${b} PROPERTIES TIMEOUT 7200)
      endforeach()
    endforeach()
  endif()
endif()

# -------------------------------------------------------------
# installation
# -------------------------------------------------------------
install(FILES
  include/gridpack/batchpf/batchpf_plugin.h
  include/gridpack/batchpf/batchpf_backend.h
  include/gridpack/batchpf/index.hpp
  DESTINATION include/gridpack/batchpf
)
# The contingency-analysis driver includes these and links the host
# library, also when it is built from the installed example
install(FILES
  host/batch_path.hpp
  host/reconcile.hpp
  DESTINATION include/gridpack/applications/modules/batch_pf/host
)
install(TARGETS gridpack_batchpf_host ARCHIVE DESTINATION lib)
