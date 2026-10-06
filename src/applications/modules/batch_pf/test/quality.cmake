# Run with ctest -S quality.cmake -DCTEST_SOURCE_DIRECTORY=.../src
# -DCTEST_BINARY_DIRECTORY=... -DCTEST_CONFIGURE_OPTIONS="<dependency options>".
# Keep analyzer choices in this CI entry point, rather than project targets.
cmake_minimum_required(VERSION 3.23)
if(NOT DEFINED CTEST_SOURCE_DIRECTORY OR NOT DEFINED CTEST_BINARY_DIRECTORY)
  message(FATAL_ERROR "Set CTEST_SOURCE_DIRECTORY and CTEST_BINARY_DIRECTORY")
endif()
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_SITE "batchpf-quality")
set(CTEST_BUILD_NAME "cppcoreguidelines")
set(CTEST_TEST_TIMEOUT 7200)
if(NOT DEFINED BATCHPF_QUALITY_JOBS)
  set(BATCHPF_QUALITY_JOBS 6)
endif()
find_program(BATCHPF_CLANG_TIDY NAMES clang-tidy-18 clang-tidy REQUIRED)
find_program(BATCHPF_IWYU NAMES include-what-you-use iwyu REQUIRED)
find_program(BATCHPF_CPPCHECK NAMES cppcheck REQUIRED)
find_program(BATCHPF_RUN_CLANG_TIDY NAMES run-clang-tidy-18 run-clang-tidy REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(BATCHPF_QUALITY_DIR "${CTEST_SOURCE_DIRECTORY}/applications/modules/batch_pf/test")
ctest_start(Experimental)
file(STRINGS "${CTEST_BINARY_DIRECTORY}/Testing/TAG" build_tag LIMIT_COUNT 1)
set(build_log "${CTEST_BINARY_DIRECTORY}/Testing/Temporary/LastBuild_${build_tag}.log")
ctest_configure(OPTIONS
  "${CTEST_CONFIGURE_OPTIONS};-DCMAKE_BUILD_TYPE=Release;-DCMAKE_EXPORT_COMPILE_COMMANDS=ON;-DCMAKE_PROJECT_INCLUDE=${BATCHPF_QUALITY_DIR}/quality-targets.cmake;-DBATCHPF_CLANG_TIDY=${BATCHPF_CLANG_TIDY};-DBATCHPF_IWYU=${BATCHPF_IWYU}"
  RETURN_VALUE configure_result)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR "Quality configuration failed")
endif()
ctest_build(PARALLEL_LEVEL "${BATCHPF_QUALITY_JOBS}" RETURN_VALUE build_result)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR "Quality build failed; inspect Testing/Temporary/LastBuild*.log")
endif()
execute_process(
  COMMAND "${BATCHPF_RUN_CLANG_TIDY}" -p "${CTEST_BINARY_DIRECTORY}"
    -j "${BATCHPF_QUALITY_JOBS}"
    "-checks=-*,cppcoreguidelines-*" "-header-filter=.*batch_pf/.*"
    ".*/batch_pf/.*\\.cpp$"
  OUTPUT_FILE "${CTEST_BINARY_DIRECTORY}/clang-quality.log"
  ERROR_FILE "${CTEST_BINARY_DIRECTORY}/clang-quality-errors.log"
  RESULT_VARIABLE clang_result)
if(NOT clang_result EQUAL 0)
  message(FATAL_ERROR "clang-tidy did not complete; inspect clang-quality*.log")
endif()
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${BATCHPF_QUALITY_DIR}/audit_quality.py"
    --write-line-filter --source-root "${CTEST_SOURCE_DIRECTORY}/.."
    --output "${CTEST_BINARY_DIRECTORY}/adapter-lines.json"
  RESULT_VARIABLE scope_result)
if(NOT scope_result EQUAL 0)
  message(FATAL_ERROR "Cannot determine changed adapter lines; use a full git checkout")
endif()
file(READ "${CTEST_BINARY_DIRECTORY}/adapter-lines.json" adapter_lines)
execute_process(
  COMMAND "${BATCHPF_RUN_CLANG_TIDY}" -p "${CTEST_BINARY_DIRECTORY}"
    -j "${BATCHPF_QUALITY_JOBS}" "-checks=-*,cppcoreguidelines-*,clang-analyzer-*"
    "-header-filter=.*(pf_matrix|powerflow|contingency_analysis).*"
    "-line-filter=${adapter_lines}"
    ".*/(pf_components|pf_app_module|pf_factory_module|ca_driver)\\.cpp$"
  OUTPUT_FILE "${CTEST_BINARY_DIRECTORY}/clang-adapters.log"
  ERROR_FILE "${CTEST_BINARY_DIRECTORY}/clang-adapters-errors.log"
  RESULT_VARIABLE adapters_result)
if(NOT adapters_result EQUAL 0)
  message(FATAL_ERROR "Adapter analysis failed; inspect clang-adapters*.log")
endif()
# cppcheck 2.13 cannot preprocess Boost MPL in the legacy adapters. Those
# changed lines use Clang's separate path-sensitive analyzer above instead;
# do not suppress the cppcheck preprocessing failure and call it coverage.
execute_process(
  COMMAND "${BATCHPF_CPPCHECK}"
    "--project=${CTEST_BINARY_DIRECTORY}/compile_commands.json"
    "--file-filter=*/batch_pf/*" --enable=warning,style,performance,portability
    --inline-suppr --xml --xml-version=2
  OUTPUT_FILE "${CTEST_BINARY_DIRECTORY}/cppcheck-quality.log"
  ERROR_FILE "${CTEST_BINARY_DIRECTORY}/cppcheck-quality.xml"
  RESULT_VARIABLE cppcheck_result)
if(NOT cppcheck_result EQUAL 0)
  message(FATAL_ERROR "cppcheck did not complete")
endif()
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${BATCHPF_QUALITY_DIR}/audit_quality.py"
    --clang-log "${CTEST_BINARY_DIRECTORY}/clang-quality.log"
    --clang-log "${CTEST_BINARY_DIRECTORY}/clang-adapters.log"
    --build-log "${build_log}"
    --cppcheck-xml "${CTEST_BINARY_DIRECTORY}/cppcheck-quality.xml"
    --adapter-lines "${CTEST_BINARY_DIRECTORY}/adapter-lines.json"
    --output "${CTEST_BINARY_DIRECTORY}/quality-findings.jsonl"
    --baseline "${CTEST_SOURCE_DIRECTORY}/../docs/gpu_n1/standards-findings.jsonl"
  RESULT_VARIABLE audit_result)
if(NOT audit_result EQUAL 0)
  message(FATAL_ERROR "Analyzer findings increased; review quality-findings.jsonl")
endif()
# CUDA compiler resource reports are produced by --resource-usage. Device
# analysis and Compute Sanitizer run separately on a GPU worker.
ctest_test(INCLUDE "^batchpf\\.(unit|parity|platform)\\."
  EXCLUDE "external_" RETURN_VALUE test_result)
if(NOT test_result EQUAL 0)
  message(FATAL_ERROR "Batch tests failed")
endif()
