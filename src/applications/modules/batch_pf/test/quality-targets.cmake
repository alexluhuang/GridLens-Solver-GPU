# This file is included only by quality.cmake. DEFER waits until the targets
# exist; existing GridPACK targets keep their established analyzer settings.
function(batchpf_quality_targets)
  foreach(target IN ITEMS gridpack_batchpf_host gridpack_batchpf_core_objects
      gridpack_batchpf_core batchpf_test_control batchpf_test_gridpack
      batchpf_test_model batchpf_test_reconcile)
    if(TARGET "${target}")
      set_property(TARGET "${target}" PROPERTY CXX_CLANG_TIDY
        "${BATCHPF_CLANG_TIDY};-checks=-*,cppcoreguidelines-*;-header-filter=.*batch_pf/.*")
      set_property(TARGET "${target}" PROPERTY CXX_INCLUDE_WHAT_YOU_USE
        "${BATCHPF_IWYU}")
    endif()
  endforeach()
endfunction()
cmake_language(DEFER CALL batchpf_quality_targets)
