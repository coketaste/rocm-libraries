# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

set(rocdensitymat_test_sources
  test_smoke.cpp
  test_handle.cpp
  test_workspace.cpp
  test_state.cpp
  test_elementary.cpp
  test_operator_term.cpp
  test_operator.cpp
  test_apply_pure.cpp
  test_apply_mixed.cpp
  test_compute_action.cpp
  test_expectation.cpp
  test_properties.cpp
  test_rk4.cpp
  test_not_supported.cpp
)

add_executable(rocdensitymat-test ${rocdensitymat_test_sources})

target_compile_features(rocdensitymat-test PRIVATE cxx_std_17)
target_link_libraries(rocdensitymat-test PRIVATE roc::rocdensitymat)
if(TARGET GTest::gtest_main)
  target_link_libraries(rocdensitymat-test PRIVATE GTest::gtest_main)
elseif(GTEST_FOUND)
  target_include_directories(rocdensitymat-test PRIVATE ${GTEST_INCLUDE_DIRS})
  target_link_libraries(rocdensitymat-test PRIVATE ${GTEST_BOTH_LIBRARIES})
endif()

include(GoogleTest)
if(COMMAND gtest_discover_tests)
  gtest_discover_tests(rocdensitymat-test)
else()
  add_test(NAME rocdensitymat-test COMMAND rocdensitymat-test)
endif()

rocm_install(TARGETS rocdensitymat-test COMPONENT tests)
