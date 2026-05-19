# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

set(rocstatevec_test_sources
  test_smoke.cpp
  test_handle.cpp
  test_init_and_z.cpp
  test_apply.cpp
  test_sampler.cpp
  test_expectation.cpp
  test_permute_and_diagonal.cpp
  test_accessor.cpp
  test_test_matrix_type.cpp
  test_device_matrix.cpp
  test_workspace_query.cpp
  test_mem_handler.cpp
)

add_executable(rocstatevec-test ${rocstatevec_test_sources})

target_compile_features(rocstatevec-test PRIVATE cxx_std_17)
target_link_libraries(rocstatevec-test PRIVATE roc::rocstatevec)
if(TARGET GTest::gtest_main)
  target_link_libraries(rocstatevec-test PRIVATE GTest::gtest_main)
elseif(GTEST_FOUND)
  target_include_directories(rocstatevec-test PRIVATE ${GTEST_INCLUDE_DIRS})
  target_link_libraries(rocstatevec-test PRIVATE ${GTEST_BOTH_LIBRARIES})
endif()

include(GoogleTest)
if(COMMAND gtest_discover_tests)
  gtest_discover_tests(rocstatevec-test)
else()
  add_test(NAME rocstatevec-test COMMAND rocstatevec-test)
endif()

rocm_install(TARGETS rocstatevec-test COMPONENT tests)
