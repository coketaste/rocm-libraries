# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

add_executable(hipstatevec-test
  test_smoke.cpp
  test_conformance.cpp
)
target_compile_features(hipstatevec-test PRIVATE cxx_std_17)
target_link_libraries(hipstatevec-test PRIVATE roc::hipstatevec)
if(HIPSTATEVEC_ENABLE_HIP)
  target_compile_definitions(hipstatevec-test PRIVATE HIPSTATEVEC_ENABLE_HIP=1)
  target_link_libraries(hipstatevec-test PRIVATE hip::host)
endif()
if(HIPSTATEVEC_ENABLE_CUDA)
  target_compile_definitions(hipstatevec-test PRIVATE HIPSTATEVEC_ENABLE_CUDA=1)
  target_link_libraries(hipstatevec-test PRIVATE CUDA::cudart)
endif()

if(TARGET GTest::gtest_main)
  target_link_libraries(hipstatevec-test PRIVATE GTest::gtest_main)
elseif(GTEST_FOUND)
  target_include_directories(hipstatevec-test PRIVATE ${GTEST_INCLUDE_DIRS})
  target_link_libraries(hipstatevec-test PRIVATE ${GTEST_BOTH_LIBRARIES})
endif()

include(GoogleTest)
if(COMMAND gtest_discover_tests)
  gtest_discover_tests(hipstatevec-test)
else()
  add_test(NAME hipstatevec-test COMMAND hipstatevec-test)
endif()
