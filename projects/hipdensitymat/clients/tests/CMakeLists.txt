# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

add_executable(hipdensitymat-test
  test_smoke.cpp
  test_conformance.cpp
)
target_compile_features(hipdensitymat-test PRIVATE cxx_std_17)
target_link_libraries(hipdensitymat-test PRIVATE roc::hipdensitymat hip::host)

if(TARGET GTest::gtest_main)
  target_link_libraries(hipdensitymat-test PRIVATE GTest::gtest_main)
elseif(GTEST_FOUND)
  target_include_directories(hipdensitymat-test PRIVATE ${GTEST_INCLUDE_DIRS})
  target_link_libraries(hipdensitymat-test PRIVATE ${GTEST_BOTH_LIBRARIES})
endif()

include(GoogleTest)
if(COMMAND gtest_discover_tests)
  gtest_discover_tests(hipdensitymat-test)
else()
  add_test(NAME hipdensitymat-test COMMAND hipdensitymat-test)
endif()
