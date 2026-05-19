# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

if(BUILD_CLIENTS_TESTS)
  if(NOT GTEST_FOUND)
    find_package(GTest QUIET)
  endif()
  if(NOT GTEST_FOUND AND NOT TARGET GTest::gtest)
    include(FetchContent)
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        release-1.12.1)
    FetchContent_MakeAvailable(googletest)
  endif()
  add_subdirectory(tests)
endif()

if(BUILD_CLIENTS_SAMPLES)
  add_subdirectory(samples)
endif()
