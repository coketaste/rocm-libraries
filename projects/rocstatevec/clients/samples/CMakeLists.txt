# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

add_executable(rocstatevec-bell-pair bell_pair.cpp)
target_link_libraries(rocstatevec-bell-pair PRIVATE roc::rocstatevec)

add_executable(rocstatevec-ghz-sample ghz_sample.cpp)
target_link_libraries(rocstatevec-ghz-sample PRIVATE roc::rocstatevec)

add_subdirectory(cudaq_pattern_demo)
