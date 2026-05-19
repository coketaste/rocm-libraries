# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

add_executable(cudaq_pattern_demo cudaq_pattern_demo.cpp)
target_link_libraries(cudaq_pattern_demo PRIVATE roctensornet)
if(TARGET hip::host)
    target_link_libraries(cudaq_pattern_demo PRIVATE hip::host)
endif()
