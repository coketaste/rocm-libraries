# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

foreach(name tensor_contraction mps_sample ghz_sample)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE roctensornet)
    if(TARGET hip::host)
        target_link_libraries(${name} PRIVATE hip::host)
    endif()
endforeach()

add_subdirectory(cudaq_pattern_demo)
