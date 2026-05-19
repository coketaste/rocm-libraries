# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

set(hiptensornet_sample_sources
    matmul_sample.cpp
    portable_demo.cpp
)

foreach(src IN LISTS hiptensornet_sample_sources)
    get_filename_component(name ${src} NAME_WE)
    add_executable(hiptensornet_${name} ${src})
    target_link_libraries(hiptensornet_${name} PRIVATE hiptensornet)
    if(TARGET hip::host)
        target_link_libraries(hiptensornet_${name} PRIVATE hip::host)
    endif()
endforeach()
