# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

find_package(GTest QUIET)
if(NOT TARGET GTest::gtest)
    message(WARNING "hiptensornet: GoogleTest not found; tests disabled")
    return()
endif()
include(GoogleTest)

set(hiptensornet_test_sources
    test_smoke.cpp
    test_network_descriptor.cpp
    test_contraction.cpp
    test_state.cpp
    test_status_strings.cpp
)

foreach(src IN LISTS hiptensornet_test_sources)
    get_filename_component(name ${src} NAME_WE)
    add_executable(hiptensornet_${name} ${src})
    target_link_libraries(hiptensornet_${name} PRIVATE hiptensornet GTest::gtest GTest::gtest_main)
    target_include_directories(hiptensornet_${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(TARGET hip::host)
        target_link_libraries(hiptensornet_${name} PRIVATE hip::host)
    endif()
    gtest_discover_tests(hiptensornet_${name} DISCOVERY_TIMEOUT 60)
endforeach()
