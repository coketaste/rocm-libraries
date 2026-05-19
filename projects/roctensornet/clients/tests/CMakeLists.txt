# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

find_package(GTest QUIET)
if(NOT TARGET GTest::gtest)
    message(WARNING "roctensornet: GoogleTest not found; tests disabled")
    return()
endif()
include(GoogleTest)

set(roctensornet_test_sources
    test_smoke.cpp
    test_network_descriptor.cpp
    test_optimizer.cpp
    test_workspace.cpp
    test_contraction_two_tensor.cpp
    test_contraction_three_tensor.cpp
    test_slicing.cpp
    test_autotune.cpp
    test_tensor_svd.cpp
    test_tensor_qr.cpp
    test_gradient.cpp
    test_network_state.cpp
    test_marginal.cpp
    test_sampler.cpp
    test_expectation.cpp
    test_accessor.cpp
    test_device_matrix.cpp
    test_workspace_query.cpp
    test_mem_handler.cpp
)

foreach(src IN LISTS roctensornet_test_sources)
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE roctensornet GTest::gtest GTest::gtest_main)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(TARGET hip::host)
        target_link_libraries(${name} PRIVATE hip::host)
    endif()
    gtest_discover_tests(${name} DISCOVERY_TIMEOUT 60)
endforeach()
