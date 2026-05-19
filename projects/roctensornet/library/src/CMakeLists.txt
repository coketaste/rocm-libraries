# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

set(roctensornet_sources
    handle.cpp
    logger.cpp
    error.cpp
    version.cpp
    network_descriptor.cpp
    tensor_descriptor.cpp
    optimizer_config.cpp
    optimizer_info.cpp
    optimizer_optimize.cpp
    workspace.cpp
    slice_group.cpp
    contraction_plan.cpp
    contraction_autotune.cpp
    contraction_execute.cpp
    tensor_svd.cpp
    tensor_qr.cpp
    gate_split.cpp
    gradient.cpp
    network_operator.cpp
    network_state.cpp
    state_derived.cpp
)

add_library(roctensornet ${roctensornet_sources})

target_compile_features(roctensornet PRIVATE cxx_std_17)
set_target_properties(roctensornet PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_include_directories(roctensornet
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(TARGET hip::host)
    target_link_libraries(roctensornet PUBLIC hip::host)
endif()
if(TARGET roc::rocprim)
    target_link_libraries(roctensornet PRIVATE roc::rocprim)
endif()

# Versioning hooks for ROCm packaging
set_target_properties(roctensornet PROPERTIES
    VERSION   "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
)
