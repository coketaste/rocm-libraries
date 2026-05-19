# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

# Generate the version header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/roctensornet-version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/roctensornet-version.h
    @ONLY
)

add_subdirectory(src)

# Public headers
set(roctensornet_public_headers
    include/roctensornet.h
    include/roctensornet-types.h
    include/roctensornet-auxiliary.h
    include/roctensornet-functions.h
    ${CMAKE_CURRENT_BINARY_DIR}/include/roctensornet-version.h
)
set(roctensornet_public_internal_headers
    include/internal/roctensornet-network.h
    include/internal/roctensornet-optimizer.h
    include/internal/roctensornet-workspace.h
    include/internal/roctensornet-contraction.h
    include/internal/roctensornet-svd.h
    include/internal/roctensornet-gradient.h
    include/internal/roctensornet-network-operator.h
    include/internal/roctensornet-state.h
    include/internal/roctensornet-derived.h
)

target_include_directories(roctensornet PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Install
install(FILES ${roctensornet_public_headers}
        DESTINATION include
        COMPONENT devel)
install(FILES ${roctensornet_public_internal_headers}
        DESTINATION include/internal
        COMPONENT devel)

if(COMMAND rocm_install_targets)
    rocm_install_targets(
        TARGETS roctensornet
        INCLUDE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}/include
    )
else()
    install(TARGETS roctensornet
            EXPORT roctensornet-targets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
            COMPONENT runtime)
    install(EXPORT roctensornet-targets
            FILE roctensornet-targets.cmake
            NAMESPACE roc::
            DESTINATION lib/cmake/roctensornet)
endif()

# Alias for downstream consumers
if(NOT TARGET roc::roctensornet)
    add_library(roc::roctensornet ALIAS roctensornet)
endif()
