# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

set(hiptensornet_sources src/amd_detail/hiptensornet_amd.cpp)

add_library(hiptensornet ${hiptensornet_sources})
target_compile_features(hiptensornet PRIVATE cxx_std_17)

set_target_properties(hiptensornet PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${hiptensornet_SOVERSION}
)

target_compile_definitions(hiptensornet PRIVATE HIPTENSORNET_BACKEND_AMD=1)

target_include_directories(hiptensornet PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(TARGET hip::host)
    target_link_libraries(hiptensornet PUBLIC hip::host)
endif()
if(TARGET roc::roctensornet)
    target_link_libraries(hiptensornet PUBLIC roc::roctensornet)
elseif(TARGET roctensornet)
    target_link_libraries(hiptensornet PUBLIC roctensornet)
endif()

install(FILES include/hiptensornet.h
        DESTINATION include
        COMPONENT devel)

install(TARGETS hiptensornet
        EXPORT hiptensornet-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
install(EXPORT hiptensornet-targets
        FILE hiptensornet-targets.cmake
        NAMESPACE roc::
        DESTINATION lib/cmake/hiptensornet)
if(NOT TARGET roc::hiptensornet)
    add_library(roc::hiptensornet ALIAS hiptensornet)
endif()
