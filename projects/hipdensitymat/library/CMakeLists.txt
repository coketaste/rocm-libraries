# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

include(GenerateExportHeader)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/hipdensitymat-version.h.in"
               "${PROJECT_BINARY_DIR}/include/hipdensitymat/hipdensitymat-version.h")

set(hipdensitymat_headers_public
  include/hipdensitymat.h
  include/hipdensitymat-types.h
  include/hipdensitymat-auxiliary.h
  include/hipdensitymat-functions.h
  include/internal/hipdensitymat-handle.h
  include/internal/hipdensitymat-workspace.h
  include/internal/hipdensitymat-state.h
  include/internal/hipdensitymat-operator.h
  include/internal/hipdensitymat-action.h
  include/internal/hipdensitymat-ode.h
  include/internal/hipdensitymat-properties.h
  include/internal/hipdensitymat-distributed.h
  include/internal/hipdensitymat-eigen.h
  include/internal/hipdensitymat-backward.h
  ${PROJECT_BINARY_DIR}/include/hipdensitymat/hipdensitymat-version.h
)

source_group("Header Files\\Public" FILES ${hipdensitymat_headers_public})

set(hipdensitymat_source
  src/amd_detail/hipdensitymat_amd.cpp
)

add_library(hipdensitymat ${hipdensitymat_source} ${hipdensitymat_headers_public})
add_library(roc::hipdensitymat ALIAS hipdensitymat)

target_compile_definitions(hipdensitymat PRIVATE HIPDENSITYMAT_BACKEND_AMD=1)

target_compile_options(hipdensitymat PRIVATE -Wno-unused-command-line-argument -Wall)

target_include_directories(hipdensitymat
  PRIVATE  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
  PUBLIC   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipdensitymat>
           $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(hipdensitymat PRIVATE hip::host roc::rocdensitymat)

set_target_properties(hipdensitymat PROPERTIES
  VERSION   ${hipdensitymat_VERSION}
  SOVERSION ${hipdensitymat_SOVERSION}
  CXX_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE ON
)

generate_export_header(hipdensitymat
  EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipdensitymat/hipdensitymat-export.h)

rocm_set_soversion(hipdensitymat ${hipdensitymat_SOVERSION})

rocm_install_targets(
  TARGETS hipdensitymat
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
)

rocm_export_targets(
  TARGETS roc::hipdensitymat
  NAMESPACE roc::
)
