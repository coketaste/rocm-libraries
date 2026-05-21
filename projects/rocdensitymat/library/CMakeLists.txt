# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GenerateExportHeader)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocdensitymat-version.h.in"
               "${PROJECT_BINARY_DIR}/include/rocdensitymat/rocdensitymat-version.h")

set(rocdensitymat_headers_public
  include/rocdensitymat.h
  include/rocdensitymat-auxiliary.h
  include/rocdensitymat-functions.h
  include/rocdensitymat-types.h
  include/internal/rocdensitymat-handle.h
  include/internal/rocdensitymat-workspace.h
  include/internal/rocdensitymat-state.h
  include/internal/rocdensitymat-operator.h
  include/internal/rocdensitymat-action.h
  include/internal/rocdensitymat-ode.h
  include/internal/rocdensitymat-properties.h
  include/internal/rocdensitymat-distributed.h
  include/internal/rocdensitymat-eigen.h
  include/internal/rocdensitymat-backward.h
  ${PROJECT_BINARY_DIR}/include/rocdensitymat/rocdensitymat-version.h
)

source_group("Header Files\\Public" FILES ${rocdensitymat_headers_public})

include(src/CMakeLists.txt)

add_library(rocdensitymat ${rocdensitymat_source} ${rocdensitymat_headers_public})
add_library(roc::rocdensitymat ALIAS rocdensitymat)

target_compile_options(rocdensitymat PRIVATE -Wno-unused-command-line-argument -Wall)

target_include_directories(rocdensitymat
  PRIVATE  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
  PUBLIC   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocdensitymat>
           $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(rocdensitymat PRIVATE roc::rocprim hip::device)
set(static_depends PACKAGE rocprim)

set_target_properties(rocdensitymat PROPERTIES
  VERSION   ${rocdensitymat_VERSION}
  SOVERSION ${rocdensitymat_SOVERSION}
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  POSITION_INDEPENDENT_CODE ON
)

generate_export_header(rocdensitymat
  EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocdensitymat/rocdensitymat-export.h)
list(APPEND rocdensitymat_headers_public
  ${PROJECT_BINARY_DIR}/include/rocdensitymat/rocdensitymat-export.h)

rocm_set_soversion(rocdensitymat ${rocdensitymat_SOVERSION})

rocm_install_targets(
  TARGETS rocdensitymat
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
)

rocm_export_targets(
  TARGETS roc::rocdensitymat
  DEPENDS PACKAGE hip
  STATIC_DEPENDS ${static_depends}
  NAMESPACE roc::
)

rocm_install_symlink_subdir(rocdensitymat)
