# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GenerateExportHeader)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocstatevec-version.h.in"
               "${PROJECT_BINARY_DIR}/include/rocstatevec/rocstatevec-version.h")

set(rocstatevec_headers_public
  include/rocstatevec.h
  include/rocstatevec-auxiliary.h
  include/rocstatevec-functions.h
  include/rocstatevec-types.h
  include/internal/rocstatevec-init.h
  include/internal/rocstatevec-apply.h
  include/internal/rocstatevec-measure.h
  include/internal/rocstatevec-sample.h
  include/internal/rocstatevec-expect.h
  include/internal/rocstatevec-accessor.h
  include/internal/rocstatevec-permute.h
  include/internal/rocstatevec-batched.h
  include/internal/rocstatevec-test.h
  ${PROJECT_BINARY_DIR}/include/rocstatevec/rocstatevec-version.h
)

source_group("Header Files\\Public" FILES ${rocstatevec_headers_public})

include(src/CMakeLists.txt)

add_library(rocstatevec ${rocstatevec_source} ${rocstatevec_headers_public})
add_library(roc::rocstatevec ALIAS rocstatevec)

target_compile_options(rocstatevec PRIVATE -Wno-unused-command-line-argument -Wall)

target_include_directories(rocstatevec
  PRIVATE  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
  PUBLIC   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocstatevec>
           $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(rocstatevec PRIVATE roc::rocprim hip::device)
set(static_depends PACKAGE rocprim)

set_target_properties(rocstatevec PROPERTIES
  VERSION   ${rocstatevec_VERSION}
  SOVERSION ${rocstatevec_SOVERSION}
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  POSITION_INDEPENDENT_CODE ON
)

generate_export_header(rocstatevec
  EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocstatevec/rocstatevec-export.h)
list(APPEND rocstatevec_headers_public
  ${PROJECT_BINARY_DIR}/include/rocstatevec/rocstatevec-export.h)

rocm_set_soversion(rocstatevec ${rocstatevec_SOVERSION})

rocm_install_targets(
  TARGETS rocstatevec
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
)

rocm_export_targets(
  TARGETS roc::rocstatevec
  DEPENDS PACKAGE hip
  STATIC_DEPENDS ${static_depends}
  NAMESPACE roc::
)

rocm_install_symlink_subdir(rocstatevec)
