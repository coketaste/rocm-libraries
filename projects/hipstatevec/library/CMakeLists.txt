# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

include(GenerateExportHeader)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/hipstatevec-version.h.in"
               "${PROJECT_BINARY_DIR}/include/hipstatevec/hipstatevec-version.h")

set(hipstatevec_headers_public
  include/hipstatevec.h
  include/hipstatevec-types.h
  include/hipstatevec-auxiliary.h
  include/hipstatevec-functions.h
  include/internal/hipstatevec-init.h
  include/internal/hipstatevec-apply.h
  include/internal/hipstatevec-measure.h
  include/internal/hipstatevec-sample.h
  include/internal/hipstatevec-expect.h
  include/internal/hipstatevec-accessor.h
  include/internal/hipstatevec-permute.h
  include/internal/hipstatevec-batched.h
  include/internal/hipstatevec-test.h
  ${PROJECT_BINARY_DIR}/include/hipstatevec/hipstatevec-version.h
)

source_group("Header Files\\Public" FILES ${hipstatevec_headers_public})

set(hipstatevec_source
  src/amd_detail/hipstatevec_amd.cpp
)

add_library(hipstatevec ${hipstatevec_source} ${hipstatevec_headers_public})
add_library(roc::hipstatevec ALIAS hipstatevec)

target_compile_definitions(hipstatevec PRIVATE HIPSTATEVEC_BACKEND_AMD=1)

target_compile_options(hipstatevec PRIVATE -Wno-unused-command-line-argument -Wall)

target_include_directories(hipstatevec
  PRIVATE  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
  PUBLIC   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipstatevec>
           $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(hipstatevec PRIVATE hip::host roc::rocstatevec)

set_target_properties(hipstatevec PROPERTIES
  VERSION   ${hipstatevec_VERSION}
  SOVERSION ${hipstatevec_SOVERSION}
  CXX_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE ON
)

generate_export_header(hipstatevec
  EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipstatevec/hipstatevec-export.h)

rocm_set_soversion(hipstatevec ${hipstatevec_SOVERSION})

rocm_install_targets(
  TARGETS hipstatevec
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
)

rocm_export_targets(
  TARGETS roc::hipstatevec
  NAMESPACE roc::
)
