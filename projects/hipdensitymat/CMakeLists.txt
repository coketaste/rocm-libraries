# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix")
endif()

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip /opt/rocm/hip/cmake)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

project(hipdensitymat LANGUAGES CXX C)

option(BUILD_SHARED_LIBS         "Build hipDENSITYMAT as a shared library" ON)
option(BUILD_CLIENTS_TESTS       "Build tests" OFF)
option(BUILD_CLIENTS_SAMPLES     "Build samples" OFF)
option(BUILD_DOCS                "Build documentation" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(cmake/Dependencies.cmake)

set(VERSION_STRING "0.1.0")
set(SOVERSION_STRING "0.1")
rocm_setup_version(VERSION ${VERSION_STRING})
set(hipdensitymat_SOVERSION ${SOVERSION_STRING})

find_package(hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm)
# Under the rocm-libraries superbuild, rocdensitymat has already been
# registered as `roc::rocdensitymat` via add_subdirectory; skip the
# external config-package lookup so we link against the in-tree target.
if(NOT TARGET roc::rocdensitymat)
  find_package(rocdensitymat REQUIRED CONFIG PATHS ${CMAKE_INSTALL_PREFIX} /opt/rocm)
endif()

if(BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_SAMPLES)
  set(BUILD_CLIENTS ON)
  if(BUILD_CLIENTS_TESTS)
    enable_testing()
  endif()
endif()

add_subdirectory(library)

if(BUILD_CLIENTS)
  add_subdirectory(clients)
endif()

if(BUILD_DOCS)
  add_subdirectory(docs)
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(package_name hipdensitymat)
rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "cuDensityMat-shaped C API thin wrapper for AMD ROCm (rocDENSITYMAT backend)"
  MAINTAINER "hipDENSITYMAT Maintainer <hipdensitymat-maintainer@amd.com>"
  LDCONFIG
)
