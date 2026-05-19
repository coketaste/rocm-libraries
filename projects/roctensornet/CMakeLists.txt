# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

cmake_minimum_required(VERSION 3.16)

# CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(VERSION_STRING "0.1.0")

# Declare the project (and supply VERSION so that
# `PROJECT_VERSION_MAJOR/MINOR/PATCH` are populated for
# `roctensornet-version.h.in`).
project(roctensornet VERSION ${VERSION_STRING} LANGUAGES CXX)

# Build options ----------------------------------------------------------------
option(BUILD_SHARED_LIBS    "Build a shared library"      ON)
option(BUILD_CLIENTS_TESTS  "Build the GoogleTest suite"  OFF)
option(BUILD_CLIENTS_SAMPLES "Build the C/C++ samples"    OFF)
option(BUILD_CLIENTS_BENCHMARKS "Build the benchmarks"    OFF)
option(BUILD_DOCS           "Build the Sphinx docs"       OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# rocm-cmake (optional in stand-alone builds). When the package is
# available we pull in its packaging helpers; otherwise we provide a
# no-op fallback for `rocm_setup_version` so the rest of the build
# script can be reused unchanged.
find_package(ROCmCMakeBuildTools CONFIG QUIET)
if(ROCmCMakeBuildTools_FOUND)
    include(ROCMSetupVersion)
    include(ROCMCreatePackage)
    include(ROCMInstallTargets)
    include(ROCMExportTargetsHeaderOnly)
    include(ROCMClients)
    include(ROCMHeaderWrapper)
endif()
if(NOT COMMAND rocm_setup_version)
    function(rocm_setup_version)
    endfunction()
endif()
rocm_setup_version(VERSION ${VERSION_STRING})

# Dependencies ----------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Dependencies.cmake)

# Library --------------------------------------------------------------------
add_subdirectory(library)

# Clients --------------------------------------------------------------------
if(BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_BENCHMARKS)
    add_subdirectory(clients)
endif()

# Docs -----------------------------------------------------------------------
if(BUILD_DOCS)
    add_subdirectory(docs)
endif()
