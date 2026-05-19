# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

cmake_minimum_required(VERSION 3.16)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(hiptensornet LANGUAGES CXX VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(hiptensornet_SOVERSION 0.1)

option(BUILD_SHARED_LIBS         "Build shared library"           ON)
option(BUILD_CLIENTS_TESTS       "Build the GoogleTest suite"     OFF)
option(BUILD_CLIENTS_SAMPLES     "Build the C/C++ samples"        OFF)
option(BUILD_CLIENTS_BENCHMARKS  "Build the benchmarks"           OFF)
option(BUILD_DOCS                "Build the Sphinx docs"          OFF)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Dependencies.cmake)
add_subdirectory(library)

if(BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_BENCHMARKS)
    add_subdirectory(clients)
endif()
if(BUILD_DOCS)
    add_subdirectory(docs)
endif()
