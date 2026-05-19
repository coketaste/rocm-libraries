# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip /opt/rocm/hip/cmake)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

project(rocstatevec LANGUAGES CXX C)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "Using hip-clang to build for amdgpu backend")
else()
  message(FATAL_ERROR "'hip-clang' compiler required to compile for ROCm platform.")
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++17 is supported")
endif()

option(BUILD_SHARED_LIBS         "Build rocSTATEVEC as a shared library" ON)
option(BUILD_CLIENTS_TESTS       "Build tests (requires googletest)" OFF)
option(BUILD_CLIENTS_BENCHMARKS  "Build benchmarks" OFF)
option(BUILD_CLIENTS_SAMPLES     "Build examples" OFF)
option(BUILD_CLIENTS_ONLY        "Build only clients" OFF)
option(BUILD_VERBOSE             "Output additional build information" OFF)
option(BUILD_ADDRESS_SANITIZER   "Build rocSTATEVEC with address sanitizer enabled" OFF)
option(BUILD_DOCS                "Build documentation" OFF)

if(BUILD_ADDRESS_SANITIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
  add_compile_definitions(ROCSTATEVEC_WITH_ASAN)
endif()

include(cmake/Dependencies.cmake)

# GPU targets — copy of rocsparse defaults; tuned to architectures with
# meaningful FP64 throughput plus modern RDNA targets for bring-up.
if(COMMAND rocm_check_target_ids)
  if(BUILD_ADDRESS_SANITIZER)
    rocm_check_target_ids(DEFAULT_GPU_TARGETS
      TARGETS "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+")
  else()
    rocm_check_target_ids(DEFAULT_GPU_TARGETS
      TARGETS "gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx942;gfx950;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201")
  endif()
else()
  set(DEFAULT_GPU_TARGETS "gfx908;gfx90a;gfx942;gfx1100")
endif()

if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
  message(DEPRECATION "AMDGPU_TARGETS use is deprecated. Use GPU_TARGETS.")
  set(GPU_TARGETS "${AMDGPU_TARGETS}")
elseif(NOT GPU_TARGETS)
  set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}" CACHE STRING "GPU architectures to build for")
endif()
set(GPU_TARGETS "${GPU_TARGETS}" CACHE STRING "GPU architectures to build for" FORCE)

message(STATUS "GPU_TARGETS: ${GPU_TARGETS}")

set(VERSION_STRING "0.1.0")
set(SOVERSION_STRING "0.1")
set(MIN_ROCPRIM_VERSION "4.0.0")

rocm_setup_version(VERSION ${VERSION_STRING})
set(rocstatevec_SOVERSION ${SOVERSION_STRING})

# In a stand-alone build we resolve rocprim through find_package; under
# the rocm-libraries superbuild it has already been added via
# add_subdirectory and the target name is registered, so skip the
# package query to avoid a redundant lookup against /opt/rocm.
if(NOT TARGET rocprim)
  find_package(rocprim REQUIRED)
endif()
find_package(hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm)

if(BUILD_CLIENTS_ONLY OR BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS)
  set(BUILD_CLIENTS ON)
  if(BUILD_CLIENTS_TESTS)
    enable_testing()
  endif()
  rocm_package_setup_component(clients)
  rocm_package_setup_client_component(clients-common)
  if(BUILD_CLIENTS_TESTS)
    rocm_package_setup_client_component(tests DEPENDS COMPONENT clients-common)
  endif()
  if(BUILD_CLIENTS_BENCHMARKS)
    rocm_package_setup_client_component(benchmarks DEPENDS COMPONENT clients-common)
  endif()
endif()

if(NOT BUILD_CLIENTS_ONLY)
  add_subdirectory(library)
endif()

if(BUILD_CLIENTS)
  add_subdirectory(clients)
endif()

if(BUILD_DOCS)
  add_subdirectory(docs)
endif()

set(HIP_RUNTIME_MINIMUM 6.0.0)
if(BUILD_ADDRESS_SANITIZER)
  set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan")
else()
  set(DEPENDS_HIP_RUNTIME "hip-runtime-amd")
endif()
rocm_package_add_dependencies(SHARED_DEPENDS "${DEPENDS_HIP_RUNTIME} >= ${HIP_RUNTIME_MINIMUM}")
rocm_package_add_deb_dependencies(STATIC_DEPENDS "hip-static-dev >= ${HIP_RUNTIME_MINIMUM}")
rocm_package_add_rpm_dependencies(STATIC_DEPENDS "hip-static-devel >= ${HIP_RUNTIME_MINIMUM}")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

if(WIN32)
  set(CPACK_SOURCE_GENERATOR "ZIP")
  set(CPACK_GENERATOR "ZIP")
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE)
  endif()
  set(INSTALL_PREFIX "C:/hipSDK")
  set(CPACK_SET_DESTDIR OFF)
  set(CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK")
  set(CPACK_PACKAGING_INSTALL_PREFIX "")
  set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
  if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
  endif()
endif()
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}")

set(package_name rocstatevec)
set(ROCSTATEVEC_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "ROCm state-vector library for quantum simulation"
  MAINTAINER "rocSTATEVEC Maintainer <rocstatevec-maintainer@amd.com>"
  LDCONFIG
  LDCONFIG_DIR ${ROCSTATEVEC_CONFIG_DIR}
)
