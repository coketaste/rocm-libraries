# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

if(BUILD_CLIENTS_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_CLIENTS_SAMPLES)
    add_subdirectory(samples)
endif()

if(BUILD_CLIENTS_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
