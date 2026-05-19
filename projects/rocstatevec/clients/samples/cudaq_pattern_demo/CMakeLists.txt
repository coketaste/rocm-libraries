# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

add_executable(rocstatevec-cudaq-pattern-demo cudaq_pattern_demo.cpp)
target_link_libraries(rocstatevec-cudaq-pattern-demo PRIVATE roc::rocstatevec)
