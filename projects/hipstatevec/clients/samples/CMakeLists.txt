# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

add_executable(hipstatevec-bell-pair bell_pair.cpp)
target_link_libraries(hipstatevec-bell-pair PRIVATE roc::hipstatevec hip::host)
