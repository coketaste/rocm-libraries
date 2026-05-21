# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

add_executable(hipdensitymat-rabi-oscillator rabi_oscillator.cpp)
target_link_libraries(hipdensitymat-rabi-oscillator PRIVATE roc::hipdensitymat hip::host)
