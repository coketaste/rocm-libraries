# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

add_executable(rocdensitymat-rabi-oscillator rabi_oscillator.cpp)
target_link_libraries(rocdensitymat-rabi-oscillator PRIVATE roc::rocdensitymat)

add_executable(rocdensitymat-bell-state-evolution bell_state_evolution.cpp)
target_link_libraries(rocdensitymat-bell-state-evolution PRIVATE roc::rocdensitymat)

add_executable(rocdensitymat-harmonic-oscillator harmonic_oscillator.cpp)
target_link_libraries(rocdensitymat-harmonic-oscillator PRIVATE roc::rocdensitymat)
