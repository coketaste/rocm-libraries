# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

set(rocdensitymat_source
  src/handle.cpp
  src/version.cpp
  src/error.cpp
  src/logger.cpp
  src/workspace.cpp
  src/state.cpp
  src/operator_elementary.cpp
  src/operator_term.cpp
  src/operator.cpp
  src/liouvillian.cpp
  src/apply_pure.cpp
  src/apply_mixed.cpp
  src/operator_compute_action.cpp
  src/expectation.cpp
  src/properties.cpp
  src/ode_rk4.cpp
  src/distributed.cpp
  src/eigen.cpp
  src/backward.cpp
)
