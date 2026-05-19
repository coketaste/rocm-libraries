# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

set(rocstatevec_source
  src/handle.cpp
  src/error.cpp
  src/logger.cpp
  src/initialize_state.cpp
  src/abs2_sum.cpp
  src/measure_z.cpp
  src/collapse.cpp
  src/batch_measure.cpp
  src/sampler.cpp
  src/apply_matrix.cpp
  src/apply_pauli_rotation.cpp
  src/apply_generalized_permutation.cpp
  src/absorb_diagonal.cpp
  src/apply_matrix_batched.cpp
  src/expectation.cpp
  src/expectations_pauli_basis.cpp
  src/accessor.cpp
  src/swap_index_bits.cpp
  src/test_matrix_type.cpp
)
