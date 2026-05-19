/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Reverse-engineered from cuStateVec public API documentation
 * (cuQuantum 24.11 / cuStateVec 1.7.x); clean-room implementation.
 * ************************************************************************ */

#ifndef ROCSTATEVEC_EXPECT_H
#define ROCSTATEVEC_EXPECT_H

#include "../rocstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Workspace query for `rocstatevec_compute_expectation`.
 *  Bijection: \p custatevecComputeExpectationGetWorkspaceSize. */
rocstatevec_status rocstatevec_compute_expectation_get_workspace_size(
    rocstatevec_handle         handle,
    rocstatevec_data_type      state_vector_data_type,
    uint32_t                   n_index_bits,
    const void*                matrix,
    rocstatevec_data_type      matrix_data_type,
    rocstatevec_matrix_layout  layout,
    uint32_t                   n_basis_bits,
    rocstatevec_compute_type   compute_type,
    size_t*                    extra_workspace_size_in_bytes);

/*! \brief Compute <psi| M |psi> for a dense Hermitian matrix on selected qubits.
 *  Bijection: \p custatevecComputeExpectation. */
rocstatevec_status rocstatevec_compute_expectation(
    rocstatevec_handle         handle,
    const void*                state_vector,
    rocstatevec_data_type      state_vector_data_type,
    uint32_t                   n_index_bits,
    void*                      expectation_value,
    rocstatevec_data_type      expectation_data_type,
    double*                    residual_norm,
    const void*                matrix,
    rocstatevec_data_type      matrix_data_type,
    rocstatevec_matrix_layout  layout,
    const int32_t*             basis_bits,
    uint32_t                   n_basis_bits,
    rocstatevec_compute_type   compute_type,
    void*                      extra_workspace,
    size_t                     extra_workspace_size_in_bytes);

/*! \brief Compute expectation values of a list of Pauli strings.
 *  Bijection: \p custatevecComputeExpectationsOnPauliBasis. */
rocstatevec_status rocstatevec_compute_expectations_on_pauli_basis(
    rocstatevec_handle               handle,
    const void*                      state_vector,
    rocstatevec_data_type            state_vector_data_type,
    uint32_t                         n_index_bits,
    double*                          expectation_values,
    const rocstatevec_pauli* const*  pauli_operators_array,
    uint32_t                         n_pauli_operator_arrays,
    const int32_t* const*            basis_bits_array,
    const uint32_t*                  n_basis_bits_array);

#ifdef __cplusplus
}
#endif

#endif /* ROCSTATEVEC_EXPECT_H */
