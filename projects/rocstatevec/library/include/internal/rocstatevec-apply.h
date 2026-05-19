/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Reverse-engineered from cuStateVec public API documentation
 * (cuQuantum 24.11 / cuStateVec 1.7.x); clean-room implementation.
 * ************************************************************************ */

#ifndef ROCSTATEVEC_APPLY_H
#define ROCSTATEVEC_APPLY_H

#include "../rocstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Query the workspace size needed by a subsequent
 *         `rocstatevec_apply_matrix` call with the same arguments.
 *
 *  Bijection: \p custatevecApplyMatrixGetWorkspaceSize. */
rocstatevec_status rocstatevec_apply_matrix_get_workspace_size(
    rocstatevec_handle         handle,
    rocstatevec_data_type      state_vector_data_type,
    uint32_t                   n_index_bits,
    const void*                matrix,
    rocstatevec_data_type      matrix_data_type,
    rocstatevec_matrix_layout  layout,
    int32_t                    adjoint,
    uint32_t                   n_targets,
    uint32_t                   n_controls,
    rocstatevec_compute_type   compute_type,
    size_t*                    extra_workspace_size_in_bytes);

/*! \brief Apply a dense matrix to selected targets, optionally with controls.
 *
 *  Bijection: \p custatevecApplyMatrix. */
rocstatevec_status rocstatevec_apply_matrix(
    rocstatevec_handle         handle,
    void*                      state_vector,
    rocstatevec_data_type      state_vector_data_type,
    uint32_t                   n_index_bits,
    const void*                matrix,
    rocstatevec_data_type      matrix_data_type,
    rocstatevec_matrix_layout  layout,
    int32_t                    adjoint,
    const int32_t*             targets,
    uint32_t                   n_targets,
    const int32_t*             controls,
    const int32_t*             control_bit_values,
    uint32_t                   n_controls,
    rocstatevec_compute_type   compute_type,
    void*                      extra_workspace,
    size_t                     extra_workspace_size_in_bytes);

/*! \brief Apply a Pauli-string rotation `exp(-i theta/2 P_0 ⊗ P_1 ⊗ ...)`.
 *
 *  Bijection: \p custatevecApplyPauliRotation. */
rocstatevec_status rocstatevec_apply_pauli_rotation(
    rocstatevec_handle           handle,
    void*                        state_vector,
    rocstatevec_data_type        state_vector_data_type,
    uint32_t                     n_index_bits,
    double                       theta,
    const rocstatevec_pauli*     paulis,
    const int32_t*               targets,
    uint32_t                     n_targets,
    const int32_t*               controls,
    const int32_t*               control_bit_values,
    uint32_t                     n_controls);

/*! \brief Query the workspace size for a subsequent
 *         `rocstatevec_apply_generalized_permutation_matrix` call.
 *
 *  Bijection: \p custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize. */
rocstatevec_status rocstatevec_apply_generalized_permutation_matrix_get_workspace_size(
    rocstatevec_handle         handle,
    rocstatevec_data_type      state_vector_data_type,
    uint32_t                   n_index_bits,
    const rocstatevec_index_t* permutation,
    const void*                diagonals,
    rocstatevec_data_type      diagonals_data_type,
    const int32_t*             targets,
    uint32_t                   n_targets,
    uint32_t                   n_controls,
    size_t*                    extra_workspace_size_in_bytes);

/*! \brief Apply a generalized permutation matrix (permutation + diagonal scaling).
 *
 *  Bijection: \p custatevecApplyGeneralizedPermutationMatrix. */
rocstatevec_status rocstatevec_apply_generalized_permutation_matrix(
    rocstatevec_handle         handle,
    void*                      state_vector,
    rocstatevec_data_type      state_vector_data_type,
    uint32_t                   n_index_bits,
    const rocstatevec_index_t* permutation,
    const void*                diagonals,
    rocstatevec_data_type      diagonals_data_type,
    int32_t                    adjoint,
    const int32_t*             targets,
    uint32_t                   n_targets,
    const int32_t*             controls,
    const int32_t*             control_bit_values,
    uint32_t                   n_controls,
    void*                      extra_workspace,
    size_t                     extra_workspace_size_in_bytes);

/*! \brief Multiply a state vector by a diagonal matrix in place.
 *
 *  Bijection: \p custatevecAbsorbDiagonalMatrix
 *  (also exposed historically as `MultiplyByDiagonalMatrix`). */
rocstatevec_status rocstatevec_absorb_diagonal_matrix(
    rocstatevec_handle      handle,
    void*                   state_vector,
    rocstatevec_data_type   state_vector_data_type,
    uint32_t                n_index_bits,
    const void*             diagonals,
    rocstatevec_data_type   diagonals_data_type,
    int32_t                 adjoint,
    const int32_t*          targets,
    uint32_t                n_targets,
    const int32_t*          controls,
    const int32_t*          control_bit_values,
    uint32_t                n_controls);

#ifdef __cplusplus
}
#endif

#endif /* ROCSTATEVEC_APPLY_H */
