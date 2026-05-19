/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Reverse-engineered from cuStateVec public API documentation
 * (cuQuantum 24.11 / cuStateVec 1.7.x); clean-room implementation.
 * ************************************************************************ */

#ifndef ROCSTATEVEC_BATCHED_H
#define ROCSTATEVEC_BATCHED_H

#include "../rocstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Workspace query for `rocstatevec_apply_matrix_batched`.
 *  Bijection: \p custatevecApplyMatrixBatchedGetWorkspaceSize. */
rocstatevec_status rocstatevec_apply_matrix_batched_get_workspace_size(
    rocstatevec_handle              handle,
    rocstatevec_data_type           state_vector_data_type,
    uint32_t                        n_index_bits,
    uint32_t                        n_state_vectors,
    rocstatevec_index_t             state_vector_size,
    rocstatevec_matrix_map_type     map_type,
    const int32_t*                  matrix_indices,
    const void*                     matrices,
    rocstatevec_data_type           matrix_data_type,
    rocstatevec_matrix_layout       layout,
    int32_t                         adjoint,
    uint32_t                        n_matrices,
    uint32_t                        n_targets,
    uint32_t                        n_controls,
    rocstatevec_compute_type        compute_type,
    size_t*                         extra_workspace_size_in_bytes);

/*! \brief Apply a batched dense matrix to a stack of statevectors.
 *  Bijection: \p custatevecApplyMatrixBatched. */
rocstatevec_status rocstatevec_apply_matrix_batched(
    rocstatevec_handle              handle,
    void*                           batched_state_vectors,
    rocstatevec_data_type           state_vector_data_type,
    uint32_t                        n_index_bits,
    uint32_t                        n_state_vectors,
    rocstatevec_index_t             state_vector_size,
    rocstatevec_matrix_map_type     map_type,
    const int32_t*                  matrix_indices,
    const void*                     matrices,
    rocstatevec_data_type           matrix_data_type,
    rocstatevec_matrix_layout       layout,
    int32_t                         adjoint,
    uint32_t                        n_matrices,
    const int32_t*                  targets,
    uint32_t                        n_targets,
    const int32_t*                  controls,
    const int32_t*                  control_bit_values,
    uint32_t                        n_controls,
    rocstatevec_compute_type        compute_type,
    void*                           extra_workspace,
    size_t                          extra_workspace_size_in_bytes);

#ifdef __cplusplus
}
#endif

#endif /* ROCSTATEVEC_BATCHED_H */
