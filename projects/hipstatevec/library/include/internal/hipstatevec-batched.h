/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPSTATEVEC_BATCHED_H
#define HIPSTATEVEC_BATCHED_H

#include "../hipstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipstatevecStatus_t hipstatevecApplyMatrixBatchedGetWorkspaceSize(
    hipstatevecHandle_t              handle,
    hipstatevecDataType_t            state_vector_data_type,
    uint32_t                         n_index_bits,
    uint32_t                         n_state_vectors,
    hipstatevecIndex_t               state_vector_size,
    hipstatevecMatrixMapType_t       map_type,
    const int32_t*                   matrix_indices,
    const void*                      matrices,
    hipstatevecDataType_t            matrix_data_type,
    hipstatevecMatrixLayout_t        layout,
    int32_t                          adjoint,
    uint32_t                         n_matrices,
    uint32_t                         n_targets,
    uint32_t                         n_controls,
    hipstatevecComputeType_t         compute_type,
    size_t*                          extra_workspace_size_in_bytes);

hipstatevecStatus_t hipstatevecApplyMatrixBatched(
    hipstatevecHandle_t              handle,
    void*                            batched_state_vectors,
    hipstatevecDataType_t            state_vector_data_type,
    uint32_t                         n_index_bits,
    uint32_t                         n_state_vectors,
    hipstatevecIndex_t               state_vector_size,
    hipstatevecMatrixMapType_t       map_type,
    const int32_t*                   matrix_indices,
    const void*                      matrices,
    hipstatevecDataType_t            matrix_data_type,
    hipstatevecMatrixLayout_t        layout,
    int32_t                          adjoint,
    uint32_t                         n_matrices,
    const int32_t*                   targets,
    uint32_t                         n_targets,
    const int32_t*                   controls,
    const int32_t*                   control_bit_values,
    uint32_t                         n_controls,
    hipstatevecComputeType_t         compute_type,
    void*                            extra_workspace,
    size_t                           extra_workspace_size_in_bytes);

#ifdef __cplusplus
}
#endif

#endif /* HIPSTATEVEC_BATCHED_H */
