/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPSTATEVEC_APPLY_H
#define HIPSTATEVEC_APPLY_H

#include "../hipstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipstatevecStatus_t hipstatevecApplyMatrixGetWorkspaceSize(
    hipstatevecHandle_t        handle,
    hipstatevecDataType_t      state_vector_data_type,
    uint32_t                   n_index_bits,
    const void*                matrix,
    hipstatevecDataType_t      matrix_data_type,
    hipstatevecMatrixLayout_t  layout,
    int32_t                    adjoint,
    uint32_t                   n_targets,
    uint32_t                   n_controls,
    hipstatevecComputeType_t   compute_type,
    size_t*                    extra_workspace_size_in_bytes);

hipstatevecStatus_t hipstatevecApplyMatrix(
    hipstatevecHandle_t        handle,
    void*                      state_vector,
    hipstatevecDataType_t      state_vector_data_type,
    uint32_t                   n_index_bits,
    const void*                matrix,
    hipstatevecDataType_t      matrix_data_type,
    hipstatevecMatrixLayout_t  layout,
    int32_t                    adjoint,
    const int32_t*             targets,
    uint32_t                   n_targets,
    const int32_t*             controls,
    const int32_t*             control_bit_values,
    uint32_t                   n_controls,
    hipstatevecComputeType_t   compute_type,
    void*                      extra_workspace,
    size_t                     extra_workspace_size_in_bytes);

hipstatevecStatus_t hipstatevecApplyPauliRotation(
    hipstatevecHandle_t          handle,
    void*                        state_vector,
    hipstatevecDataType_t        state_vector_data_type,
    uint32_t                     n_index_bits,
    double                       theta,
    const hipstatevecPauli_t*    paulis,
    const int32_t*               targets,
    uint32_t                     n_targets,
    const int32_t*               controls,
    const int32_t*               control_bit_values,
    uint32_t                     n_controls);

hipstatevecStatus_t hipstatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(
    hipstatevecHandle_t        handle,
    hipstatevecDataType_t      state_vector_data_type,
    uint32_t                   n_index_bits,
    const hipstatevecIndex_t*  permutation,
    const void*                diagonals,
    hipstatevecDataType_t      diagonals_data_type,
    const int32_t*             targets,
    uint32_t                   n_targets,
    uint32_t                   n_controls,
    size_t*                    extra_workspace_size_in_bytes);

hipstatevecStatus_t hipstatevecApplyGeneralizedPermutationMatrix(
    hipstatevecHandle_t        handle,
    void*                      state_vector,
    hipstatevecDataType_t      state_vector_data_type,
    uint32_t                   n_index_bits,
    const hipstatevecIndex_t*  permutation,
    const void*                diagonals,
    hipstatevecDataType_t      diagonals_data_type,
    int32_t                    adjoint,
    const int32_t*             targets,
    uint32_t                   n_targets,
    const int32_t*             controls,
    const int32_t*             control_bit_values,
    uint32_t                   n_controls,
    void*                      extra_workspace,
    size_t                     extra_workspace_size_in_bytes);

hipstatevecStatus_t hipstatevecAbsorbDiagonalMatrix(
    hipstatevecHandle_t        handle,
    void*                      state_vector,
    hipstatevecDataType_t      state_vector_data_type,
    uint32_t                   n_index_bits,
    const void*                diagonals,
    hipstatevecDataType_t      diagonals_data_type,
    int32_t                    adjoint,
    const int32_t*             targets,
    uint32_t                   n_targets,
    const int32_t*             controls,
    const int32_t*             control_bit_values,
    uint32_t                   n_controls);

#ifdef __cplusplus
}
#endif

#endif /* HIPSTATEVEC_APPLY_H */
