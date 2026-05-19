/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPSTATEVEC_EXPECT_H
#define HIPSTATEVEC_EXPECT_H

#include "../hipstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipstatevecStatus_t hipstatevecComputeExpectationGetWorkspaceSize(
    hipstatevecHandle_t        handle,
    hipstatevecDataType_t      state_vector_data_type,
    uint32_t                   n_index_bits,
    const void*                matrix,
    hipstatevecDataType_t      matrix_data_type,
    hipstatevecMatrixLayout_t  layout,
    uint32_t                   n_basis_bits,
    hipstatevecComputeType_t   compute_type,
    size_t*                    extra_workspace_size_in_bytes);

hipstatevecStatus_t hipstatevecComputeExpectation(
    hipstatevecHandle_t        handle,
    const void*                state_vector,
    hipstatevecDataType_t      state_vector_data_type,
    uint32_t                   n_index_bits,
    void*                      expectation_value,
    hipstatevecDataType_t      expectation_data_type,
    double*                    residual_norm,
    const void*                matrix,
    hipstatevecDataType_t      matrix_data_type,
    hipstatevecMatrixLayout_t  layout,
    const int32_t*             basis_bits,
    uint32_t                   n_basis_bits,
    hipstatevecComputeType_t   compute_type,
    void*                      extra_workspace,
    size_t                     extra_workspace_size_in_bytes);

hipstatevecStatus_t hipstatevecComputeExpectationsOnPauliBasis(
    hipstatevecHandle_t              handle,
    const void*                      state_vector,
    hipstatevecDataType_t            state_vector_data_type,
    uint32_t                         n_index_bits,
    double*                          expectation_values,
    const hipstatevecPauli_t* const* pauli_operators_array,
    uint32_t                         n_pauli_operator_arrays,
    const int32_t* const*            basis_bits_array,
    const uint32_t*                  n_basis_bits_array);

#ifdef __cplusplus
}
#endif

#endif /* HIPSTATEVEC_EXPECT_H */
