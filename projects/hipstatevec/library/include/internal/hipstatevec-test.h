/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPSTATEVEC_TEST_H
#define HIPSTATEVEC_TEST_H

#include "../hipstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipstatevecStatus_t hipstatevecTestMatrixTypeGetWorkspaceSize(
    hipstatevecHandle_t        handle,
    hipstatevecMatrixType_t    matrix_type,
    const void*                matrix,
    hipstatevecDataType_t      matrix_data_type,
    hipstatevecMatrixLayout_t  layout,
    uint32_t                   n_targets,
    int32_t                    adjoint,
    hipstatevecComputeType_t   compute_type,
    size_t*                    extra_workspace_size_in_bytes);

hipstatevecStatus_t hipstatevecTestMatrixType(
    hipstatevecHandle_t        handle,
    double*                    residual_norm,
    hipstatevecMatrixType_t    matrix_type,
    const void*                matrix,
    hipstatevecDataType_t      matrix_data_type,
    hipstatevecMatrixLayout_t  layout,
    uint32_t                   n_targets,
    int32_t                    adjoint,
    hipstatevecComputeType_t   compute_type,
    void*                      extra_workspace,
    size_t                     extra_workspace_size_in_bytes);

#ifdef __cplusplus
}
#endif

#endif /* HIPSTATEVEC_TEST_H */
