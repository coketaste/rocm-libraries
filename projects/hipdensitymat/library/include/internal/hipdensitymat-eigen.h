/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Operator eigenspectrum compute path. NOT_SUPPORTED in v0.1.0.
 * ************************************************************************ */

#ifndef HIPDENSITYMAT_EIGEN_H
#define HIPDENSITYMAT_EIGEN_H

#include "../hipdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipdensitymatStatus_t hipdensitymatOperatorPrepareEigenspectrum(
    hipdensitymatHandle_t handle,
    hipdensitymatOperator_t op,
    hipdensitymatState_t state,
    int32_t numEigenpairs,
    hipdensitymatComputeType_t computeType,
    size_t workspaceSizeLimit,
    hipdensitymatWorkspaceDescriptor_t workspace);

hipdensitymatStatus_t hipdensitymatOperatorComputeEigenspectrum(
    hipdensitymatHandle_t handle,
    hipdensitymatOperator_t op,
    int32_t numEigenpairs,
    hipdensitymatComplexDouble_t* eigenvalues,
    hipdensitymatState_t* eigenvectors,
    hipdensitymatWorkspaceDescriptor_t workspace);

#ifdef __cplusplus
}
#endif

#endif /* HIPDENSITYMAT_EIGEN_H */
