/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Backward-differentiation compute path. NOT_SUPPORTED in v0.1.0.
 * ************************************************************************ */

#ifndef HIPDENSITYMAT_BACKWARD_H
#define HIPDENSITYMAT_BACKWARD_H

#include "../hipdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipdensitymatStatus_t hipdensitymatOperatorPrepareActionBackwardDiff(
    hipdensitymatHandle_t handle,
    hipdensitymatOperator_t op,
    hipdensitymatState_t stateIn, hipdensitymatState_t stateOut,
    hipdensitymatComputeType_t computeType,
    size_t workspaceSizeLimit,
    hipdensitymatWorkspaceDescriptor_t workspace);

hipdensitymatStatus_t hipdensitymatOperatorComputeActionBackwardDiff(
    hipdensitymatHandle_t handle,
    hipdensitymatOperator_t op,
    double t, int32_t numParams, const double* params,
    hipdensitymatState_t stateIn,
    hipdensitymatState_t stateOutGrad,
    hipdensitymatState_t stateInGrad,
    double* paramsGrad,
    hipdensitymatWorkspaceDescriptor_t workspace);

#ifdef __cplusplus
}
#endif

#endif /* HIPDENSITYMAT_BACKWARD_H */
