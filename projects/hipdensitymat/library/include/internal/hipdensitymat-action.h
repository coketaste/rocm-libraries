/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPDENSITYMAT_ACTION_H
#define HIPDENSITYMAT_ACTION_H

#include "../hipdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipdensitymatStatus_t hipdensitymatOperatorPrepareAction(
    hipdensitymatHandle_t handle,
    hipdensitymatOperator_t op,
    hipdensitymatState_t stateIn, hipdensitymatState_t stateOut,
    hipdensitymatComputeType_t computeType,
    size_t workspaceSizeLimit,
    hipdensitymatWorkspaceDescriptor_t workspace);

hipdensitymatStatus_t hipdensitymatOperatorComputeAction(
    hipdensitymatHandle_t handle,
    hipdensitymatOperator_t op,
    double t,
    int32_t numParams,
    const double* params,
    hipdensitymatState_t stateIn, hipdensitymatState_t stateOut,
    hipdensitymatWorkspaceDescriptor_t workspace);

hipdensitymatStatus_t hipdensitymatOperatorComputeExpectation(
    hipdensitymatHandle_t handle,
    hipdensitymatOperator_t op,
    double t,
    int32_t numParams,
    const double* params,
    hipdensitymatState_t state,
    hipdensitymatWorkspaceDescriptor_t workspace,
    hipdensitymatComplexDouble_t* expectation);

#ifdef __cplusplus
}
#endif

#endif /* HIPDENSITYMAT_ACTION_H */
