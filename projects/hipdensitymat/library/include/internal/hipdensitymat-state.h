/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPDENSITYMAT_STATE_H
#define HIPDENSITYMAT_STATE_H

#include "../hipdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipdensitymatStatus_t hipdensitymatCreateState(
    hipdensitymatHandle_t handle,
    hipdensitymatStatePurity_t purity,
    int32_t numSpaceModes,
    const int64_t* spaceShape,
    int64_t batchSize,
    hipdensitymatDataType_t dataType,
    hipdensitymatState_t* state);

hipdensitymatStatus_t hipdensitymatDestroyState(hipdensitymatState_t state);

hipdensitymatStatus_t hipdensitymatStateGetNumComponents(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    int32_t* numComponents);

hipdensitymatStatus_t hipdensitymatStateGetComponentInfo(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    int32_t componentId,
    int32_t* numModes, int64_t* modeExtents,
    size_t* componentSizeBytes);

hipdensitymatStatus_t hipdensitymatStateAttachComponentBuffer(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    int32_t componentId, void* componentBuffer, size_t componentBufferSize);

hipdensitymatStatus_t hipdensitymatStateInitializeZero(
    hipdensitymatHandle_t handle, hipdensitymatState_t state);
hipdensitymatStatus_t hipdensitymatStateInitializeUniform(
    hipdensitymatHandle_t handle, hipdensitymatState_t state);
hipdensitymatStatus_t hipdensitymatStateInitializeBasis(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    const int64_t* basisIndices);

hipdensitymatStatus_t hipdensitymatStateComputeNorm(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    hipdensitymatWorkspaceDescriptor_t workspace, double* norm);

hipdensitymatStatus_t hipdensitymatStateComputeTrace(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    hipdensitymatWorkspaceDescriptor_t workspace,
    hipdensitymatComplexDouble_t* trace);

hipdensitymatStatus_t hipdensitymatStateComputeOverlap(
    hipdensitymatHandle_t handle,
    hipdensitymatState_t lhs, hipdensitymatState_t rhs,
    hipdensitymatWorkspaceDescriptor_t workspace,
    hipdensitymatComplexDouble_t* overlap);

#ifdef __cplusplus
}
#endif

#endif /* HIPDENSITYMAT_STATE_H */
