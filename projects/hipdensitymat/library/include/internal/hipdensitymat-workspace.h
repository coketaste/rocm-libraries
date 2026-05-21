/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPDENSITYMAT_WORKSPACE_H
#define HIPDENSITYMAT_WORKSPACE_H

#include "../hipdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipdensitymatStatus_t hipdensitymatCreateWorkspace(
    hipdensitymatHandle_t handle, hipdensitymatWorkspaceDescriptor_t* workspace);
hipdensitymatStatus_t hipdensitymatDestroyWorkspace(
    hipdensitymatWorkspaceDescriptor_t workspace);

hipdensitymatStatus_t hipdensitymatWorkspaceGetMemorySize(
    hipdensitymatHandle_t handle,
    hipdensitymatWorkspaceDescriptor_t workspace,
    hipdensitymatMemspace_t memSpace,
    hipdensitymatWorkspaceKind_t kind,
    size_t* memorySizeBytes);

hipdensitymatStatus_t hipdensitymatWorkspaceSetMemory(
    hipdensitymatHandle_t handle,
    hipdensitymatWorkspaceDescriptor_t workspace,
    hipdensitymatMemspace_t memSpace,
    hipdensitymatWorkspaceKind_t kind,
    void* memoryPtr, size_t memorySizeBytes);

hipdensitymatStatus_t hipdensitymatWorkspaceGetMemory(
    hipdensitymatHandle_t handle,
    hipdensitymatWorkspaceDescriptor_t workspace,
    hipdensitymatMemspace_t memSpace,
    hipdensitymatWorkspaceKind_t kind,
    void** memoryPtr, size_t* memorySizeBytes);

#ifdef __cplusplus
}
#endif

#endif /* HIPDENSITYMAT_WORKSPACE_H */
