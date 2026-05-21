/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Distributed (multi-GPU/multi-node) configuration. v0.1.0 of
 * hipDENSITYMAT exposes this surface but every entry point returns
 * HIPDENSITYMAT_STATUS_NOT_SUPPORTED at runtime.
 * ************************************************************************ */

#ifndef HIPDENSITYMAT_DISTRIBUTED_H
#define HIPDENSITYMAT_DISTRIBUTED_H

#include "../hipdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipdensitymatStatus_t hipdensitymatResetDistributedConfiguration(
    hipdensitymatHandle_t handle,
    hipdensitymatDistributedProvider_t provider,
    const void* commPtr, size_t commSize);

hipdensitymatStatus_t hipdensitymatGetNumRanks(
    hipdensitymatHandle_t handle, int32_t* numRanks);

hipdensitymatStatus_t hipdensitymatGetProcRank(
    hipdensitymatHandle_t handle, int32_t* procRank);

#ifdef __cplusplus
}
#endif

#endif /* HIPDENSITYMAT_DISTRIBUTED_H */
