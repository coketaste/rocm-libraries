/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPDENSITYMAT_HANDLE_H
#define HIPDENSITYMAT_HANDLE_H

#include "../hipdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipdensitymatStatus_t hipdensitymatCreate(hipdensitymatHandle_t* handle);
hipdensitymatStatus_t hipdensitymatDestroy(hipdensitymatHandle_t handle);
hipdensitymatStatus_t hipdensitymatResetRandomSeed(
    hipdensitymatHandle_t handle, uint32_t seed);
hipdensitymatStatus_t hipdensitymatSetStream(hipdensitymatHandle_t handle,
                                             hipStream_t stream);
hipdensitymatStatus_t hipdensitymatGetStream(hipdensitymatHandle_t handle,
                                             hipStream_t* stream);

#ifdef __cplusplus
}
#endif

#endif /* HIPDENSITYMAT_HANDLE_H */
