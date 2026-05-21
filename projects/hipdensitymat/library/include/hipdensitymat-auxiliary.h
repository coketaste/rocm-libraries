/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPDENSITYMAT_AUXILIARY_H
#define HIPDENSITYMAT_AUXILIARY_H

#include "hipdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t hipdensitymatGetVersion(void);

hipdensitymatStatus_t hipdensitymatGetProperty(
    hipdensitymatLibraryPropertyType_t type, int32_t* value);

const char* hipdensitymatGetErrorName(hipdensitymatStatus_t status);
const char* hipdensitymatGetErrorString(hipdensitymatStatus_t status);

hipdensitymatStatus_t hipdensitymatSetDeviceMemHandler(
    hipdensitymatHandle_t handle,
    const hipdensitymatDeviceMemHandler_t* memHandler);

hipdensitymatStatus_t hipdensitymatGetDeviceMemHandler(
    hipdensitymatHandle_t handle,
    hipdensitymatDeviceMemHandler_t* memHandler);

hipdensitymatStatus_t hipdensitymatLoggerSetCallback(
    hipdensitymatLoggerCallback_t cb);
hipdensitymatStatus_t hipdensitymatLoggerSetCallbackData(
    hipdensitymatLoggerCallbackData_t cb, void* userData);
hipdensitymatStatus_t hipdensitymatLoggerSetFile(void* file);
hipdensitymatStatus_t hipdensitymatLoggerOpenFile(const char* path);
hipdensitymatStatus_t hipdensitymatLoggerSetLevel(int32_t level);
hipdensitymatStatus_t hipdensitymatLoggerSetMask(int32_t mask);
hipdensitymatStatus_t hipdensitymatLoggerForceDisable(void);

#ifdef __cplusplus
}
#endif

#endif /* HIPDENSITYMAT_AUXILIARY_H */
