/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPSTATEVEC_AUXILIARY_H
#define HIPSTATEVEC_AUXILIARY_H

#include "hipstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipstatevecStatus_t hipstatevecCreate(hipstatevecHandle_t* handle);
hipstatevecStatus_t hipstatevecDestroy(hipstatevecHandle_t handle);
hipstatevecStatus_t hipstatevecGetVersion(hipstatevecHandle_t handle, int* version);
hipstatevecStatus_t hipstatevecGetProperty(hipstatevecLibraryPropertyType_t type, int* value);
hipstatevecStatus_t hipstatevecSetStream(hipstatevecHandle_t handle, hipStream_t stream);
hipstatevecStatus_t hipstatevecGetStream(hipstatevecHandle_t handle, hipStream_t* stream);

const char* hipstatevecGetErrorName(hipstatevecStatus_t status);
const char* hipstatevecGetErrorString(hipstatevecStatus_t status);

hipstatevecStatus_t hipstatevecSetDeviceMemHandler(
    hipstatevecHandle_t handle,
    const hipstatevecDeviceMemHandler_t* mem_handler);
hipstatevecStatus_t hipstatevecGetDeviceMemHandler(
    hipstatevecHandle_t handle,
    hipstatevecDeviceMemHandler_t* mem_handler);

hipstatevecStatus_t hipstatevecLoggerSetCallback(hipstatevecLoggerCallback_t cb);
hipstatevecStatus_t hipstatevecLoggerSetCallbackData(hipstatevecLoggerCallbackData_t cb, void* user_data);
hipstatevecStatus_t hipstatevecLoggerSetFile(void* file);
hipstatevecStatus_t hipstatevecLoggerOpenFile(const char* path);
hipstatevecStatus_t hipstatevecLoggerSetLevel(int32_t level);
hipstatevecStatus_t hipstatevecLoggerSetMask(int32_t mask);
hipstatevecStatus_t hipstatevecLoggerForceDisable(void);

#ifdef __cplusplus
}
#endif

#endif /* HIPSTATEVEC_AUXILIARY_H */
