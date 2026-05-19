/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Library, handle, logger, and device-memory-handler entry points.
 * ************************************************************************ */

#ifndef ROCTENSORNET_AUXILIARY_H
#define ROCTENSORNET_AUXILIARY_H

#include "roctensornet-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Library context lifecycle ---- */
roctensornet_status roctensornet_create   (roctensornet_handle* handle);
roctensornet_status roctensornet_destroy  (roctensornet_handle  handle);

/* ---- Version / driver introspection ---- */
roctensornet_status roctensornet_get_version             (int* version);
roctensornet_status roctensornet_get_hip_runtime_version (int* version);

/* ---- Error string ---- */
const char* roctensornet_get_error_string (roctensornet_status status);

/* ---- Stream management ---- */
roctensornet_status roctensornet_set_stream (roctensornet_handle handle, hipStream_t stream);
roctensornet_status roctensornet_get_stream (roctensornet_handle handle, hipStream_t* stream);

/* ---- Device memory handler ---- */
roctensornet_status
roctensornet_set_device_mem_handler (roctensornet_handle handle,
                                     const roctensornet_device_mem_handler_t* dev_mem_handler);
roctensornet_status
roctensornet_get_device_mem_handler (roctensornet_handle handle,
                                     roctensornet_device_mem_handler_t* dev_mem_handler);

/* ---- Logger ---- */
roctensornet_status roctensornet_logger_set_callback      (roctensornet_logger_callback_t      callback);
roctensornet_status roctensornet_logger_set_callback_data (roctensornet_logger_callback_data_t callback, void* user);
roctensornet_status roctensornet_logger_set_file          (void* file);
roctensornet_status roctensornet_logger_open_file         (const char* file_name);
roctensornet_status roctensornet_logger_set_level         (int32_t level);
roctensornet_status roctensornet_logger_set_mask          (int32_t mask);
roctensornet_status roctensornet_logger_force_disable     (void);

#ifdef __cplusplus
}
#endif

#endif /* ROCTENSORNET_AUXILIARY_H */
