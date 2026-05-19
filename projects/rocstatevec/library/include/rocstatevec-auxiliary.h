/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Reverse-engineered from cuStateVec public API documentation
 * (cuQuantum 24.11 / cuStateVec 1.7.x); clean-room implementation.
 * ************************************************************************ */

/*! \file
 *  \brief Library context, version, error, logger, and memory-handler API.
 */

#ifndef ROCSTATEVEC_AUXILIARY_H
#define ROCSTATEVEC_AUXILIARY_H

#include "rocstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Create a new rocSTATEVEC library context handle.
 *
 *  Bijection: \p custatevecCreate. */
rocstatevec_status rocstatevec_create_handle(rocstatevec_handle* handle);

/*! \brief Destroy a rocSTATEVEC library context handle.
 *
 *  Bijection: \p custatevecDestroy. */
rocstatevec_status rocstatevec_destroy_handle(rocstatevec_handle handle);

/*! \brief Return the rocSTATEVEC library version (major*1000 + minor*100 + patch).
 *
 *  Bijection: \p custatevecGetVersion. */
rocstatevec_status rocstatevec_get_version(rocstatevec_handle handle,
                                           int* version);

/*! \brief Return a single library property value.
 *
 *  Bijection: \p custatevecGetProperty. */
rocstatevec_status rocstatevec_get_property(rocstatevec_library_property_type type,
                                            int* value);

/*! \brief Bind a HIP stream to a library context. Subsequent operations are
 *         enqueued on this stream. Bijection: \p custatevecSetStream. */
rocstatevec_status rocstatevec_set_stream(rocstatevec_handle handle,
                                          hipStream_t stream);

/*! \brief Retrieve the HIP stream currently bound to a library context.
 *
 *  Bijection: \p custatevecGetStream. */
rocstatevec_status rocstatevec_get_stream(rocstatevec_handle handle,
                                          hipStream_t* stream);

/*! \brief Return a stable human-readable name for a status enum value.
 *
 *  Bijection: \p custatevecGetErrorName. */
const char* rocstatevec_get_error_name(rocstatevec_status status);

/*! \brief Return a stable human-readable description of a status enum value.
 *
 *  Bijection: \p custatevecGetErrorString. */
const char* rocstatevec_get_error_string(rocstatevec_status status);

/*! \brief Install a custom device-memory handler on a library context.
 *
 *  Bijection: \p custatevecSetDeviceMemHandler. */
rocstatevec_status rocstatevec_set_device_mem_handler(
    rocstatevec_handle handle,
    const rocstatevec_device_mem_handler_t* mem_handler);

/*! \brief Read the device-memory handler currently installed on a library
 *         context. Bijection: \p custatevecGetDeviceMemHandler. */
rocstatevec_status rocstatevec_get_device_mem_handler(
    rocstatevec_handle handle,
    rocstatevec_device_mem_handler_t* mem_handler);

/*! \brief Install a logger callback. Bijection: \p custatevecLoggerSetCallback. */
rocstatevec_status rocstatevec_logger_set_callback(rocstatevec_logger_callback_t cb);

/*! \brief Install a logger callback that receives a user-data pointer.
 *  Bijection: \p custatevecLoggerSetCallbackData. */
rocstatevec_status rocstatevec_logger_set_callback_data(
    rocstatevec_logger_callback_data_t cb,
    void* user_data);

/*! \brief Direct logger output to an open `FILE*`.
 *  Bijection: \p custatevecLoggerSetFile. */
rocstatevec_status rocstatevec_logger_set_file(void* file);

/*! \brief Direct logger output to a path on disk.
 *  Bijection: \p custatevecLoggerOpenFile. */
rocstatevec_status rocstatevec_logger_open_file(const char* path);

/*! \brief Set the logger verbosity level (0..5).
 *  Bijection: \p custatevecLoggerSetLevel. */
rocstatevec_status rocstatevec_logger_set_level(int32_t level);

/*! \brief Set the logger event mask. Bijection: \p custatevecLoggerSetMask. */
rocstatevec_status rocstatevec_logger_set_mask(int32_t mask);

/*! \brief Disable the logger. Bijection: \p custatevecLoggerForceDisable. */
rocstatevec_status rocstatevec_logger_force_disable(void);

#ifdef __cplusplus
}
#endif

#endif /* ROCSTATEVEC_AUXILIARY_H */
