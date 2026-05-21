/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Reverse-engineered from cuDensityMat public API documentation
 * (cuQuantum 24.11 / cuDensityMat 0.1.0); clean-room implementation.
 * ************************************************************************ */

/*! \file
 *  \brief Library context, version, error, logger, and memory-handler API.
 */

#ifndef ROCDENSITYMAT_AUXILIARY_H
#define ROCDENSITYMAT_AUXILIARY_H

#include "rocdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Return the rocDENSITYMAT library version (major*1000 + minor*100 + patch).
 *
 *  Bijection: \p cudensitymatGetVersion. */
size_t rocdensitymat_get_version(void);

/*! \brief Return a single library property value.
 *
 *  Bijection: \p cudensitymatGetProperty. */
rocdensitymat_status rocdensitymat_get_property(
    rocdensitymat_library_property_type type, int32_t* value);

/*! \brief Return a stable human-readable name for a status enum value.
 *
 *  Bijection: \p cudensitymatGetErrorName. */
const char* rocdensitymat_get_error_name(rocdensitymat_status status);

/*! \brief Return a stable human-readable description of a status enum value.
 *
 *  Bijection: \p cudensitymatGetErrorString. */
const char* rocdensitymat_get_error_string(rocdensitymat_status status);

/*! \brief Install a custom device-memory handler on a library context.
 *
 *  Bijection: \p cudensitymatSetDeviceMemHandler. */
rocdensitymat_status rocdensitymat_set_device_mem_handler(
    rocdensitymat_handle handle,
    const rocdensitymat_device_mem_handler_t* mem_handler);

/*! \brief Read the device-memory handler currently installed on a library
 *         context. Bijection: \p cudensitymatGetDeviceMemHandler. */
rocdensitymat_status rocdensitymat_get_device_mem_handler(
    rocdensitymat_handle handle,
    rocdensitymat_device_mem_handler_t* mem_handler);

/*! \brief Install a logger callback. Bijection: \p cudensitymatLoggerSetCallback. */
rocdensitymat_status rocdensitymat_logger_set_callback(
    rocdensitymat_logger_callback_t cb);

/*! \brief Install a logger callback that receives a user-data pointer.
 *  Bijection: \p cudensitymatLoggerSetCallbackData. */
rocdensitymat_status rocdensitymat_logger_set_callback_data(
    rocdensitymat_logger_callback_data_t cb, void* user_data);

/*! \brief Direct logger output to an open `FILE*`.
 *  Bijection: \p cudensitymatLoggerSetFile. */
rocdensitymat_status rocdensitymat_logger_set_file(void* file);

/*! \brief Direct logger output to a path on disk.
 *  Bijection: \p cudensitymatLoggerOpenFile. */
rocdensitymat_status rocdensitymat_logger_open_file(const char* path);

/*! \brief Set the logger verbosity level (0..5).
 *  Bijection: \p cudensitymatLoggerSetLevel. */
rocdensitymat_status rocdensitymat_logger_set_level(int32_t level);

/*! \brief Set the logger event mask. Bijection: \p cudensitymatLoggerSetMask. */
rocdensitymat_status rocdensitymat_logger_set_mask(int32_t mask);

/*! \brief Disable the logger. Bijection: \p cudensitymatLoggerForceDisable. */
rocdensitymat_status rocdensitymat_logger_force_disable(void);

#ifdef __cplusplus
}
#endif

#endif /* ROCDENSITYMAT_AUXILIARY_H */
