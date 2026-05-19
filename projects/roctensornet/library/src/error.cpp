/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "roctensornet_internal.hpp"

extern "C" {

const char* roctensornet_get_error_string(roctensornet_status s)
{
    switch(s)
    {
    case ROCTENSORNET_STATUS_SUCCESS:                  return "ROCTENSORNET_STATUS_SUCCESS";
    case ROCTENSORNET_STATUS_NOT_INITIALIZED:          return "ROCTENSORNET_STATUS_NOT_INITIALIZED";
    case ROCTENSORNET_STATUS_ALLOC_FAILED:             return "ROCTENSORNET_STATUS_ALLOC_FAILED";
    case ROCTENSORNET_STATUS_INVALID_VALUE:            return "ROCTENSORNET_STATUS_INVALID_VALUE";
    case ROCTENSORNET_STATUS_ARCH_MISMATCH:            return "ROCTENSORNET_STATUS_ARCH_MISMATCH";
    case ROCTENSORNET_STATUS_EXECUTION_FAILED:         return "ROCTENSORNET_STATUS_EXECUTION_FAILED";
    case ROCTENSORNET_STATUS_INTERNAL_ERROR:           return "ROCTENSORNET_STATUS_INTERNAL_ERROR";
    case ROCTENSORNET_STATUS_NOT_SUPPORTED:            return "ROCTENSORNET_STATUS_NOT_SUPPORTED";
    case ROCTENSORNET_STATUS_LICENSE_ERROR:            return "ROCTENSORNET_STATUS_LICENSE_ERROR";
    case ROCTENSORNET_STATUS_DEVICE_ALLOCATOR_ERROR:   return "ROCTENSORNET_STATUS_DEVICE_ALLOCATOR_ERROR";
    case ROCTENSORNET_STATUS_IO_ERROR:                 return "ROCTENSORNET_STATUS_IO_ERROR";
    case ROCTENSORNET_STATUS_INSUFFICIENT_WORKSPACE:   return "ROCTENSORNET_STATUS_INSUFFICIENT_WORKSPACE";
    case ROCTENSORNET_STATUS_INSUFFICIENT_DRIVER:      return "ROCTENSORNET_STATUS_INSUFFICIENT_DRIVER";
    case ROCTENSORNET_STATUS_INTERRUPTED:              return "ROCTENSORNET_STATUS_INTERRUPTED";
    }
    return "ROCTENSORNET_STATUS_UNKNOWN";
}

} // extern "C"
