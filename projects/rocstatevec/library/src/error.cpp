/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Status-to-string mappings.
 * ************************************************************************ */

#include "rocstatevec_internal.hpp"

extern "C" const char* rocstatevec_get_error_name(rocstatevec_status status)
{
    switch(status)
    {
    case ROCSTATEVEC_STATUS_SUCCESS:                  return "ROCSTATEVEC_STATUS_SUCCESS";
    case ROCSTATEVEC_STATUS_NOT_INITIALIZED:          return "ROCSTATEVEC_STATUS_NOT_INITIALIZED";
    case ROCSTATEVEC_STATUS_ALLOC_FAILED:             return "ROCSTATEVEC_STATUS_ALLOC_FAILED";
    case ROCSTATEVEC_STATUS_INVALID_VALUE:            return "ROCSTATEVEC_STATUS_INVALID_VALUE";
    case ROCSTATEVEC_STATUS_ARCH_MISMATCH:            return "ROCSTATEVEC_STATUS_ARCH_MISMATCH";
    case ROCSTATEVEC_STATUS_EXECUTION_FAILED:         return "ROCSTATEVEC_STATUS_EXECUTION_FAILED";
    case ROCSTATEVEC_STATUS_INTERNAL_ERROR:           return "ROCSTATEVEC_STATUS_INTERNAL_ERROR";
    case ROCSTATEVEC_STATUS_NOT_SUPPORTED:            return "ROCSTATEVEC_STATUS_NOT_SUPPORTED";
    case ROCSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE:   return "ROCSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE";
    case ROCSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED: return "ROCSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED";
    case ROCSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR:      return "ROCSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR";
    case ROCSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR:   return "ROCSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR";
    case ROCSTATEVEC_STATUS_COMMUNICATOR_ERROR:       return "ROCSTATEVEC_STATUS_COMMUNICATOR_ERROR";
    case ROCSTATEVEC_STATUS_LOADING_LIBRARY_FAILED:   return "ROCSTATEVEC_STATUS_LOADING_LIBRARY_FAILED";
    default:                                          return "ROCSTATEVEC_STATUS_UNKNOWN";
    }
}

extern "C" const char* rocstatevec_get_error_string(rocstatevec_status status)
{
    switch(status)
    {
    case ROCSTATEVEC_STATUS_SUCCESS:                  return "the operation completed successfully.";
    case ROCSTATEVEC_STATUS_NOT_INITIALIZED:          return "the library has not been initialized or a null handle was passed.";
    case ROCSTATEVEC_STATUS_ALLOC_FAILED:             return "a host or device allocation failed.";
    case ROCSTATEVEC_STATUS_INVALID_VALUE:            return "an invalid argument was passed.";
    case ROCSTATEVEC_STATUS_ARCH_MISMATCH:            return "the targeted GPU architecture is not supported.";
    case ROCSTATEVEC_STATUS_EXECUTION_FAILED:         return "a kernel launch or HIP runtime call failed.";
    case ROCSTATEVEC_STATUS_INTERNAL_ERROR:           return "an unexpected internal error occurred.";
    case ROCSTATEVEC_STATUS_NOT_SUPPORTED:            return "the operation is not supported by this library version.";
    case ROCSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE:   return "the workspace buffer supplied is too small.";
    case ROCSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED: return "the sampler was sampled before preprocess was called.";
    case ROCSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR:      return "no device-memory handler is bound to this handle.";
    case ROCSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR:   return "the user-supplied device-memory handler returned an error.";
    case ROCSTATEVEC_STATUS_COMMUNICATOR_ERROR:       return "a communicator-level operation failed.";
    case ROCSTATEVEC_STATUS_LOADING_LIBRARY_FAILED:   return "the runtime failed to load a required shared library.";
    default:                                          return "an unknown rocSTATEVEC error occurred.";
    }
}
