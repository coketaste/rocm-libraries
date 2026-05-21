/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "rocdensitymat_internal.hpp"

extern "C" {

const char* rocdensitymat_get_error_name(rocdensitymat_status status)
{
    switch(status)
    {
    case ROCDENSITYMAT_STATUS_SUCCESS:                 return "ROCDENSITYMAT_STATUS_SUCCESS";
    case ROCDENSITYMAT_STATUS_NOT_INITIALIZED:         return "ROCDENSITYMAT_STATUS_NOT_INITIALIZED";
    case ROCDENSITYMAT_STATUS_ALLOC_FAILED:            return "ROCDENSITYMAT_STATUS_ALLOC_FAILED";
    case ROCDENSITYMAT_STATUS_INVALID_VALUE:           return "ROCDENSITYMAT_STATUS_INVALID_VALUE";
    case ROCDENSITYMAT_STATUS_ARCH_MISMATCH:           return "ROCDENSITYMAT_STATUS_ARCH_MISMATCH";
    case ROCDENSITYMAT_STATUS_EXECUTION_FAILED:        return "ROCDENSITYMAT_STATUS_EXECUTION_FAILED";
    case ROCDENSITYMAT_STATUS_INTERNAL_ERROR:          return "ROCDENSITYMAT_STATUS_INTERNAL_ERROR";
    case ROCDENSITYMAT_STATUS_NOT_SUPPORTED:           return "ROCDENSITYMAT_STATUS_NOT_SUPPORTED";
    case ROCDENSITYMAT_STATUS_CALLBACK_ERROR:          return "ROCDENSITYMAT_STATUS_CALLBACK_ERROR";
    case ROCDENSITYMAT_STATUS_BLAS_ERROR:              return "ROCDENSITYMAT_STATUS_BLAS_ERROR";
    case ROCDENSITYMAT_STATUS_HIP_ERROR:               return "ROCDENSITYMAT_STATUS_HIP_ERROR";
    case ROCDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE:  return "ROCDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE";
    case ROCDENSITYMAT_STATUS_INSUFFICIENT_DRIVER:     return "ROCDENSITYMAT_STATUS_INSUFFICIENT_DRIVER";
    case ROCDENSITYMAT_STATUS_IO_ERROR:                return "ROCDENSITYMAT_STATUS_IO_ERROR";
    case ROCDENSITYMAT_STATUS_NO_DEVICE_ALLOCATOR:     return "ROCDENSITYMAT_STATUS_NO_DEVICE_ALLOCATOR";
    case ROCDENSITYMAT_STATUS_DEVICE_ALLOCATOR_ERROR:  return "ROCDENSITYMAT_STATUS_DEVICE_ALLOCATOR_ERROR";
    case ROCDENSITYMAT_STATUS_DISTRIBUTED_FAILURE:     return "ROCDENSITYMAT_STATUS_DISTRIBUTED_FAILURE";
    case ROCDENSITYMAT_STATUS_INTERRUPTED:             return "ROCDENSITYMAT_STATUS_INTERRUPTED";
    case ROCDENSITYMAT_STATUS_DRIVER_VERSION_MISMATCH: return "ROCDENSITYMAT_STATUS_DRIVER_VERSION_MISMATCH";
    case ROCDENSITYMAT_STATUS_MAX_VALUE:               return "ROCDENSITYMAT_STATUS_MAX_VALUE";
    }
    return "ROCDENSITYMAT_STATUS_UNKNOWN";
}

const char* rocdensitymat_get_error_string(rocdensitymat_status status)
{
    switch(status)
    {
    case ROCDENSITYMAT_STATUS_SUCCESS:
        return "Operation completed successfully.";
    case ROCDENSITYMAT_STATUS_NOT_INITIALIZED:
        return "The library context handle was not properly initialized.";
    case ROCDENSITYMAT_STATUS_ALLOC_FAILED:
        return "A host or device allocation failed.";
    case ROCDENSITYMAT_STATUS_INVALID_VALUE:
        return "An argument has an unsupported value or layout.";
    case ROCDENSITYMAT_STATUS_ARCH_MISMATCH:
        return "The current GPU architecture does not satisfy the kernel requirements.";
    case ROCDENSITYMAT_STATUS_EXECUTION_FAILED:
        return "Kernel execution on the device failed.";
    case ROCDENSITYMAT_STATUS_INTERNAL_ERROR:
        return "An internal invariant of the library was violated.";
    case ROCDENSITYMAT_STATUS_NOT_SUPPORTED:
        return "The requested operation is not implemented in the current rocDENSITYMAT release.";
    case ROCDENSITYMAT_STATUS_CALLBACK_ERROR:
        return "A user-supplied callback returned an error.";
    case ROCDENSITYMAT_STATUS_BLAS_ERROR:
        return "The underlying BLAS library reported a failure.";
    case ROCDENSITYMAT_STATUS_HIP_ERROR:
        return "A HIP runtime call reported a failure.";
    case ROCDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE:
        return "The supplied workspace buffer is smaller than the prepared minimum.";
    case ROCDENSITYMAT_STATUS_INSUFFICIENT_DRIVER:
        return "The installed AMD GPU driver is too old for the current rocDENSITYMAT release.";
    case ROCDENSITYMAT_STATUS_IO_ERROR:
        return "Reading from or writing to a file failed.";
    case ROCDENSITYMAT_STATUS_NO_DEVICE_ALLOCATOR:
        return "An operation that needs a device allocator was issued on a handle without one.";
    case ROCDENSITYMAT_STATUS_DEVICE_ALLOCATOR_ERROR:
        return "The user-supplied device allocator returned an error.";
    case ROCDENSITYMAT_STATUS_DISTRIBUTED_FAILURE:
        return "A distributed (multi-rank) operation failed; v0.1.0 is single-rank only.";
    case ROCDENSITYMAT_STATUS_INTERRUPTED:
        return "The operation was interrupted.";
    case ROCDENSITYMAT_STATUS_DRIVER_VERSION_MISMATCH:
        return "Loaded driver and library versions are incompatible.";
    case ROCDENSITYMAT_STATUS_MAX_VALUE:
        return "Sentinel value; not a real status.";
    }
    return "Unknown rocDENSITYMAT status.";
}

} // extern "C"
