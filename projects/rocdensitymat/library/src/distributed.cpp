/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Distributed (multi-GPU multi-node) configuration. v0.1.0 is single-rank
 * only and every entry point returns ROCDENSITYMAT_STATUS_NOT_SUPPORTED
 * with a clear ABI surface so v0.1.0 callers do not break when the v0.2
 * implementation lands.
 * ************************************************************************ */

#include "rocdensitymat_descriptors.hpp"

extern "C" {

rocdensitymat_status rocdensitymat_reset_distributed_configuration(
    rocdensitymat_handle handle,
    rocdensitymat_distributed_provider provider,
    const void* /*comm_ptr*/,
    size_t /*comm_size_bytes*/)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    if(provider == ROCDENSITYMAT_DISTRIBUTED_PROVIDER_NONE)
    {
        // Single-rank no-op: explicitly accepted as a degenerate config so
        // user code that always issues this on startup does not break.
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;
}

rocdensitymat_status rocdensitymat_get_num_ranks(
    rocdensitymat_handle handle, int32_t* num_ranks)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(num_ranks);
    *num_ranks = 1;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_get_proc_rank(
    rocdensitymat_handle handle, int32_t* rank)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(rank);
    *rank = 0;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

} // extern "C"
