/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "rocdensitymat_descriptors.hpp"

extern "C" {

rocdensitymat_status rocdensitymat_create_workspace(
    rocdensitymat_handle handle,
    rocdensitymat_workspace_descriptor* workspace)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(workspace);

    auto* ws = new(std::nothrow) _rocdensitymat_workspace_descriptor();
    if(ws == nullptr) return ROCDENSITYMAT_STATUS_ALLOC_FAILED;
    *workspace = ws;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_destroy_workspace(
    rocdensitymat_workspace_descriptor workspace)
{
    if(workspace == nullptr) return ROCDENSITYMAT_STATUS_SUCCESS;
    delete workspace;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_workspace_get_memory_size(
    rocdensitymat_handle handle,
    rocdensitymat_workspace_descriptor workspace,
    rocdensitymat_memspace mem_space,
    rocdensitymat_workspace_kind kind,
    size_t* memory_size_bytes)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(workspace);
    ROCDENSITYMAT_CHECK_PTR(memory_size_bytes);

    if(kind == ROCDENSITYMAT_WORKSPACE_CACHE)
    {
        // v0.1.0 has no cross-call cache, so the cache request is a no-op.
        *memory_size_bytes = 0;
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    if(kind != ROCDENSITYMAT_WORKSPACE_SCRATCH)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;

    if(mem_space == ROCDENSITYMAT_MEMSPACE_DEVICE)
    {
        *memory_size_bytes = workspace->required_device_scratch_bytes;
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    if(mem_space == ROCDENSITYMAT_MEMSPACE_HOST)
    {
        // v0.1.0 stepper / apply paths run all reductions on-device, so the
        // host scratch requirement is always 0.
        *memory_size_bytes = 0;
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    return ROCDENSITYMAT_STATUS_INVALID_VALUE;
}

rocdensitymat_status rocdensitymat_workspace_set_memory(
    rocdensitymat_handle handle,
    rocdensitymat_workspace_descriptor workspace,
    rocdensitymat_memspace mem_space,
    rocdensitymat_workspace_kind kind,
    void* memory_ptr,
    size_t memory_size_bytes)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(workspace);

    if(kind == ROCDENSITYMAT_WORKSPACE_CACHE)
        return ROCDENSITYMAT_STATUS_SUCCESS;
    if(kind != ROCDENSITYMAT_WORKSPACE_SCRATCH)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;

    if(mem_space == ROCDENSITYMAT_MEMSPACE_DEVICE)
    {
        if(memory_size_bytes < workspace->required_device_scratch_bytes)
            return ROCDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE;
        workspace->device_scratch_ptr   = memory_ptr;
        workspace->device_scratch_bytes = memory_size_bytes;
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    if(mem_space == ROCDENSITYMAT_MEMSPACE_HOST)
    {
        workspace->host_scratch_ptr   = memory_ptr;
        workspace->host_scratch_bytes = memory_size_bytes;
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    return ROCDENSITYMAT_STATUS_INVALID_VALUE;
}

rocdensitymat_status rocdensitymat_workspace_get_memory(
    rocdensitymat_handle handle,
    rocdensitymat_workspace_descriptor workspace,
    rocdensitymat_memspace mem_space,
    rocdensitymat_workspace_kind kind,
    void** memory_ptr,
    size_t* memory_size_bytes)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(workspace);
    ROCDENSITYMAT_CHECK_PTR(memory_ptr);
    ROCDENSITYMAT_CHECK_PTR(memory_size_bytes);

    if(kind == ROCDENSITYMAT_WORKSPACE_CACHE)
    {
        *memory_ptr        = nullptr;
        *memory_size_bytes = 0;
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    if(kind != ROCDENSITYMAT_WORKSPACE_SCRATCH)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;

    if(mem_space == ROCDENSITYMAT_MEMSPACE_DEVICE)
    {
        *memory_ptr        = workspace->device_scratch_ptr;
        *memory_size_bytes = workspace->device_scratch_bytes;
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    if(mem_space == ROCDENSITYMAT_MEMSPACE_HOST)
    {
        *memory_ptr        = workspace->host_scratch_ptr;
        *memory_size_bytes = workspace->host_scratch_bytes;
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    return ROCDENSITYMAT_STATUS_INVALID_VALUE;
}

} // extern "C"
