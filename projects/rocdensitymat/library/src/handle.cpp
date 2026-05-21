/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "rocdensitymat_descriptors.hpp"

extern "C" {

rocdensitymat_status rocdensitymat_create(rocdensitymat_handle* handle)
{
    if(handle == nullptr) return ROCDENSITYMAT_STATUS_INVALID_VALUE;

    auto* h     = new(std::nothrow) _rocdensitymat_handle();
    if(h == nullptr) return ROCDENSITYMAT_STATUS_ALLOC_FAILED;

    int dev = 0;
    if(hipGetDevice(&dev) == hipSuccess) h->device_id = dev;
    h->stream = nullptr;
    h->seed   = 0;
    *handle   = h;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_destroy(rocdensitymat_handle handle)
{
    if(handle == nullptr) return ROCDENSITYMAT_STATUS_SUCCESS;
    delete handle;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_reset_random_seed(
    rocdensitymat_handle handle, uint32_t seed)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    handle->seed = seed;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_set_stream(rocdensitymat_handle handle,
                                              hipStream_t stream)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    handle->stream = stream;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_get_stream(rocdensitymat_handle handle,
                                              hipStream_t* stream)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(stream);
    *stream = handle->stream;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_set_device_mem_handler(
    rocdensitymat_handle handle,
    const rocdensitymat_device_mem_handler_t* mem_handler)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    if(mem_handler == nullptr)
    {
        handle->has_mem = false;
        std::memset(&handle->mem, 0, sizeof(handle->mem));
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    if(mem_handler->device_alloc == nullptr || mem_handler->device_free == nullptr)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    handle->mem     = *mem_handler;
    handle->has_mem = true;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_get_device_mem_handler(
    rocdensitymat_handle handle,
    rocdensitymat_device_mem_handler_t* mem_handler)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(mem_handler);
    if(!handle->has_mem) return ROCDENSITYMAT_STATUS_NO_DEVICE_ALLOCATOR;
    *mem_handler = handle->mem;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

} // extern "C"
