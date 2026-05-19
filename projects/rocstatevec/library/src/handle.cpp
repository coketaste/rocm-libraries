/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Library context: handle lifetime, stream binding, version, device-memory
 * handler, library properties.
 * ************************************************************************ */

#include "rocstatevec_internal.hpp"

using namespace rocstatevec;

extern "C" rocstatevec_status rocstatevec_create_handle(rocstatevec_handle* out)
{
    ROCSTATEVEC_CHECK_PTR(out);
    auto* h = new(std::nothrow) handle();
    if(h == nullptr) return ROCSTATEVEC_STATUS_ALLOC_FAILED;

    int dev = 0;
    if(hipGetDevice(&dev) != hipSuccess)
    {
        delete h;
        return ROCSTATEVEC_STATUS_EXECUTION_FAILED;
    }
    h->device_id = dev;
    *out = reinterpret_cast<rocstatevec_handle>(h);
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_destroy_handle(rocstatevec_handle handle_in)
{
    if(handle_in == nullptr) return ROCSTATEVEC_STATUS_SUCCESS;
    delete reinterpret_cast<handle*>(handle_in);
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_get_version(rocstatevec_handle h, int* version)
{
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(version);
    *version = ROCSTATEVEC_VERSION_MAJOR * 1000
             + ROCSTATEVEC_VERSION_MINOR *  100
             + ROCSTATEVEC_VERSION_PATCH;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_get_property(rocstatevec_library_property_type type, int* value)
{
    ROCSTATEVEC_CHECK_PTR(value);
    switch(type)
    {
    case ROCSTATEVEC_PROPERTY_MAJOR_VERSION: *value = ROCSTATEVEC_VERSION_MAJOR; return ROCSTATEVEC_STATUS_SUCCESS;
    case ROCSTATEVEC_PROPERTY_MINOR_VERSION: *value = ROCSTATEVEC_VERSION_MINOR; return ROCSTATEVEC_STATUS_SUCCESS;
    case ROCSTATEVEC_PROPERTY_PATCH_LEVEL:   *value = ROCSTATEVEC_VERSION_PATCH; return ROCSTATEVEC_STATUS_SUCCESS;
    }
    return ROCSTATEVEC_STATUS_INVALID_VALUE;
}

extern "C" rocstatevec_status rocstatevec_set_stream(rocstatevec_handle h, hipStream_t stream)
{
    ROCSTATEVEC_CHECK_HANDLE(h);
    reinterpret_cast<handle*>(h)->stream = stream;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_get_stream(rocstatevec_handle h, hipStream_t* stream)
{
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(stream);
    *stream = reinterpret_cast<handle*>(h)->stream;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_set_device_mem_handler(
    rocstatevec_handle h, const rocstatevec_device_mem_handler_t* mem_handler)
{
    ROCSTATEVEC_CHECK_HANDLE(h);
    auto* hh = reinterpret_cast<handle*>(h);
    if(mem_handler == nullptr)
    {
        hh->has_mem = false;
        std::memset(&hh->mem, 0, sizeof(hh->mem));
    }
    else
    {
        if(mem_handler->device_alloc == nullptr || mem_handler->device_free == nullptr)
            return ROCSTATEVEC_STATUS_INVALID_VALUE;
        hh->mem     = *mem_handler;
        hh->has_mem = true;
    }
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_get_device_mem_handler(
    rocstatevec_handle h, rocstatevec_device_mem_handler_t* mem_handler)
{
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(mem_handler);
    auto* hh = reinterpret_cast<handle*>(h);
    if(!hh->has_mem) return ROCSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR;
    *mem_handler = hh->mem;
    return ROCSTATEVEC_STATUS_SUCCESS;
}
