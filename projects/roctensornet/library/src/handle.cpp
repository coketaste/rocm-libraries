/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "roctensornet_internal.hpp"

using namespace roctensornet;

extern "C" {

roctensornet_status roctensornet_create(roctensornet_handle* out)
{
    ROCTENSORNET_CHECK_PTR(out);
    auto* h = new (std::nothrow) handle();
    if(h == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    int dev = 0;
    if(hipGetDevice(&dev) == hipSuccess) h->device_id = dev;
    *out = reinterpret_cast<roctensornet_handle>(h);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status roctensornet_destroy(roctensornet_handle h)
{
    if(h == nullptr) return ROCTENSORNET_STATUS_SUCCESS;
    delete reinterpret_cast<handle*>(h);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status roctensornet_set_stream(roctensornet_handle h, hipStream_t s)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    reinterpret_cast<handle*>(h)->stream = s;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status roctensornet_get_stream(roctensornet_handle h, hipStream_t* s)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    ROCTENSORNET_CHECK_PTR(s);
    *s = reinterpret_cast<handle*>(h)->stream;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_set_device_mem_handler(roctensornet_handle h,
                                    const roctensornet_device_mem_handler_t* m)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* hh = reinterpret_cast<handle*>(h);
    if(m == nullptr)
    {
        hh->has_mem = false;
        hh->mem    = {};
        return ROCTENSORNET_STATUS_SUCCESS;
    }
    if(m->device_alloc == nullptr || m->device_free == nullptr)
        return ROCTENSORNET_STATUS_INVALID_VALUE;
    hh->mem     = *m;
    hh->has_mem = true;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_get_device_mem_handler(roctensornet_handle h,
                                    roctensornet_device_mem_handler_t* out)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    ROCTENSORNET_CHECK_PTR(out);
    auto* hh = reinterpret_cast<handle*>(h);
    if(!hh->has_mem) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    *out = hh->mem;
    return ROCTENSORNET_STATUS_SUCCESS;
}

} // extern "C"
