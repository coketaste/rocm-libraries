/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Shared GoogleTest helpers for the rocTENSORNET test suite.
 * ************************************************************************ */

#pragma once

#include "roctensornet.h"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <vector>

#define ROC_TN_EXPECT_OK(expr)            EXPECT_EQ((expr), ROCTENSORNET_STATUS_SUCCESS)
#define ROC_TN_ASSERT_OK(expr)            ASSERT_EQ((expr), ROCTENSORNET_STATUS_SUCCESS)
#define ROC_TN_EXPECT_NOT_SUPPORTED(expr) EXPECT_EQ((expr), ROCTENSORNET_STATUS_NOT_SUPPORTED)

inline bool roctn_skip_if_no_gpu()
{
    int n = 0;
    hipError_t rc = hipGetDeviceCount(&n);
    if(rc != hipSuccess || n <= 0)
    {
        GTEST_SKIP() << "No HIP device available";
        return true;
    }
    return false;
}

/* Upload a host buffer to a fresh device allocation. The caller is
 * responsible for hipFree-ing the returned pointer. */
template <typename T>
inline T* roctn_upload(const std::vector<T>& host)
{
    T* d = nullptr;
    EXPECT_EQ(hipMalloc(&d, host.size() * sizeof(T)), hipSuccess);
    EXPECT_EQ(hipMemcpy(d, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice), hipSuccess);
    return d;
}

template <typename T>
inline std::vector<T> roctn_download(const T* device, size_t n)
{
    std::vector<T> host(n);
    EXPECT_EQ(hipMemcpy(host.data(), device, n * sizeof(T), hipMemcpyDeviceToHost), hipSuccess);
    return host;
}

/* Tracking allocator: records each alloc / free call for verification
 * that the user-provided workspace is consumed and that the device
 * memory handler is honored. */
struct tracking_allocator
{
    std::atomic<int> alloc_calls{0};
    std::atomic<int> free_calls{0};

    static int do_alloc(void* ctx, void** ptr, size_t bytes, hipStream_t stream)
    {
        auto* self = static_cast<tracking_allocator*>(ctx);
        self->alloc_calls.fetch_add(1);
        return hipMallocAsync(ptr, bytes, stream) == hipSuccess ? 0 : 1;
    }
    static int do_free(void* ctx, void* ptr, size_t bytes, hipStream_t stream)
    {
        auto* self = static_cast<tracking_allocator*>(ctx);
        (void)bytes;
        self->free_calls.fetch_add(1);
        return hipFreeAsync(ptr, stream) == hipSuccess ? 0 : 1;
    }
    roctensornet_device_mem_handler_t handler()
    {
        roctensornet_device_mem_handler_t h{};
        h.ctx          = this;
        h.device_alloc = &tracking_allocator::do_alloc;
        h.device_free  = &tracking_allocator::do_free;
        std::strncpy(h.name, "tracking", sizeof(h.name) - 1);
        return h;
    }
};
