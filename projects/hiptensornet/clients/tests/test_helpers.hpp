/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Shared GoogleTest helpers for the hipTENSORNET test suite.
 *
 * The hipTENSORNET tests exercise the *wrapper* layer: every call is
 * a thin forwarding shim, so the suite is intentionally smaller than
 * the rocTENSORNET test suite. The smoke tests verify that the AMD
 * forwarding path is wired up correctly and that opaque-handle casts
 * preserve identity through the wrapper boundary.
 * ************************************************************************ */

#pragma once

#include "hiptensornet.h"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <vector>

#define HTN_EXPECT_OK(expr) EXPECT_EQ((expr), HIPTENSORNET_STATUS_SUCCESS)
#define HTN_ASSERT_OK(expr) ASSERT_EQ((expr), HIPTENSORNET_STATUS_SUCCESS)

inline bool htn_skip_if_no_gpu()
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

template <typename T>
inline T* htn_upload(const std::vector<T>& host)
{
    T* d = nullptr;
    EXPECT_EQ(hipMalloc(&d, host.size() * sizeof(T)), hipSuccess);
    EXPECT_EQ(hipMemcpy(d, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice), hipSuccess);
    return d;
}

template <typename T>
inline std::vector<T> htn_download(const T* device, size_t n)
{
    std::vector<T> host(n);
    EXPECT_EQ(hipMemcpy(host.data(), device, n * sizeof(T), hipMemcpyDeviceToHost), hipSuccess);
    return host;
}
