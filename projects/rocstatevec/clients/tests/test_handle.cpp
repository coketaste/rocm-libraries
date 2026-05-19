/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Phase 2 — handle lifecycle, stream binding, mem-handler plumbing.
 * ************************************************************************ */

#include <rocstatevec.h>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

TEST(Handle, create_destroy_round_trip)
{
    rocstatevec_handle h = nullptr;
    EXPECT_EQ(ROCSTATEVEC_STATUS_SUCCESS, rocstatevec_create_handle(&h));
    EXPECT_NE(nullptr, h);

    int v = 0;
    EXPECT_EQ(ROCSTATEVEC_STATUS_SUCCESS, rocstatevec_get_version(h, &v));
    EXPECT_GT(v, 0);

    EXPECT_EQ(ROCSTATEVEC_STATUS_SUCCESS, rocstatevec_destroy_handle(h));
    EXPECT_EQ(ROCSTATEVEC_STATUS_SUCCESS, rocstatevec_destroy_handle(nullptr));
}

TEST(Handle, null_handle_returns_not_initialized)
{
    int v = 0;
    EXPECT_EQ(ROCSTATEVEC_STATUS_NOT_INITIALIZED, rocstatevec_get_version(nullptr, &v));
    EXPECT_EQ(ROCSTATEVEC_STATUS_NOT_INITIALIZED, rocstatevec_set_stream(nullptr, nullptr));
}

TEST(Handle, stream_round_trip)
{
    rocstatevec_handle h = nullptr;
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS, rocstatevec_create_handle(&h));

    hipStream_t s = nullptr;
    ASSERT_EQ(hipSuccess, hipStreamCreate(&s));
    EXPECT_EQ(ROCSTATEVEC_STATUS_SUCCESS, rocstatevec_set_stream(h, s));

    hipStream_t got = nullptr;
    EXPECT_EQ(ROCSTATEVEC_STATUS_SUCCESS, rocstatevec_get_stream(h, &got));
    EXPECT_EQ(s, got);

    EXPECT_EQ(hipSuccess, hipStreamDestroy(s));
    EXPECT_EQ(ROCSTATEVEC_STATUS_SUCCESS, rocstatevec_destroy_handle(h));
}

TEST(Handle, mem_handler_round_trip)
{
    rocstatevec_handle h = nullptr;
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS, rocstatevec_create_handle(&h));

    rocstatevec_device_mem_handler_t out{};
    EXPECT_EQ(ROCSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR,
              rocstatevec_get_device_mem_handler(h, &out));

    rocstatevec_device_mem_handler_t in{};
    in.ctx          = nullptr;
    in.device_alloc = +[](void*, void**, size_t, hipStream_t) { return 0; };
    in.device_free  = +[](void*, void*, size_t, hipStream_t) { return 0; };
    EXPECT_EQ(ROCSTATEVEC_STATUS_SUCCESS, rocstatevec_set_device_mem_handler(h, &in));
    EXPECT_EQ(ROCSTATEVEC_STATUS_SUCCESS, rocstatevec_get_device_mem_handler(h, &out));
    EXPECT_EQ(in.device_alloc, out.device_alloc);

    EXPECT_EQ(ROCSTATEVEC_STATUS_SUCCESS, rocstatevec_destroy_handle(h));
}
