/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Phase 8 — accessor.
 * ************************************************************************ */

#include "test_helpers.hpp"

#include <gtest/gtest.h>

using namespace rocstatevec::test;
using cd = std::complex<double>;

TEST(Accessor, get_after_set_round_trip)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 3;
    constexpr size_t   N = size_t{1} << n;
    device_buffer<cd> dsv(N);
    rocstatevec_initialize_state_vector(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                        ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO);

    int32_t bo[3] = {0, 1, 2};
    rocstatevec_accessor_descriptor a = nullptr;
    size_t ws = 0;
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_accessor_create(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                          &a, bo, 3, nullptr, nullptr, 0, &ws));

    std::vector<cd> in_buf(N);
    for(size_t i = 0; i < N; ++i) in_buf[i] = cd{double(i + 1), 0};
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_accessor_set(h.h, a, in_buf.data(), 0, N));

    std::vector<cd> out_buf(N);
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_accessor_get(h.h, a, out_buf.data(), 0, N));
    for(size_t i = 0; i < N; ++i) EXPECT_NEAR(out_buf[i].real(), in_buf[i].real(), tol_fp64);

    rocstatevec_accessor_destroy(a);
}

TEST(Accessor, view_get_after_init)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 2;
    constexpr size_t   N = size_t{1} << n;
    device_buffer<cd> dsv(N);
    rocstatevec_initialize_state_vector(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                        ROCSTATEVEC_STATE_VECTOR_TYPE_GHZ);

    int32_t bo[2] = {0, 1};
    rocstatevec_accessor_descriptor a = nullptr;
    size_t ws = 0;
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_accessor_create_view(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                               &a, bo, 2, nullptr, nullptr, 0, &ws));
    std::vector<cd> out_buf(N);
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_accessor_get(h.h, a, out_buf.data(), 0, N));
    EXPECT_NEAR(out_buf[0].real(),     1.0 / std::sqrt(2.0), tol_fp64);
    EXPECT_NEAR(out_buf[3].real(),     1.0 / std::sqrt(2.0), tol_fp64);

    rocstatevec_accessor_destroy(a);
}
