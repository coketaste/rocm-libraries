/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Phase 7 — permutation, absorb-diagonal, swap-index-bits.
 * ************************************************************************ */

#include "test_helpers.hpp"

#include <gtest/gtest.h>

using namespace rocstatevec::test;
using cd = std::complex<double>;

TEST(Permute, swap_index_bits_round_trips)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 3;
    constexpr size_t   N = size_t{1} << n;
    device_buffer<cd> dsv(N);

    std::vector<cd> init(N);
    for(size_t i = 0; i < N; ++i) init[i] = cd{double(i + 1), 0};
    copy_to_device(dsv.ptr, init.data(), N);

    rocstatevec_index_pair_t swap = {0, 2};
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_swap_index_bits(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                          &swap, 1, nullptr, nullptr, 0));
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_swap_index_bits(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                          &swap, 1, nullptr, nullptr, 0));
    std::vector<cd> hv(N);
    copy_to_host(hv.data(), dsv.ptr, N);
    for(size_t i = 0; i < N; ++i) EXPECT_NEAR(hv[i].real(), double(i + 1), tol_fp64);
}

TEST(Permute, absorb_diagonal_phase)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 1;
    device_buffer<cd> dsv(2);

    cd init[2] = { {0.6, 0}, {0.8, 0} };
    copy_to_device(dsv.ptr, init, 2);

    cd diag[2] = { {1, 0}, {0, 1} };
    int32_t q  = 0;
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_absorb_diagonal_matrix(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                                 diag, ROCSTATEVEC_C_64F, 0,
                                                 &q, 1, nullptr, nullptr, 0));
    cd hv[2];
    copy_to_host(hv, dsv.ptr, 2);
    EXPECT_NEAR(hv[0].real(), 0.6, tol_fp64);
    EXPECT_NEAR(hv[0].imag(), 0.0, tol_fp64);
    EXPECT_NEAR(hv[1].real(), 0.0, tol_fp64);
    EXPECT_NEAR(hv[1].imag(), 0.8, tol_fp64);
}

TEST(Permute, generalized_permutation_swap)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 1;
    device_buffer<cd> dsv(2);
    cd init[2] = { {1, 0}, {0, 0} };
    copy_to_device(dsv.ptr, init, 2);

    rocstatevec_index_t perm[2] = { 1, 0 };
    int32_t q = 0;
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_apply_generalized_permutation_matrix(
                  h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                  perm, nullptr, ROCSTATEVEC_C_64F,
                  0, &q, 1, nullptr, nullptr, 0, nullptr, 0));
    cd hv[2];
    copy_to_host(hv, dsv.ptr, 2);
    EXPECT_NEAR(hv[0].real(), 0.0, tol_fp64);
    EXPECT_NEAR(hv[1].real(), 1.0, tol_fp64);
}
