/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Phase 9 — test_matrix_type, apply_matrix_batched.
 * ************************************************************************ */

#include "test_helpers.hpp"

#include <gtest/gtest.h>

using namespace rocstatevec::test;
using cd = std::complex<double>;

TEST(TestMatrixType, identity_unitary_and_hermitian)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    cd I[4] = {{1, 0}, {0, 0}, {0, 0}, {1, 0}};
    double r = 1.0;
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_test_matrix_type(h.h, &r, ROCSTATEVEC_MATRIX_TYPE_UNITARY,
                                           I, ROCSTATEVEC_C_64F, ROCSTATEVEC_MATRIX_LAYOUT_ROW,
                                           1, 0, ROCSTATEVEC_COMPUTE_64F, nullptr, 0));
    EXPECT_NEAR(r, 0.0, tol_fp64);
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_test_matrix_type(h.h, &r, ROCSTATEVEC_MATRIX_TYPE_HERMITIAN,
                                           I, ROCSTATEVEC_C_64F, ROCSTATEVEC_MATRIX_LAYOUT_ROW,
                                           1, 0, ROCSTATEVEC_COMPUTE_64F, nullptr, 0));
    EXPECT_NEAR(r, 0.0, tol_fp64);
}

TEST(TestMatrixType, non_hermitian_detected)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    cd M[4] = {{1, 0}, {0, 1}, {0, 0}, {1, 0}};
    double r = 0.0;
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_test_matrix_type(h.h, &r, ROCSTATEVEC_MATRIX_TYPE_HERMITIAN,
                                           M, ROCSTATEVEC_C_64F, ROCSTATEVEC_MATRIX_LAYOUT_ROW,
                                           1, 0, ROCSTATEVEC_COMPUTE_64F, nullptr, 0));
    EXPECT_GT(r, 0.5);
}

TEST(ApplyMatrixBatched, broadcast_x_on_two_state_vectors)
{
    if(skip_if_no_gpu()) return;
    handle_guard       h;
    constexpr uint32_t n = 1;
    constexpr size_t   N = size_t{1} << n;
    constexpr uint32_t B = 2;

    device_buffer<cd> dsv(N * B);

    rocstatevec_initialize_state_vector(h.h, dsv.ptr,           ROCSTATEVEC_C_64F, n,
                                        ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO);
    rocstatevec_initialize_state_vector(h.h, dsv.ptr + N,       ROCSTATEVEC_C_64F, n,
                                        ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO);

    cd X[4] = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    int32_t targets[1] = {0};
    int32_t mi[1]      = {0};
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_apply_matrix_batched(
                  h.h, dsv.ptr, ROCSTATEVEC_C_64F, n, B, rocstatevec_index_t(N),
                  ROCSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, mi, X,
                  ROCSTATEVEC_C_64F, ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1,
                  targets, 1, nullptr, nullptr, 0,
                  ROCSTATEVEC_COMPUTE_64F, nullptr, 0));

    std::vector<cd> host(N * B);
    copy_to_host(host.data(), dsv.ptr, N * B);
    EXPECT_NEAR(host[0].real(), 0.0, tol_fp64);
    EXPECT_NEAR(host[1].real(), 1.0, tol_fp64);
    EXPECT_NEAR(host[N + 0].real(), 0.0, tol_fp64);
    EXPECT_NEAR(host[N + 1].real(), 1.0, tol_fp64);
}
