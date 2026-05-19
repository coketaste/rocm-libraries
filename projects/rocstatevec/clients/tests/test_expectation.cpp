/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Phase 6 — Expectation suite.
 * ************************************************************************ */

#include "test_helpers.hpp"

#include <gtest/gtest.h>

#include <complex>

using namespace rocstatevec::test;
using cd = std::complex<double>;

namespace
{
constexpr cd zero{0, 0};
constexpr cd one{1, 0};

device_buffer<cd> make_bell_pair(handle_guard& h)
{
    constexpr uint32_t n = 2;
    constexpr size_t   N = size_t{1} << n;
    device_buffer<cd> dsv(N);
    rocstatevec_initialize_state_vector(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                        ROCSTATEVEC_STATE_VECTOR_TYPE_GHZ);
    return dsv;
}
} // namespace

TEST(Expectation, bell_pair_zz_is_one)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    auto dsv = make_bell_pair(h);

    rocstatevec_pauli zz[2]   = { ROCSTATEVEC_PAULI_Z, ROCSTATEVEC_PAULI_Z };
    int32_t           qb[2]   = { 0, 1 };
    const rocstatevec_pauli* p_arr[1] = { zz };
    const int32_t*           b_arr[1] = { qb };
    uint32_t                 n_arr[1] = { 2 };
    double                   ev[1]    = { 0.0 };

    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_compute_expectations_on_pauli_basis(h.h, dsv.ptr, ROCSTATEVEC_C_64F, 2,
                                                              ev, p_arr, 1, b_arr, n_arr));
    EXPECT_NEAR(ev[0], 1.0, tol_fp64);
}

TEST(Expectation, bell_pair_yy_is_minus_one)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    auto dsv = make_bell_pair(h);

    rocstatevec_pauli yy[2]   = { ROCSTATEVEC_PAULI_Y, ROCSTATEVEC_PAULI_Y };
    int32_t           qb[2]   = { 0, 1 };
    const rocstatevec_pauli* p_arr[1] = { yy };
    const int32_t*           b_arr[1] = { qb };
    uint32_t                 n_arr[1] = { 2 };
    double                   ev[1]    = { 99.0 };

    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_compute_expectations_on_pauli_basis(h.h, dsv.ptr, ROCSTATEVEC_C_64F, 2,
                                                              ev, p_arr, 1, b_arr, n_arr));
    EXPECT_NEAR(ev[0], -1.0, tol_fp64);
}

TEST(Expectation, bell_pair_xx_is_one)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    auto dsv = make_bell_pair(h);

    rocstatevec_pauli xx[2]   = { ROCSTATEVEC_PAULI_X, ROCSTATEVEC_PAULI_X };
    int32_t           qb[2]   = { 0, 1 };
    const rocstatevec_pauli* p_arr[1] = { xx };
    const int32_t*           b_arr[1] = { qb };
    uint32_t                 n_arr[1] = { 2 };
    double                   ev[1]    = { 0.0 };

    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_compute_expectations_on_pauli_basis(h.h, dsv.ptr, ROCSTATEVEC_C_64F, 2,
                                                              ev, p_arr, 1, b_arr, n_arr));
    EXPECT_NEAR(ev[0], 1.0, tol_fp64);
}

TEST(Expectation, bell_pair_zero_state_z_is_one)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 1;
    device_buffer<cd> dsv(2);
    rocstatevec_initialize_state_vector(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                        ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO);

    rocstatevec_pauli z[1] = { ROCSTATEVEC_PAULI_Z };
    int32_t           q[1] = { 0 };
    const rocstatevec_pauli* p_arr[1] = { z };
    const int32_t*           b_arr[1] = { q };
    uint32_t                 n_arr[1] = { 1 };
    double                   ev[1]    = { 0.0 };

    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_compute_expectations_on_pauli_basis(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                                              ev, p_arr, 1, b_arr, n_arr));
    EXPECT_NEAR(ev[0], 1.0, tol_fp64);
}

TEST(Expectation, dense_z_matrix_zero_state)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 1;
    device_buffer<cd> dsv(2);
    rocstatevec_initialize_state_vector(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                        ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO);

    cd      Z_mat[4] = { one, zero, zero, {-1, 0} };
    int32_t q[1]     = { 0 };
    cd      ev{ 0, 0 };
    double  rn = 0;

    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_compute_expectation(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                              &ev, ROCSTATEVEC_C_64F, &rn,
                                              Z_mat, ROCSTATEVEC_C_64F,
                                              ROCSTATEVEC_MATRIX_LAYOUT_ROW,
                                              q, 1, ROCSTATEVEC_COMPUTE_64F, nullptr, 0));
    EXPECT_NEAR(ev.real(), 1.0, tol_fp64);
    EXPECT_NEAR(ev.imag(), 0.0, tol_fp64);
}
