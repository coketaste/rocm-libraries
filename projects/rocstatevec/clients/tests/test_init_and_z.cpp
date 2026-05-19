/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Phase 3 - InitializeStateVector + Abs2SumOnZBasis + MeasureOnZBasis.
 * Eigen-free reference: amplitudes for the four named presets are
 * determined analytically.
 * ************************************************************************ */

#include "test_helpers.hpp"

#include <gtest/gtest.h>

using namespace rocstatevec::test;

TEST(InitAndZ, zero_state_amplitudes)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 3;
    constexpr size_t   N = size_t{1} << n;
    device_buffer<std::complex<double>> dsv(N);

    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_initialize_state_vector(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                                  ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO));

    std::vector<std::complex<double>> hv(N);
    copy_to_host(hv.data(), dsv.ptr, N);
    EXPECT_NEAR(hv[0].real(), 1.0, tol_fp64);
    for(size_t i = 1; i < N; ++i) EXPECT_NEAR(std::abs(hv[i]), 0.0, tol_fp64);
}

TEST(InitAndZ, uniform_state_norm)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 4;
    constexpr size_t   N = size_t{1} << n;
    device_buffer<std::complex<double>> dsv(N);

    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_initialize_state_vector(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                                  ROCSTATEVEC_STATE_VECTOR_TYPE_UNIFORM));
    std::vector<std::complex<double>> hv(N);
    copy_to_host(hv.data(), dsv.ptr, N);
    EXPECT_NEAR(l2_norm_squared(hv), 1.0, tol_fp64);
    for(auto& z : hv) EXPECT_NEAR(z.real(), 1.0 / std::sqrt(double(N)), tol_fp64);
}

TEST(InitAndZ, ghz_state_amplitudes)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 5;
    constexpr size_t   N = size_t{1} << n;
    device_buffer<std::complex<double>> dsv(N);
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_initialize_state_vector(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                                  ROCSTATEVEC_STATE_VECTOR_TYPE_GHZ));
    std::vector<std::complex<double>> hv(N);
    copy_to_host(hv.data(), dsv.ptr, N);
    const double inv = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(hv[0].real(),     inv, tol_fp64);
    EXPECT_NEAR(hv[N - 1].real(), inv, tol_fp64);
    for(size_t i = 1; i + 1 < N; ++i) EXPECT_NEAR(std::abs(hv[i]), 0.0, tol_fp64);
}

TEST(InitAndZ, abs2sum_z_basis_zero_state)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 3;
    device_buffer<std::complex<double>> dsv(size_t{1} << n);
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_initialize_state_vector(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                                  ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO));

    int32_t qubits[3] = {0, 1, 2};
    double  s0 = -1, s1 = -1;
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_abs2_sum_on_z_basis(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                              &s0, &s1, qubits, 3));
    EXPECT_NEAR(s0, 1.0, tol_fp64);
    EXPECT_NEAR(s1, 0.0, tol_fp64);
}

TEST(InitAndZ, measure_ghz_collapses_to_one_bit_string)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 4;
    constexpr size_t   N = size_t{1} << n;
    device_buffer<std::complex<double>> dsv(N);
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_initialize_state_vector(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                                  ROCSTATEVEC_STATE_VECTOR_TYPE_GHZ));

    int32_t qubits[4] = {0, 1, 2, 3};
    int32_t parity    = -1;
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_measure_on_z_basis(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                             &parity, qubits, 4, 0.25,
                                             ROCSTATEVEC_COLLAPSE_NORMALIZE_AS_SPECIFIED));
    EXPECT_TRUE(parity == 0 || parity == 1);

    std::vector<std::complex<double>> hv(N);
    copy_to_host(hv.data(), dsv.ptr, N);
    EXPECT_NEAR(l2_norm_squared(hv), 1.0, tol_fp64);
    if(parity == 0)
    {
        EXPECT_NEAR(hv[0].real(),     1.0, tol_fp64);
        EXPECT_NEAR(hv[N - 1].real(), 0.0, tol_fp64);
    }
    else
    {
        EXPECT_NEAR(hv[0].real(),     0.0, tol_fp64);
        EXPECT_NEAR(hv[N - 1].real(), 1.0, tol_fp64);
    }
}
