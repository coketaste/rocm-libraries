/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Phase 4 — ApplyMatrix and ApplyPauliRotation.
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
}

TEST(Apply, hadamard_then_cnot_makes_bell_pair)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 2;
    constexpr size_t   N = size_t{1} << n;
    device_buffer<cd> dsv(N);
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_initialize_state_vector(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                                  ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO));

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    cd H_mat[4] = { {inv_sqrt2, 0}, {inv_sqrt2, 0}, {inv_sqrt2, 0}, {-inv_sqrt2, 0} };
    cd X_mat[4] = { zero, one, one, zero };
    int32_t q0 = 0, q1 = 1;

    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_apply_matrix(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                       H_mat, ROCSTATEVEC_C_64F,
                                       ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                       &q0, 1, nullptr, nullptr, 0,
                                       ROCSTATEVEC_COMPUTE_64F, nullptr, 0));
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_apply_matrix(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                       X_mat, ROCSTATEVEC_C_64F,
                                       ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                       &q1, 1, &q0, nullptr, 1,
                                       ROCSTATEVEC_COMPUTE_64F, nullptr, 0));

    std::vector<cd> hv(N);
    copy_to_host(hv.data(), dsv.ptr, N);
    EXPECT_NEAR(hv[0].real(), inv_sqrt2, tol_fp64);
    EXPECT_NEAR(hv[3].real(), inv_sqrt2, tol_fp64);
    EXPECT_NEAR(std::abs(hv[1]), 0.0, tol_fp64);
    EXPECT_NEAR(std::abs(hv[2]), 0.0, tol_fp64);
}

TEST(Apply, x_gate_flips_bit_zero)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 1;
    device_buffer<cd> dsv(2);
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_initialize_state_vector(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                                  ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO));
    cd X_mat[4] = { zero, one, one, zero };
    int32_t q0 = 0;
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_apply_matrix(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                       X_mat, ROCSTATEVEC_C_64F,
                                       ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                       &q0, 1, nullptr, nullptr, 0,
                                       ROCSTATEVEC_COMPUTE_64F, nullptr, 0));
    std::vector<cd> hv(2);
    copy_to_host(hv.data(), dsv.ptr, 2);
    EXPECT_NEAR(hv[0].real(), 0.0, tol_fp64);
    EXPECT_NEAR(hv[1].real(), 1.0, tol_fp64);
}

TEST(Apply, pauli_rotation_x_pi_is_x)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 1;
    device_buffer<cd> dsv(2);
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_initialize_state_vector(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                                  ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO));

    rocstatevec_pauli paulis[1] = { ROCSTATEVEC_PAULI_X };
    int32_t           targets[1] = { 0 };
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_apply_pauli_rotation(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                               M_PI, paulis, targets, 1,
                                               nullptr, nullptr, 0));
    std::vector<cd> hv(2);
    copy_to_host(hv.data(), dsv.ptr, 2);
    EXPECT_NEAR(std::abs(hv[0]), 0.0, tol_fp64);
    EXPECT_NEAR(std::abs(hv[1]), 1.0, tol_fp64);
}

TEST(Apply, pauli_z_rotation_diagonal)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 2;
    device_buffer<cd> dsv(4);
    cd init[4] = {
        { 1.0 / std::sqrt(4.0), 0 }, { 1.0 / std::sqrt(4.0), 0 },
        { 1.0 / std::sqrt(4.0), 0 }, { 1.0 / std::sqrt(4.0), 0 }
    };
    copy_to_device(dsv.ptr, init, 4);

    rocstatevec_pauli paulis[1] = { ROCSTATEVEC_PAULI_Z };
    int32_t           targets[1] = { 0 };
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_apply_pauli_rotation(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                               M_PI, paulis, targets, 1,
                                               nullptr, nullptr, 0));
    std::vector<cd> hv(4);
    copy_to_host(hv.data(), dsv.ptr, 4);
    // exp(-i * pi/2 * Z) on |q0=0> states gives factor exp(-i pi/2) = -i;
    // on |q0=1> states gives exp(+i pi/2) = +i. Original amplitudes were real.
    EXPECT_NEAR(hv[0].imag(), -0.5, tol_fp64);
    EXPECT_NEAR(hv[2].imag(), -0.5, tol_fp64);
    EXPECT_NEAR(hv[1].imag(),  0.5, tol_fp64);
    EXPECT_NEAR(hv[3].imag(),  0.5, tol_fp64);
}
