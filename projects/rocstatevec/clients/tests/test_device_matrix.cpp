/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Regression tests for matrix pointers that already live on the device.
 *
 * The fix exercises `is_device_pointer()` inside every entry point that
 * accepts a matrix-shaped buffer; before the fix, these functions
 * unconditionally `hipMemcpyHostToDevice`'d the pointer, which silently
 * read garbage from a device address.
 * ************************************************************************ */

#include "test_helpers.hpp"

#include <gtest/gtest.h>
#include <rocstatevec.h>

#include <cmath>
#include <complex>
#include <vector>

using namespace rocstatevec::test;
using cd = std::complex<double>;

namespace
{
constexpr uint32_t n_qubits = 2;
constexpr size_t   N        = size_t{1} << n_qubits;

void make_hadamard(cd* out)
{
    const double s = 1.0 / std::sqrt(2.0);
    out[0] = { s, 0}; out[1] = { s, 0};
    out[2] = { s, 0}; out[3] = {-s, 0};
}
} // namespace

TEST(DeviceMatrix, ApplyMatrixWithDeviceResidentMatrix)
{
    if(skip_if_no_gpu()) GTEST_SKIP();
    handle_guard hg;
    ASSERT_NE(hg.h, nullptr);

    // Reference run: matrix on host.
    std::vector<cd> host_ref(N);
    {
        device_buffer<cd> dsv(N);
        ASSERT_EQ(rocstatevec_initialize_state_vector(hg.h, dsv.ptr,
                                                      ROCSTATEVEC_C_64F, n_qubits,
                                                      ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO),
                  ROCSTATEVEC_STATUS_SUCCESS);
        cd      H_mat[4]; make_hadamard(H_mat);
        int32_t q0 = 0;
        ASSERT_EQ(rocstatevec_apply_matrix(hg.h, dsv.ptr, ROCSTATEVEC_C_64F, n_qubits,
                                           H_mat, ROCSTATEVEC_C_64F,
                                           ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                           &q0, 1, nullptr, nullptr, 0,
                                           ROCSTATEVEC_COMPUTE_64F, nullptr, 0),
                  ROCSTATEVEC_STATUS_SUCCESS);
        copy_to_host(host_ref.data(), dsv.ptr, N);
    }

    // Same call but the matrix lives on the device.
    std::vector<cd> host_test(N);
    {
        device_buffer<cd> dsv(N);
        device_buffer<cd> d_mat(4);
        ASSERT_EQ(rocstatevec_initialize_state_vector(hg.h, dsv.ptr,
                                                      ROCSTATEVEC_C_64F, n_qubits,
                                                      ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO),
                  ROCSTATEVEC_STATUS_SUCCESS);
        cd H_mat[4]; make_hadamard(H_mat);
        copy_to_device(d_mat.ptr, H_mat, 4);

        int32_t q0 = 0;
        ASSERT_EQ(rocstatevec_apply_matrix(hg.h, dsv.ptr, ROCSTATEVEC_C_64F, n_qubits,
                                           d_mat.ptr, ROCSTATEVEC_C_64F,
                                           ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                           &q0, 1, nullptr, nullptr, 0,
                                           ROCSTATEVEC_COMPUTE_64F, nullptr, 0),
                  ROCSTATEVEC_STATUS_SUCCESS);
        copy_to_host(host_test.data(), dsv.ptr, N);
    }

    for(size_t i = 0; i < N; ++i)
    {
        EXPECT_NEAR(host_ref[i].real(), host_test[i].real(), tol_fp64);
        EXPECT_NEAR(host_ref[i].imag(), host_test[i].imag(), tol_fp64);
    }
}

TEST(DeviceMatrix, AbsorbDiagonalWithDeviceResidentDiag)
{
    if(skip_if_no_gpu()) GTEST_SKIP();
    handle_guard hg;
    ASSERT_NE(hg.h, nullptr);

    cd diag[2] = {{2.0, 0}, {3.0, 0}};

    // Initial state |11>: amplitude 1 only at index 3.
    std::vector<cd> host_init(N, cd{0, 0});
    host_init[3] = cd{1.0, 0};

    auto run = [&](const cd* matrix_ptr, std::vector<cd>& out, bool dev_matrix) {
        device_buffer<cd> dsv(N);
        copy_to_device(dsv.ptr, host_init.data(), N);
        device_buffer<cd> d_mat(2);
        const cd*         use_ptr = matrix_ptr;
        if(dev_matrix)
        {
            copy_to_device(d_mat.ptr, matrix_ptr, 2);
            use_ptr = d_mat.ptr;
        }
        int32_t q0 = 0;
        ASSERT_EQ(rocstatevec_absorb_diagonal_matrix(hg.h, dsv.ptr, ROCSTATEVEC_C_64F, n_qubits,
                                                     use_ptr, ROCSTATEVEC_C_64F, 0,
                                                     &q0, 1, nullptr, nullptr, 0),
                  ROCSTATEVEC_STATUS_SUCCESS);
        out.resize(N);
        copy_to_host(out.data(), dsv.ptr, N);
    };

    std::vector<cd> ref, dev;
    run(diag, ref, /*dev_matrix=*/false);
    run(diag, dev, /*dev_matrix=*/true);
    for(size_t i = 0; i < N; ++i)
    {
        EXPECT_NEAR(ref[i].real(), dev[i].real(), tol_fp64);
        EXPECT_NEAR(ref[i].imag(), dev[i].imag(), tol_fp64);
    }
}

TEST(DeviceMatrix, ComputeExpectationWithDeviceResidentMatrix)
{
    if(skip_if_no_gpu()) GTEST_SKIP();
    handle_guard hg;
    ASSERT_NE(hg.h, nullptr);

    // |+> on qubit 0, |0> on qubit 1: <Z_0> = 0, <X_0> = 1.
    device_buffer<cd> dsv(N);
    ASSERT_EQ(rocstatevec_initialize_state_vector(hg.h, dsv.ptr, ROCSTATEVEC_C_64F,
                                                  n_qubits, ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO),
              ROCSTATEVEC_STATUS_SUCCESS);
    cd       H_mat[4]; make_hadamard(H_mat);
    int32_t  q0 = 0;
    ASSERT_EQ(rocstatevec_apply_matrix(hg.h, dsv.ptr, ROCSTATEVEC_C_64F, n_qubits,
                                       H_mat, ROCSTATEVEC_C_64F,
                                       ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0, &q0, 1,
                                       nullptr, nullptr, 0,
                                       ROCSTATEVEC_COMPUTE_64F, nullptr, 0),
              ROCSTATEVEC_STATUS_SUCCESS);

    cd                X_mat[4] = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    device_buffer<cd> d_X(4);
    copy_to_device(d_X.ptr, X_mat, 4);

    double e_dev = 0.0, residual_dev = 0.0;
    ASSERT_EQ(rocstatevec_compute_expectation(hg.h, dsv.ptr, ROCSTATEVEC_C_64F, n_qubits,
                                              &e_dev, ROCSTATEVEC_R_64F, &residual_dev,
                                              d_X.ptr, ROCSTATEVEC_C_64F,
                                              ROCSTATEVEC_MATRIX_LAYOUT_ROW,
                                              &q0, 1,
                                              ROCSTATEVEC_COMPUTE_64F, nullptr, 0),
              ROCSTATEVEC_STATUS_SUCCESS);
    EXPECT_NEAR(e_dev, 1.0, tol_fp64);
    EXPECT_NEAR(residual_dev, 0.0, 1e-10);

    double e_host = 0.0;
    ASSERT_EQ(rocstatevec_compute_expectation(hg.h, dsv.ptr, ROCSTATEVEC_C_64F, n_qubits,
                                              &e_host, ROCSTATEVEC_R_64F, nullptr,
                                              X_mat, ROCSTATEVEC_C_64F,
                                              ROCSTATEVEC_MATRIX_LAYOUT_ROW,
                                              &q0, 1,
                                              ROCSTATEVEC_COMPUTE_64F, nullptr, 0),
              ROCSTATEVEC_STATUS_SUCCESS);
    EXPECT_NEAR(e_dev, e_host, tol_fp64);
}

TEST(DeviceMatrix, TestMatrixTypeWithDeviceResidentMatrix)
{
    if(skip_if_no_gpu()) GTEST_SKIP();
    handle_guard hg;
    ASSERT_NE(hg.h, nullptr);

    cd                X_mat[4] = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    device_buffer<cd> d_X(4);
    copy_to_device(d_X.ptr, X_mat, 4);

    double r = 12345.0;
    ASSERT_EQ(rocstatevec_test_matrix_type(hg.h, &r, ROCSTATEVEC_MATRIX_TYPE_UNITARY,
                                           d_X.ptr, ROCSTATEVEC_C_64F,
                                           ROCSTATEVEC_MATRIX_LAYOUT_ROW, 1, 0,
                                           ROCSTATEVEC_COMPUTE_64F, nullptr, 0),
              ROCSTATEVEC_STATUS_SUCCESS);
    EXPECT_NEAR(r, 0.0, 1e-10);
}
