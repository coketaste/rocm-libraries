/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Asserts every `*_get_workspace_size` entry point now returns a
 * non-zero byte count for a non-trivial configuration, and that the
 * apply path actually runs to completion when handed a workspace of
 * exactly that size.
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
constexpr uint32_t n_qubits = 3;
constexpr size_t   N        = size_t{1} << n_qubits;
} // namespace

TEST(WorkspaceQuery, ApplyMatrixReturnsNonzero)
{
    if(skip_if_no_gpu()) GTEST_SKIP();
    handle_guard hg; ASSERT_NE(hg.h, nullptr);

    size_t bytes = 0;
    ASSERT_EQ(rocstatevec_apply_matrix_get_workspace_size(hg.h, ROCSTATEVEC_C_64F, n_qubits,
                                                          nullptr, ROCSTATEVEC_C_64F,
                                                          ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                                          /*n_targets=*/1, /*n_controls=*/0,
                                                          ROCSTATEVEC_COMPUTE_64F, &bytes),
              ROCSTATEVEC_STATUS_SUCCESS);
    EXPECT_GT(bytes, 0u);
    // Must be at least N elements of complex64 (the scratch portion).
    EXPECT_GE(bytes, N * sizeof(cd));
}

TEST(WorkspaceQuery, ApplyMatrixUsesUserBuffer)
{
    if(skip_if_no_gpu()) GTEST_SKIP();
    handle_guard hg; ASSERT_NE(hg.h, nullptr);

    size_t bytes = 0;
    ASSERT_EQ(rocstatevec_apply_matrix_get_workspace_size(hg.h, ROCSTATEVEC_C_64F, n_qubits,
                                                          nullptr, ROCSTATEVEC_C_64F,
                                                          ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                                          1, 0,
                                                          ROCSTATEVEC_COMPUTE_64F, &bytes),
              ROCSTATEVEC_STATUS_SUCCESS);
    ASSERT_GT(bytes, 0u);

    void* d_ws = nullptr;
    ASSERT_EQ(hipMalloc(&d_ws, bytes), hipSuccess);

    device_buffer<cd> dsv(N);
    ASSERT_EQ(rocstatevec_initialize_state_vector(hg.h, dsv.ptr, ROCSTATEVEC_C_64F, n_qubits,
                                                  ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO),
              ROCSTATEVEC_STATUS_SUCCESS);

    const double s = 1.0 / std::sqrt(2.0);
    cd      H_mat[4] = {{s, 0}, {s, 0}, {s, 0}, {-s, 0}};
    int32_t q0       = 0;
    ASSERT_EQ(rocstatevec_apply_matrix(hg.h, dsv.ptr, ROCSTATEVEC_C_64F, n_qubits,
                                       H_mat, ROCSTATEVEC_C_64F,
                                       ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                       &q0, 1, nullptr, nullptr, 0,
                                       ROCSTATEVEC_COMPUTE_64F, d_ws, bytes),
              ROCSTATEVEC_STATUS_SUCCESS);

    std::vector<cd> hsv(N);
    copy_to_host(hsv.data(), dsv.ptr, N);
    EXPECT_NEAR(hsv[0].real(), s, tol_fp64);
    EXPECT_NEAR(hsv[1].real(), s, tol_fp64);

    hipFree(d_ws);
}

TEST(WorkspaceQuery, GeneralizedPermutationReturnsNonzero)
{
    if(skip_if_no_gpu()) GTEST_SKIP();
    handle_guard hg; ASSERT_NE(hg.h, nullptr);

    cd     diag[2] = {{1, 0}, {-1, 0}};
    size_t bytes = 0;
    ASSERT_EQ(rocstatevec_apply_generalized_permutation_matrix_get_workspace_size(
                  hg.h, ROCSTATEVEC_C_64F, n_qubits,
                  /*permutation=*/nullptr, diag, ROCSTATEVEC_C_64F,
                  /*targets=*/nullptr, /*n_targets=*/1, /*n_controls=*/0, &bytes),
              ROCSTATEVEC_STATUS_SUCCESS);
    EXPECT_GT(bytes, 0u);
}

TEST(WorkspaceQuery, ComputeExpectationReturnsNonzero)
{
    if(skip_if_no_gpu()) GTEST_SKIP();
    handle_guard hg; ASSERT_NE(hg.h, nullptr);

    size_t bytes = 0;
    ASSERT_EQ(rocstatevec_compute_expectation_get_workspace_size(
                  hg.h, ROCSTATEVEC_C_64F, n_qubits, nullptr, ROCSTATEVEC_C_64F,
                  ROCSTATEVEC_MATRIX_LAYOUT_ROW, /*n_basis_bits=*/1,
                  ROCSTATEVEC_COMPUTE_64F, &bytes),
              ROCSTATEVEC_STATUS_SUCCESS);
    EXPECT_GT(bytes, 0u);
}

TEST(WorkspaceQuery, ApplyMatrixBatchedReturnsNonzero)
{
    if(skip_if_no_gpu()) GTEST_SKIP();
    handle_guard hg; ASSERT_NE(hg.h, nullptr);

    size_t bytes = 0;
    ASSERT_EQ(rocstatevec_apply_matrix_batched_get_workspace_size(
                  hg.h, ROCSTATEVEC_C_64F, n_qubits,
                  /*n_state_vectors=*/2, /*state_vector_size=*/N,
                  ROCSTATEVEC_MATRIX_MAP_TYPE_BROADCAST,
                  nullptr, nullptr, ROCSTATEVEC_C_64F,
                  ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                  /*n_matrices=*/1, /*n_targets=*/1, /*n_controls=*/0,
                  ROCSTATEVEC_COMPUTE_64F, &bytes),
              ROCSTATEVEC_STATUS_SUCCESS);
    EXPECT_GT(bytes, 0u);
}
