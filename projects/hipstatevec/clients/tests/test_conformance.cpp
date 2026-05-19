/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Conformance tests for the AMD pass-through to rocstatevec.
 *
 * Numerical tolerances are chosen to match the FP64/FP32 budgets
 * documented in the rocSTATEVEC test fixtures.
 * ************************************************************************ */

#include <hipstatevec.h>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

namespace
{
constexpr double tol_fp64 = 1e-10;

inline bool has_gpu()
{
    int n = 0;
    return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}
inline void* dev_alloc(size_t b) { void* p = nullptr; hipMalloc(&p, b); return p; }
inline void  dev_free(void* p)   { if(p) hipFree(p); }
inline void  dev_h2d(void* d, const void* h, size_t b) { hipMemcpy(d, h, b, hipMemcpyHostToDevice); }
inline void  dev_d2h(void* h, const void* d, size_t b) { hipMemcpy(h, d, b, hipMemcpyDeviceToHost); }

struct hipsv_handle
{
    hipstatevecHandle_t h = nullptr;
    hipsv_handle()
    {
        if(hipstatevecCreate(&h) != HIPSTATEVEC_STATUS_SUCCESS) h = nullptr;
    }
    ~hipsv_handle() { if(h) hipstatevecDestroy(h); }
    hipsv_handle(const hipsv_handle&)            = delete;
    hipsv_handle& operator=(const hipsv_handle&) = delete;
};
} // namespace

TEST(Conformance, status_values_match_custatevec_documented)
{
    EXPECT_EQ(0, HIPSTATEVEC_STATUS_SUCCESS);
    EXPECT_EQ(1, HIPSTATEVEC_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(2, HIPSTATEVEC_STATUS_ALLOC_FAILED);
    EXPECT_EQ(3, HIPSTATEVEC_STATUS_INVALID_VALUE);
    EXPECT_EQ(7, HIPSTATEVEC_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(8, HIPSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE);
    EXPECT_EQ(9, HIPSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED);
}

TEST(Conformance, pauli_enum_values_match_custatevec)
{
    EXPECT_EQ(0, HIPSTATEVEC_PAULI_I);
    EXPECT_EQ(1, HIPSTATEVEC_PAULI_X);
    EXPECT_EQ(2, HIPSTATEVEC_PAULI_Y);
    EXPECT_EQ(3, HIPSTATEVEC_PAULI_Z);
}

TEST(Conformance, init_zero_then_X_yields_one_state)
{
    if(!has_gpu()) GTEST_SKIP() << "no GPU available";
    hipsv_handle H;
    if(!H.h) GTEST_SKIP() << "hipstatevecCreate failed";
    using cd = std::complex<double>;
    constexpr uint32_t n = 1;
    constexpr size_t   N = size_t{1} << n;
    cd*   dsv = static_cast<cd*>(dev_alloc(N * sizeof(cd)));
    ASSERT_NE(dsv, nullptr);

    auto rc = hipstatevecInitializeStateVector(
        H.h, dsv, HIPSTATEVEC_C_64F, n, HIPSTATEVEC_STATE_VECTOR_TYPE_ZERO);
    if(rc == HIPSTATEVEC_STATUS_NOT_SUPPORTED) { dev_free(dsv); GTEST_SKIP() << "backend stub"; }
    ASSERT_EQ(HIPSTATEVEC_STATUS_SUCCESS, rc);

    cd      X[4]      = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    int32_t targets[1] = {0};
    rc                = hipstatevecApplyMatrix(
        H.h, dsv, HIPSTATEVEC_C_64F, n, X, HIPSTATEVEC_C_64F,
        HIPSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 1, nullptr, nullptr, 0,
        HIPSTATEVEC_COMPUTE_64F, nullptr, 0);
    ASSERT_EQ(HIPSTATEVEC_STATUS_SUCCESS, rc);

    std::vector<cd> host(N);
    dev_d2h(host.data(), dsv, N * sizeof(cd));
    EXPECT_NEAR(host[0].real(), 0.0, tol_fp64);
    EXPECT_NEAR(host[1].real(), 1.0, tol_fp64);
    dev_free(dsv);
}

TEST(Conformance, ghz_state_zz_expectation_is_unity)
{
    if(!has_gpu()) GTEST_SKIP() << "no GPU available";
    hipsv_handle H;
    if(!H.h) GTEST_SKIP() << "hipstatevecCreate failed";
    using cd = std::complex<double>;
    constexpr uint32_t n = 3;
    constexpr size_t   N = size_t{1} << n;
    cd*   dsv = static_cast<cd*>(dev_alloc(N * sizeof(cd)));
    ASSERT_NE(dsv, nullptr);

    auto rc = hipstatevecInitializeStateVector(
        H.h, dsv, HIPSTATEVEC_C_64F, n, HIPSTATEVEC_STATE_VECTOR_TYPE_GHZ);
    if(rc == HIPSTATEVEC_STATUS_NOT_SUPPORTED) { dev_free(dsv); GTEST_SKIP() << "backend stub"; }
    ASSERT_EQ(HIPSTATEVEC_STATUS_SUCCESS, rc);

    hipstatevecPauli_t  zzz[3]     = {HIPSTATEVEC_PAULI_Z, HIPSTATEVEC_PAULI_Z, HIPSTATEVEC_PAULI_Z};
    int32_t             qubits[3]  = {0, 1, 2};
    const hipstatevecPauli_t* paulis_arr[1] = {zzz};
    const int32_t*            qubits_arr[1] = {qubits};
    uint32_t                  qcounts[1]    = {3};
    double                    expvals[1]    = {0.0};

    rc = hipstatevecComputeExpectationsOnPauliBasis(
        H.h, dsv, HIPSTATEVEC_C_64F, n, expvals, paulis_arr, 1, qubits_arr, qcounts);
    ASSERT_EQ(HIPSTATEVEC_STATUS_SUCCESS, rc);
    EXPECT_NEAR(expvals[0], 1.0, tol_fp64);
    dev_free(dsv);
}
