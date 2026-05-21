/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Conformance tests for hipDENSITYMAT.
 *
 * Every shared status code, enum, and library property must take
 * exactly the documented cuDensityMat 0.1.0 / cuQuantum 24.11 numeric
 * value. Static-asserting them in addition to runtime tests means the
 * conformance contract is checked at compile time, so an ABI-breaking
 * change to one of the enums fails the build, not just the test run.
 * ************************************************************************ */

#include <hipdensitymat.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

/* ---- compile-time pins on status_t numeric values ---- */

static_assert(HIPDENSITYMAT_STATUS_SUCCESS                 == 0,  "");
static_assert(HIPDENSITYMAT_STATUS_NOT_INITIALIZED         == 1,  "");
static_assert(HIPDENSITYMAT_STATUS_ALLOC_FAILED            == 2,  "");
static_assert(HIPDENSITYMAT_STATUS_INVALID_VALUE           == 3,  "");
static_assert(HIPDENSITYMAT_STATUS_ARCH_MISMATCH           == 4,  "");
static_assert(HIPDENSITYMAT_STATUS_EXECUTION_FAILED        == 5,  "");
static_assert(HIPDENSITYMAT_STATUS_INTERNAL_ERROR          == 6,  "");
static_assert(HIPDENSITYMAT_STATUS_NOT_SUPPORTED           == 7,  "");
static_assert(HIPDENSITYMAT_STATUS_CALLBACK_ERROR          == 8,  "");
static_assert(HIPDENSITYMAT_STATUS_BLAS_ERROR              == 9,  "");
static_assert(HIPDENSITYMAT_STATUS_HIP_ERROR               == 10, "");
static_assert(HIPDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE  == 11, "");
static_assert(HIPDENSITYMAT_STATUS_INSUFFICIENT_DRIVER     == 12, "");
static_assert(HIPDENSITYMAT_STATUS_IO_ERROR                == 13, "");
static_assert(HIPDENSITYMAT_STATUS_NO_DEVICE_ALLOCATOR     == 15, "");
static_assert(HIPDENSITYMAT_STATUS_DEVICE_ALLOCATOR_ERROR  == 19, "");
static_assert(HIPDENSITYMAT_STATUS_DISTRIBUTED_FAILURE     == 20, "");
static_assert(HIPDENSITYMAT_STATUS_INTERRUPTED             == 21, "");
static_assert(HIPDENSITYMAT_STATUS_DRIVER_VERSION_MISMATCH == 23, "");

/* ---- compute type ---- */

static_assert(HIPDENSITYMAT_COMPUTE_DEFAULT == 0,        "");
static_assert(HIPDENSITYMAT_COMPUTE_32F     == (1U << 2),"");
static_assert(HIPDENSITYMAT_COMPUTE_64F     == (1U << 4),"");

/* ---- data type ---- */

static_assert(HIPDENSITYMAT_R_32F == 0, "");
static_assert(HIPDENSITYMAT_R_64F == 1, "");
static_assert(HIPDENSITYMAT_C_32F == 4, "");
static_assert(HIPDENSITYMAT_C_64F == 5, "");

/* ---- purity ---- */

static_assert(HIPDENSITYMAT_STATE_PURITY_PURE  == 0, "");
static_assert(HIPDENSITYMAT_STATE_PURITY_MIXED == 1, "");
static_assert(HIPDENSITYMAT_STATE_PURITY_MPS   == 2, "");

/* ---- memspace + workspace kind ---- */

static_assert(HIPDENSITYMAT_MEMSPACE_DEVICE == 0, "");
static_assert(HIPDENSITYMAT_MEMSPACE_HOST   == 1, "");

static_assert(HIPDENSITYMAT_WORKSPACE_SCRATCH == 0, "");
static_assert(HIPDENSITYMAT_WORKSPACE_CACHE   == 1, "");

/* ---- distributed provider ---- */

static_assert(HIPDENSITYMAT_DISTRIBUTED_PROVIDER_NONE == 0, "");
static_assert(HIPDENSITYMAT_DISTRIBUTED_PROVIDER_MPI  == 1, "");
static_assert(HIPDENSITYMAT_DISTRIBUTED_PROVIDER_NCCL == 2, "");
static_assert(HIPDENSITYMAT_DISTRIBUTED_PROVIDER_RCCL == 3, "");

/* ---- duality flag ---- */

static_assert(HIPDENSITYMAT_DUALITY_KET == 0, "");
static_assert(HIPDENSITYMAT_DUALITY_BRA == 1, "");

/* ---- library property selector ---- */

static_assert(HIPDENSITYMAT_PROPERTY_MAJOR_VERSION == 0, "");
static_assert(HIPDENSITYMAT_PROPERTY_MINOR_VERSION == 1, "");
static_assert(HIPDENSITYMAT_PROPERTY_PATCH_LEVEL   == 2, "");

/* ---- complex layout ---- */

static_assert(sizeof(hipdensitymatComplexDouble_t) == 2 * sizeof(double), "");
static_assert(sizeof(hipdensitymatComplexFloat_t)  == 2 * sizeof(float),  "");

/* ---- runtime conformance ---- */

TEST(Conformance, status_values_match_cudensitymat)
{
    EXPECT_EQ(0,  static_cast<int>(HIPDENSITYMAT_STATUS_SUCCESS));
    EXPECT_EQ(7,  static_cast<int>(HIPDENSITYMAT_STATUS_NOT_SUPPORTED));
    EXPECT_EQ(11, static_cast<int>(HIPDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE));
}

TEST(Conformance, purity_values_match_cudensitymat)
{
    EXPECT_EQ(0, static_cast<int>(HIPDENSITYMAT_STATE_PURITY_PURE));
    EXPECT_EQ(1, static_cast<int>(HIPDENSITYMAT_STATE_PURITY_MIXED));
    EXPECT_EQ(2, static_cast<int>(HIPDENSITYMAT_STATE_PURITY_MPS));
}

TEST(Conformance, workspace_enums_match_cudensitymat)
{
    EXPECT_EQ(0, static_cast<int>(HIPDENSITYMAT_MEMSPACE_DEVICE));
    EXPECT_EQ(1, static_cast<int>(HIPDENSITYMAT_MEMSPACE_HOST));
    EXPECT_EQ(0, static_cast<int>(HIPDENSITYMAT_WORKSPACE_SCRATCH));
    EXPECT_EQ(1, static_cast<int>(HIPDENSITYMAT_WORKSPACE_CACHE));
}

TEST(Conformance, mps_purity_returns_not_supported)
{
    hipdensitymatHandle_t h = nullptr;
    auto rc = hipdensitymatCreate(&h);
    if(rc == HIPDENSITYMAT_STATUS_NOT_SUPPORTED) GTEST_SKIP() << "no GPU";
    ASSERT_EQ(HIPDENSITYMAT_STATUS_SUCCESS, rc);

    int64_t              shape[2] = {2, 2};
    hipdensitymatState_t s        = nullptr;
    auto                 rc_s     = hipdensitymatCreateState(
        h, HIPDENSITYMAT_STATE_PURITY_MPS, 2, shape, 1,
        HIPDENSITYMAT_C_64F, &s);
    EXPECT_EQ(HIPDENSITYMAT_STATUS_NOT_SUPPORTED, rc_s);

    EXPECT_EQ(HIPDENSITYMAT_STATUS_SUCCESS, hipdensitymatDestroy(h));
}

TEST(Conformance, distributed_returns_not_supported)
{
    hipdensitymatHandle_t h = nullptr;
    auto rc = hipdensitymatCreate(&h);
    if(rc == HIPDENSITYMAT_STATUS_NOT_SUPPORTED) GTEST_SKIP() << "no GPU";
    ASSERT_EQ(HIPDENSITYMAT_STATUS_SUCCESS, rc);

    int32_t nranks = -1;
    EXPECT_EQ(HIPDENSITYMAT_STATUS_NOT_SUPPORTED,
              hipdensitymatGetNumRanks(h, &nranks));

    EXPECT_EQ(HIPDENSITYMAT_STATUS_SUCCESS, hipdensitymatDestroy(h));
}
