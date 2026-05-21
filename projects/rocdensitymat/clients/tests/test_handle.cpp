/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "test_helpers.hpp"

TEST(HandleTest, CreateAndDestroy)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    ASSERT_NE(h, nullptr);
    ROC_DM_OK(rocdensitymat_destroy(h));
}

TEST(HandleTest, SetAndGetStream)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    hipStream_t s = nullptr;
    HIP_OK(hipStreamCreate(&s));
    ROC_DM_OK(rocdensitymat_set_stream(h, s));
    hipStream_t got = nullptr;
    ROC_DM_OK(rocdensitymat_get_stream(h, &got));
    EXPECT_EQ(s, got);
    HIP_OK(hipStreamDestroy(s));
    ROC_DM_OK(rocdensitymat_destroy(h));
}

TEST(HandleTest, ResetRandomSeedAcceptsAnyValue)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    ROC_DM_OK(rocdensitymat_reset_random_seed(h, 42));
    ROC_DM_OK(rocdensitymat_reset_random_seed(h, 0));
    ROC_DM_OK(rocdensitymat_destroy(h));
}

TEST(HandleTest, NullPointerArguments)
{
    EXPECT_EQ(rocdensitymat_create(nullptr), ROCDENSITYMAT_STATUS_INVALID_VALUE);
    EXPECT_EQ(rocdensitymat_set_stream(nullptr, nullptr),
              ROCDENSITYMAT_STATUS_NOT_INITIALIZED);
}
