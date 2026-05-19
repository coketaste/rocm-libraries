/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * hipSTATEVEC smoke test: header is self-contained, library loads,
 * status enum values agree with cuStateVec for the documented stable codes.
 * ************************************************************************ */

#include <hipstatevec.h>
#include <gtest/gtest.h>

TEST(HipstatevecSmoke, status_values_are_stable)
{
    EXPECT_EQ(0, HIPSTATEVEC_STATUS_SUCCESS);
    EXPECT_EQ(1, HIPSTATEVEC_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(2, HIPSTATEVEC_STATUS_ALLOC_FAILED);
    EXPECT_EQ(3, HIPSTATEVEC_STATUS_INVALID_VALUE);
}

TEST(HipstatevecSmoke, version_macros_set)
{
    int v = 0;
    EXPECT_EQ(HIPSTATEVEC_STATUS_SUCCESS, hipstatevecGetProperty(HIPSTATEVEC_PROPERTY_MAJOR_VERSION, &v));
    EXPECT_EQ(HIPSTATEVEC_VERSION_MAJOR, v);
}

TEST(HipstatevecSmoke, error_strings_present)
{
    EXPECT_NE(nullptr, hipstatevecGetErrorName(HIPSTATEVEC_STATUS_SUCCESS));
    EXPECT_NE(nullptr, hipstatevecGetErrorString(HIPSTATEVEC_STATUS_SUCCESS));
}

TEST(HipstatevecSmoke, create_destroy)
{
    hipstatevecHandle_t h = nullptr;
    auto rc = hipstatevecCreate(&h);
    if(rc == HIPSTATEVEC_STATUS_NOT_SUPPORTED) GTEST_SKIP() << "no GPU?";
    EXPECT_EQ(HIPSTATEVEC_STATUS_SUCCESS, rc);
    EXPECT_EQ(HIPSTATEVEC_STATUS_SUCCESS, hipstatevecDestroy(h));
}
