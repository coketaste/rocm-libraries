/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * hipDENSITYMAT smoke test: header is self-contained, library loads,
 * status enum values agree with cuDensityMat for the documented codes.
 * ************************************************************************ */

#include <hipdensitymat.h>
#include <gtest/gtest.h>

TEST(HipdensitymatSmoke, status_values_are_stable)
{
    EXPECT_EQ(0, HIPDENSITYMAT_STATUS_SUCCESS);
    EXPECT_EQ(1, HIPDENSITYMAT_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(2, HIPDENSITYMAT_STATUS_ALLOC_FAILED);
    EXPECT_EQ(3, HIPDENSITYMAT_STATUS_INVALID_VALUE);
    EXPECT_EQ(7, HIPDENSITYMAT_STATUS_NOT_SUPPORTED);
}

TEST(HipdensitymatSmoke, version_macros_set)
{
    int v = 0;
    EXPECT_EQ(HIPDENSITYMAT_STATUS_SUCCESS,
              hipdensitymatGetProperty(HIPDENSITYMAT_PROPERTY_MAJOR_VERSION, &v));
    EXPECT_EQ(HIPDENSITYMAT_VERSION_MAJOR, v);
}

TEST(HipdensitymatSmoke, error_strings_present)
{
    EXPECT_NE(nullptr, hipdensitymatGetErrorName(HIPDENSITYMAT_STATUS_SUCCESS));
    EXPECT_NE(nullptr, hipdensitymatGetErrorString(HIPDENSITYMAT_STATUS_SUCCESS));
    EXPECT_NE(nullptr, hipdensitymatGetErrorName(HIPDENSITYMAT_STATUS_NOT_SUPPORTED));
}

TEST(HipdensitymatSmoke, create_destroy)
{
    hipdensitymatHandle_t h = nullptr;
    auto rc = hipdensitymatCreate(&h);
    if(rc == HIPDENSITYMAT_STATUS_NOT_SUPPORTED) GTEST_SKIP() << "no GPU?";
    EXPECT_EQ(HIPDENSITYMAT_STATUS_SUCCESS, rc);
    EXPECT_EQ(HIPDENSITYMAT_STATUS_SUCCESS, hipdensitymatDestroy(h));
}
