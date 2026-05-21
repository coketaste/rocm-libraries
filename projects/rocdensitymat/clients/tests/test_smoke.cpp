/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "test_helpers.hpp"

TEST(SmokeTest, VersionAndProperty)
{
    size_t v = rocdensitymat_get_version();
    EXPECT_GT(v, size_t{0});

    int32_t major = -1, minor = -1, patch = -1;
    ROC_DM_OK(rocdensitymat_get_property(ROCDENSITYMAT_PROPERTY_MAJOR_VERSION, &major));
    ROC_DM_OK(rocdensitymat_get_property(ROCDENSITYMAT_PROPERTY_MINOR_VERSION, &minor));
    ROC_DM_OK(rocdensitymat_get_property(ROCDENSITYMAT_PROPERTY_PATCH_LEVEL,   &patch));
    EXPECT_GE(major, 0);
    EXPECT_GE(minor, 0);
    EXPECT_GE(patch, 0);
}

TEST(SmokeTest, ErrorStrings)
{
    EXPECT_STRNE(rocdensitymat_get_error_name(ROCDENSITYMAT_STATUS_SUCCESS),  "");
    EXPECT_STRNE(rocdensitymat_get_error_string(ROCDENSITYMAT_STATUS_NOT_SUPPORTED), "");
}
