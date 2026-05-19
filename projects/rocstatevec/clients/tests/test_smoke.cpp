/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Phase-1 smoke test: the library loads, version macros agree, error
 * names round-trip. Does not require a GPU.
 * ************************************************************************ */

#include <rocstatevec.h>

#include <cstring>
#include <gtest/gtest.h>

TEST(Smoke, version_macros_agree)
{
    int major = 0, minor = 0, patch = 0;
    EXPECT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_get_property(ROCSTATEVEC_PROPERTY_MAJOR_VERSION, &major));
    EXPECT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_get_property(ROCSTATEVEC_PROPERTY_MINOR_VERSION, &minor));
    EXPECT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_get_property(ROCSTATEVEC_PROPERTY_PATCH_LEVEL,   &patch));
    EXPECT_EQ(ROCSTATEVEC_VERSION_MAJOR, major);
    EXPECT_EQ(ROCSTATEVEC_VERSION_MINOR, minor);
    EXPECT_EQ(ROCSTATEVEC_VERSION_PATCH, patch);
}

TEST(Smoke, error_strings_present)
{
    EXPECT_STREQ("ROCSTATEVEC_STATUS_SUCCESS",
                 rocstatevec_get_error_name(ROCSTATEVEC_STATUS_SUCCESS));
    EXPECT_NE(nullptr, rocstatevec_get_error_string(ROCSTATEVEC_STATUS_SUCCESS));
    for(int s = 0; s < ROCSTATEVEC_STATUS_MAX_VALUE; ++s)
    {
        const char* name = rocstatevec_get_error_name(static_cast<rocstatevec_status>(s));
        const char* desc = rocstatevec_get_error_string(static_cast<rocstatevec_status>(s));
        EXPECT_NE(nullptr, name);
        EXPECT_NE(nullptr, desc);
        EXPECT_GT(std::strlen(name), 0u);
        EXPECT_GT(std::strlen(desc), 0u);
    }
}
