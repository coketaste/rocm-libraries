/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "test_helpers.hpp"

TEST(PropertiesTest, GetPurityAndDataType)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    int64_t shape[1] = {2};
    rocdensitymat_state s = nullptr;
    ROC_DM_OK(rocdensitymat_create_state(
        h, ROCDENSITYMAT_STATE_PURITY_MIXED, 1, shape, 1,
        ROCDENSITYMAT_C_32F, &s));
    rocdensitymat_state_purity p = ROCDENSITYMAT_STATE_PURITY_PURE;
    ROC_DM_OK(rocdensitymat_state_get_purity(h, s, &p));
    EXPECT_EQ(p, ROCDENSITYMAT_STATE_PURITY_MIXED);

    rocdensitymat_data_type dt = ROCDENSITYMAT_R_32F;
    ROC_DM_OK(rocdensitymat_state_get_data_type(h, s, &dt));
    EXPECT_EQ(dt, ROCDENSITYMAT_C_32F);

    int32_t nm = -1;
    int64_t shape_out[1] = {0};
    ROC_DM_OK(rocdensitymat_state_get_space_shape(h, s, &nm, shape_out));
    EXPECT_EQ(nm, 1);
    EXPECT_EQ(shape_out[0], 2);

    ROC_DM_OK(rocdensitymat_destroy_state(s));
    ROC_DM_OK(rocdensitymat_destroy(h));
}
