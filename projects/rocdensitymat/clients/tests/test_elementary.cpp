/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "test_helpers.hpp"

TEST(ElementaryTest, CreatePauliOperators)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    int64_t mode = 2;
    rocdensitymat_elementary_operator op = nullptr;
    for(auto kind : {ROCDENSITYMAT_ELEMENTARY_IDENTITY,
                     ROCDENSITYMAT_ELEMENTARY_PAULI_X,
                     ROCDENSITYMAT_ELEMENTARY_PAULI_Y,
                     ROCDENSITYMAT_ELEMENTARY_PAULI_Z})
    {
        op = nullptr;
        ROC_DM_OK(rocdensitymat_create_elementary_operator(
            h, 1, &mode, kind, ROCDENSITYMAT_C_64F, nullptr, nullptr, &op));
        ASSERT_NE(op, nullptr);
        ROC_DM_OK(rocdensitymat_destroy_elementary_operator(op));
    }
    ROC_DM_OK(rocdensitymat_destroy(h));
}

TEST(ElementaryTest, DenseRequiresData)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    int64_t mode = 2;
    rocdensitymat_elementary_operator op = nullptr;
    EXPECT_EQ(rocdensitymat_create_elementary_operator(
                  h, 1, &mode, ROCDENSITYMAT_ELEMENTARY_DENSE,
                  ROCDENSITYMAT_C_64F, nullptr, nullptr, &op),
              ROCDENSITYMAT_STATUS_INVALID_VALUE);
    ROC_DM_OK(rocdensitymat_destroy(h));
}

TEST(ElementaryTest, DiagonalRoundTrip)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    int64_t mode = 4;
    std::vector<std::complex<double>> diag = {{1,0},{2,0},{3,0},{4,0}};
    rocdensitymat_elementary_operator op = nullptr;
    ROC_DM_OK(rocdensitymat_create_elementary_operator(
        h, 1, &mode, ROCDENSITYMAT_ELEMENTARY_DIAGONAL,
        ROCDENSITYMAT_C_64F, diag.data(), nullptr, &op));
    ROC_DM_OK(rocdensitymat_destroy_elementary_operator(op));
    ROC_DM_OK(rocdensitymat_destroy(h));
}
