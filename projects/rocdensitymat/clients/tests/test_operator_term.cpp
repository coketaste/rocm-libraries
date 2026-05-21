/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "test_helpers.hpp"

TEST(OperatorTermTest, AppendSingleQubitProduct)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    int64_t shape[1] = {2};
    rocdensitymat_elementary_operator x = nullptr;
    ROC_DM_OK(rocdensitymat_create_elementary_operator(
        h, 1, shape, ROCDENSITYMAT_ELEMENTARY_PAULI_X,
        ROCDENSITYMAT_C_64F, nullptr, nullptr, &x));
    rocdensitymat_operator_term term = nullptr;
    ROC_DM_OK(rocdensitymat_create_operator_term(h, 1, shape, &term));
    int32_t modes[1] = {0};
    int32_t dual[1]  = {ROCDENSITYMAT_DUALITY_KET};
    rocdensitymat_complex_double one{1,0};
    ROC_DM_OK(rocdensitymat_operator_term_append_elementary_product(
        h, term, 1, &x, modes, dual, one, nullptr));
    ROC_DM_OK(rocdensitymat_destroy_operator_term(term));
    ROC_DM_OK(rocdensitymat_destroy_elementary_operator(x));
    ROC_DM_OK(rocdensitymat_destroy(h));
}

TEST(OperatorTermTest, ExtentsMustMatch)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    int64_t term_shape[1] = {3};
    int64_t op_extent     = 2;
    rocdensitymat_elementary_operator x = nullptr;
    ROC_DM_OK(rocdensitymat_create_elementary_operator(
        h, 1, &op_extent, ROCDENSITYMAT_ELEMENTARY_PAULI_X,
        ROCDENSITYMAT_C_64F, nullptr, nullptr, &x));
    rocdensitymat_operator_term term = nullptr;
    ROC_DM_OK(rocdensitymat_create_operator_term(h, 1, term_shape, &term));
    int32_t modes[1] = {0};
    int32_t dual[1]  = {ROCDENSITYMAT_DUALITY_KET};
    rocdensitymat_complex_double one{1,0};
    EXPECT_EQ(rocdensitymat_operator_term_append_elementary_product(
                  h, term, 1, &x, modes, dual, one, nullptr),
              ROCDENSITYMAT_STATUS_INVALID_VALUE);
    ROC_DM_OK(rocdensitymat_destroy_operator_term(term));
    ROC_DM_OK(rocdensitymat_destroy_elementary_operator(x));
    ROC_DM_OK(rocdensitymat_destroy(h));
}
