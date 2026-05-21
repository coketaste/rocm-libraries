/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "test_helpers.hpp"

TEST(OperatorTest, BuildHamiltonianFromSingleTerm)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    int64_t shape[1] = {2};
    rocdensitymat_elementary_operator x = nullptr;
    ROC_DM_OK(rocdensitymat_create_elementary_operator(
        h, 1, shape, ROCDENSITYMAT_ELEMENTARY_PAULI_X,
        ROCDENSITYMAT_C_64F, nullptr, nullptr, &x));
    rocdensitymat_operator_term t = nullptr;
    ROC_DM_OK(rocdensitymat_create_operator_term(h, 1, shape, &t));
    int32_t modes[1] = {0};
    int32_t dual[1]  = {ROCDENSITYMAT_DUALITY_KET};
    rocdensitymat_complex_double one{1,0};
    ROC_DM_OK(rocdensitymat_operator_term_append_elementary_product(
        h, t, 1, &x, modes, dual, one, nullptr));
    rocdensitymat_operator op = nullptr;
    ROC_DM_OK(rocdensitymat_create_operator(h, 1, shape, &op));
    ROC_DM_OK(rocdensitymat_operator_append_term(h, op, t, 0, one, nullptr));
    ROC_DM_OK(rocdensitymat_destroy_operator(op));
    ROC_DM_OK(rocdensitymat_destroy_operator_term(t));
    ROC_DM_OK(rocdensitymat_destroy_elementary_operator(x));
    ROC_DM_OK(rocdensitymat_destroy(h));
}
