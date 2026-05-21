/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * compute_action with a Lindblad-collapse term must report NOT_SUPPORTED.
 * This is the regression oracle for the v0.1.0 deferral contract.
 * ************************************************************************ */

#include "test_helpers.hpp"

TEST(ComputeActionTest, LindbladCollapseIsRejected)
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
    rocdensitymat_operator H = nullptr;
    ROC_DM_OK(rocdensitymat_create_operator(h, 1, shape, &H));
    // duality_offset != 0 marks this as a Lindblad-collapse term.
    ROC_DM_OK(rocdensitymat_operator_append_term(h, H, t, /*duality_offset*/1, one, nullptr));

    rocdensitymat_state rho = nullptr;
    ROC_DM_OK(rocdensitymat_create_state(
        h, ROCDENSITYMAT_STATE_PURITY_MIXED, 1, shape, 1,
        ROCDENSITYMAT_C_64F, &rho));
    rocdensitymat_workspace_descriptor ws = nullptr;
    ROC_DM_OK(rocdensitymat_create_workspace(h, &ws));
    ROC_DM_EXPECT_NOT_SUPPORTED(rocdensitymat_operator_prepare_action(
        h, H, rho, rho, ROCDENSITYMAT_COMPUTE_64F, 0, ws));

    ROC_DM_OK(rocdensitymat_destroy_workspace(ws));
    ROC_DM_OK(rocdensitymat_destroy_state(rho));
    ROC_DM_OK(rocdensitymat_destroy_operator(H));
    ROC_DM_OK(rocdensitymat_destroy_operator_term(t));
    ROC_DM_OK(rocdensitymat_destroy_elementary_operator(x));
    ROC_DM_OK(rocdensitymat_destroy(h));
}
