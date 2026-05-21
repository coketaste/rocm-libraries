/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Regression oracles for v0.1.0's NOT_SUPPORTED contract: every entry
 * point that v0.1.0 explicitly defers must return
 * ROCDENSITYMAT_STATUS_NOT_SUPPORTED so v0.1.0 callers see a clean
 * error instead of an undefined silent success.
 * ************************************************************************ */

#include "test_helpers.hpp"

TEST(NotSupportedTest, EigenSpectrumPrepareIsRejected)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    int64_t shape[1] = {2};
    rocdensitymat_operator op = nullptr;
    ROC_DM_OK(rocdensitymat_create_operator(h, 1, shape, &op));
    rocdensitymat_state s = nullptr;
    ROC_DM_OK(rocdensitymat_create_state(
        h, ROCDENSITYMAT_STATE_PURITY_PURE, 1, shape, 1,
        ROCDENSITYMAT_C_64F, &s));
    rocdensitymat_workspace_descriptor ws = nullptr;
    ROC_DM_OK(rocdensitymat_create_workspace(h, &ws));
    ROC_DM_EXPECT_NOT_SUPPORTED(rocdensitymat_operator_prepare_eigenspectrum(
        h, op, s, 1, ROCDENSITYMAT_COMPUTE_64F, 0, ws));
    ROC_DM_OK(rocdensitymat_destroy_workspace(ws));
    ROC_DM_OK(rocdensitymat_destroy_state(s));
    ROC_DM_OK(rocdensitymat_destroy_operator(op));
    ROC_DM_OK(rocdensitymat_destroy(h));
}

TEST(NotSupportedTest, BackwardDiffPrepareIsRejected)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    int64_t shape[1] = {2};
    rocdensitymat_operator op = nullptr;
    ROC_DM_OK(rocdensitymat_create_operator(h, 1, shape, &op));
    rocdensitymat_state s = nullptr;
    ROC_DM_OK(rocdensitymat_create_state(
        h, ROCDENSITYMAT_STATE_PURITY_PURE, 1, shape, 1,
        ROCDENSITYMAT_C_64F, &s));
    rocdensitymat_workspace_descriptor ws = nullptr;
    ROC_DM_OK(rocdensitymat_create_workspace(h, &ws));
    ROC_DM_EXPECT_NOT_SUPPORTED(rocdensitymat_operator_prepare_action_backward_diff(
        h, op, s, s, ROCDENSITYMAT_COMPUTE_64F, 0, ws));
    ROC_DM_OK(rocdensitymat_destroy_workspace(ws));
    ROC_DM_OK(rocdensitymat_destroy_state(s));
    ROC_DM_OK(rocdensitymat_destroy_operator(op));
    ROC_DM_OK(rocdensitymat_destroy(h));
}

TEST(NotSupportedTest, MpiDistributedConfigIsRejected)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    ROC_DM_EXPECT_NOT_SUPPORTED(rocdensitymat_reset_distributed_configuration(
        h, ROCDENSITYMAT_DISTRIBUTED_PROVIDER_MPI, nullptr, 0));
    ROC_DM_OK(rocdensitymat_destroy(h));
}

TEST(NotSupportedTest, NoneDistributedConfigIsAccepted)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    ROC_DM_OK(rocdensitymat_reset_distributed_configuration(
        h, ROCDENSITYMAT_DISTRIBUTED_PROVIDER_NONE, nullptr, 0));
    int32_t nr = 0, r = -1;
    ROC_DM_OK(rocdensitymat_get_num_ranks(h, &nr));
    ROC_DM_OK(rocdensitymat_get_proc_rank(h, &r));
    EXPECT_EQ(nr, 1);
    EXPECT_EQ(r, 0);
    ROC_DM_OK(rocdensitymat_destroy(h));
}
