/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "test_helpers.hpp"

namespace
{
struct StateFixture : public ::testing::Test
{
    rocdensitymat_handle h = nullptr;
    void SetUp() override { ROC_DM_OK(rocdensitymat_create(&h)); }
    void TearDown() override { ROC_DM_OK(rocdensitymat_destroy(h)); }
};
}

TEST_F(StateFixture, CreatePureC64)
{
    int64_t shape[2] = {2, 2};
    rocdensitymat_state s = nullptr;
    ROC_DM_OK(rocdensitymat_create_state(
        h, ROCDENSITYMAT_STATE_PURITY_PURE, 2, shape, 1,
        ROCDENSITYMAT_C_64F, &s));
    int32_t nc = 0;
    ROC_DM_OK(rocdensitymat_state_get_num_components(h, s, &nc));
    EXPECT_EQ(nc, 1);
    int32_t nm = 0;
    size_t  sz = 0;
    ROC_DM_OK(rocdensitymat_state_get_component_info(h, s, 0, &nm, nullptr, &sz));
    EXPECT_EQ(nm, 2);
    EXPECT_EQ(sz, 4 * sizeof(hipDoubleComplex));
    ROC_DM_OK(rocdensitymat_destroy_state(s));
}

TEST_F(StateFixture, MpsIsNotSupported)
{
    int64_t shape[1] = {2};
    rocdensitymat_state s = nullptr;
    ROC_DM_EXPECT_NOT_SUPPORTED(rocdensitymat_create_state(
        h, ROCDENSITYMAT_STATE_PURITY_MPS, 1, shape, 1,
        ROCDENSITYMAT_C_64F, &s));
}

TEST_F(StateFixture, InitializeBasisAndOverlap)
{
    int64_t shape[1] = {2};
    rocdensitymat_state s = nullptr;
    ROC_DM_OK(rocdensitymat_create_state(
        h, ROCDENSITYMAT_STATE_PURITY_PURE, 1, shape, 1,
        ROCDENSITYMAT_C_64F, &s));
    void* d = nullptr;
    HIP_OK(hipMalloc(&d, 2 * sizeof(hipDoubleComplex)));
    ROC_DM_OK(rocdensitymat_state_attach_component_buffer(
        h, s, 0, d, 2 * sizeof(hipDoubleComplex)));
    int64_t basis[1] = {1};
    ROC_DM_OK(rocdensitymat_state_initialize_basis(h, s, basis));

    rocdensitymat_workspace_descriptor ws = nullptr;
    ROC_DM_OK(rocdensitymat_create_workspace(h, &ws));
    rocdensitymat_complex_double ov{0,0};
    ROC_DM_OK(rocdensitymat_state_compute_overlap(h, s, s, ws, &ov));
    EXPECT_NEAR(ov.x, 1.0, 1e-9);
    EXPECT_NEAR(ov.y, 0.0, 1e-9);

    double n = 0;
    ROC_DM_OK(rocdensitymat_state_compute_norm(h, s, ws, &n));
    EXPECT_NEAR(n, 1.0, 1e-9);

    HIP_OK(hipFree(d));
    ROC_DM_OK(rocdensitymat_destroy_workspace(ws));
    ROC_DM_OK(rocdensitymat_destroy_state(s));
}

TEST_F(StateFixture, MixedTraceUniformIsOne)
{
    int64_t shape[1] = {3};
    rocdensitymat_state s = nullptr;
    ROC_DM_OK(rocdensitymat_create_state(
        h, ROCDENSITYMAT_STATE_PURITY_MIXED, 1, shape, 1,
        ROCDENSITYMAT_C_64F, &s));
    void* d = nullptr;
    HIP_OK(hipMalloc(&d, 9 * sizeof(hipDoubleComplex)));
    ROC_DM_OK(rocdensitymat_state_attach_component_buffer(
        h, s, 0, d, 9 * sizeof(hipDoubleComplex)));
    ROC_DM_OK(rocdensitymat_state_initialize_uniform(h, s));

    rocdensitymat_workspace_descriptor ws = nullptr;
    ROC_DM_OK(rocdensitymat_create_workspace(h, &ws));
    rocdensitymat_complex_double tr{0,0};
    ROC_DM_OK(rocdensitymat_state_compute_trace(h, s, ws, &tr));
    EXPECT_NEAR(tr.x, 1.0, 1e-9);
    EXPECT_NEAR(tr.y, 0.0, 1e-9);

    HIP_OK(hipFree(d));
    ROC_DM_OK(rocdensitymat_destroy_workspace(ws));
    ROC_DM_OK(rocdensitymat_destroy_state(s));
}
