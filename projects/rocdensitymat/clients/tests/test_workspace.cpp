/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "test_helpers.hpp"

TEST(WorkspaceTest, CreateAndDestroy)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    rocdensitymat_workspace_descriptor ws = nullptr;
    ROC_DM_OK(rocdensitymat_create_workspace(h, &ws));
    ASSERT_NE(ws, nullptr);
    ROC_DM_OK(rocdensitymat_destroy_workspace(ws));
    ROC_DM_OK(rocdensitymat_destroy(h));
}

TEST(WorkspaceTest, GetAndSetMemory)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    rocdensitymat_workspace_descriptor ws = nullptr;
    ROC_DM_OK(rocdensitymat_create_workspace(h, &ws));

    void* d = nullptr;
    HIP_OK(hipMalloc(&d, 1024));
    ROC_DM_OK(rocdensitymat_workspace_set_memory(
        h, ws, ROCDENSITYMAT_MEMSPACE_DEVICE,
        ROCDENSITYMAT_WORKSPACE_SCRATCH, d, 1024));

    void*  got      = nullptr;
    size_t got_size = 0;
    ROC_DM_OK(rocdensitymat_workspace_get_memory(
        h, ws, ROCDENSITYMAT_MEMSPACE_DEVICE,
        ROCDENSITYMAT_WORKSPACE_SCRATCH, &got, &got_size));
    EXPECT_EQ(got, d);
    EXPECT_EQ(got_size, size_t{1024});

    size_t initial_size = 1234;
    ROC_DM_OK(rocdensitymat_workspace_get_memory_size(
        h, ws, ROCDENSITYMAT_MEMSPACE_DEVICE,
        ROCDENSITYMAT_WORKSPACE_SCRATCH, &initial_size));
    EXPECT_EQ(initial_size, size_t{0}); // not yet prepared

    HIP_OK(hipFree(d));
    ROC_DM_OK(rocdensitymat_destroy_workspace(ws));
    ROC_DM_OK(rocdensitymat_destroy(h));
}

TEST(WorkspaceTest, CacheKindIsAcceptedAsNoop)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    rocdensitymat_workspace_descriptor ws = nullptr;
    ROC_DM_OK(rocdensitymat_create_workspace(h, &ws));
    size_t s = 999;
    ROC_DM_OK(rocdensitymat_workspace_get_memory_size(
        h, ws, ROCDENSITYMAT_MEMSPACE_DEVICE,
        ROCDENSITYMAT_WORKSPACE_CACHE, &s));
    EXPECT_EQ(s, size_t{0});
    ROC_DM_OK(rocdensitymat_destroy_workspace(ws));
    ROC_DM_OK(rocdensitymat_destroy(h));
}
