#include "test_fixtures.hpp"

TEST(Workspace, ContractionSizeReportsPositive)
{
    if(roctn_skip_if_no_gpu()) return;
    MatMulNetwork net; net.setup_dims(4, 3, 5); net.build();
    int64_t s = 0;
    ROC_TN_EXPECT_OK(roctensornet_workspace_get_memory_size(
        net.h, net.ws, ROCTENSORNET_WORKSIZE_PREF_RECOMMENDED,
        ROCTENSORNET_MEMSPACE_DEVICE, ROCTENSORNET_WORKSPACE_SCRATCH, &s));
    EXPECT_GT(s, 0);
}

TEST(Workspace, SetGetMemory)
{
    if(roctn_skip_if_no_gpu()) return;
    MatMulNetwork net; net.setup_dims(2, 2, 2); net.build();
    void* dummy = (void*)0xDEAD;
    ROC_TN_EXPECT_OK(roctensornet_workspace_set_memory(
        net.h, net.ws, ROCTENSORNET_MEMSPACE_DEVICE,
        ROCTENSORNET_WORKSPACE_SCRATCH, dummy, 1024));
    void* got = nullptr; int64_t bytes = 0;
    ROC_TN_EXPECT_OK(roctensornet_workspace_get_memory(
        net.h, net.ws, ROCTENSORNET_MEMSPACE_DEVICE,
        ROCTENSORNET_WORKSPACE_SCRATCH, &got, &bytes));
    EXPECT_EQ(got, dummy);
    EXPECT_EQ(bytes, 1024);
}

TEST(Workspace, PurgeCacheNoOpUserOwned)
{
    if(roctn_skip_if_no_gpu()) return;
    MatMulNetwork net; net.setup_dims(2, 2, 2); net.build();
    ROC_TN_EXPECT_OK(roctensornet_workspace_purge_cache(
        net.h, net.ws, ROCTENSORNET_MEMSPACE_DEVICE));
}
