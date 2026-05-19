#include "test_fixtures.hpp"

TEST(WorkspaceQuery, ReturnsPositiveSize)
{
    if(roctn_skip_if_no_gpu()) return;
    MatMulNetwork net; net.setup_dims(4, 4, 4); net.build();
    int64_t scratch = 0, cache = 0;
    ROC_TN_EXPECT_OK(roctensornet_workspace_get_memory_size(
        net.h, net.ws, ROCTENSORNET_WORKSIZE_PREF_RECOMMENDED,
        ROCTENSORNET_MEMSPACE_DEVICE, ROCTENSORNET_WORKSPACE_SCRATCH, &scratch));
    ROC_TN_EXPECT_OK(roctensornet_workspace_get_memory_size(
        net.h, net.ws, ROCTENSORNET_WORKSIZE_PREF_RECOMMENDED,
        ROCTENSORNET_MEMSPACE_DEVICE, ROCTENSORNET_WORKSPACE_CACHE, &cache));
    EXPECT_GT(scratch, 0);
    EXPECT_GT(cache, 0);
}

TEST(WorkspaceQuery, SVDSizeMatchesDescriptors)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    int32_t modes_in[2] = {0, 1}, modes_u[2] = {0, 2}, modes_v[2] = {1, 2};
    roctensornet_index_t ein[2] = {4, 3}, eu[2] = {4, 3}, ev[2] = {3, 3};
    roctensornet_tensor_descriptor tin, tu, tv;
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_descriptor(h, 2, ein, nullptr, modes_in, ROCTENSORNET_R_64F, &tin));
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_descriptor(h, 2, eu,  nullptr, modes_u,  ROCTENSORNET_R_64F, &tu));
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_descriptor(h, 2, ev,  nullptr, modes_v,  ROCTENSORNET_R_64F, &tv));
    roctensornet_workspace_descriptor ws = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_workspace_descriptor(h, &ws));
    ROC_TN_EXPECT_OK(roctensornet_workspace_compute_svd_sizes(h, tin, tu, tv, nullptr, ws));
    int64_t s = 0;
    ROC_TN_EXPECT_OK(roctensornet_workspace_get_memory_size(
        h, ws, ROCTENSORNET_WORKSIZE_PREF_RECOMMENDED,
        ROCTENSORNET_MEMSPACE_DEVICE, ROCTENSORNET_WORKSPACE_SCRATCH, &s));
    EXPECT_GT(s, 0);
    roctensornet_destroy_workspace_descriptor(ws);
    roctensornet_destroy_tensor_descriptor(tin);
    roctensornet_destroy_tensor_descriptor(tu);
    roctensornet_destroy_tensor_descriptor(tv);
    roctensornet_destroy(h);
}
