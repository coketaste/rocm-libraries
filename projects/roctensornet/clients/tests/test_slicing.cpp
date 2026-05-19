#include "test_helpers.hpp"

TEST(SliceGroup, FromIdRange)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    roctensornet_slice_group g = nullptr;
    ROC_TN_EXPECT_OK(roctensornet_create_slice_group_from_id_range(h, 0, 10, 2, &g));
    EXPECT_NE(g, nullptr);
    ROC_TN_EXPECT_OK(roctensornet_destroy_slice_group(g));
    ROC_TN_EXPECT_OK(roctensornet_destroy(h));
}

TEST(SliceGroup, FromIds)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    int64_t ids[3] = {0, 3, 7};
    roctensornet_slice_group g = nullptr;
    ROC_TN_EXPECT_OK(roctensornet_create_slice_group_from_ids(h, ids, 3, &g));
    EXPECT_NE(g, nullptr);
    ROC_TN_EXPECT_OK(roctensornet_destroy_slice_group(g));
    ROC_TN_EXPECT_OK(roctensornet_destroy(h));
}

TEST(SliceGroup, ZeroStepInvalid)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    roctensornet_slice_group g = nullptr;
    EXPECT_EQ(roctensornet_create_slice_group_from_id_range(h, 0, 10, 0, &g),
              ROCTENSORNET_STATUS_INVALID_VALUE);
    ROC_TN_EXPECT_OK(roctensornet_destroy(h));
}
