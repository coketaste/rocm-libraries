#include "test_helpers.hpp"

TEST(NetworkState, Lifecycle)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    roctensornet_index_t ext[2] = {2, 2};
    roctensornet_state s = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_state(
        h, ROCTENSORNET_STATE_PURITY_PURE, 2, ext, ROCTENSORNET_C_64F, &s));
    EXPECT_NE(s, nullptr);
    int32_t hs = 16;
    ROC_TN_EXPECT_OK(roctensornet_state_configure(
        h, s, ROCTENSORNET_STATE_CONFIG_NUM_HYPER_SAMPLES, &hs, sizeof(hs)));
    int32_t out = 0;
    ROC_TN_EXPECT_OK(roctensornet_state_get_info(
        h, s, ROCTENSORNET_STATE_CONFIG_NUM_HYPER_SAMPLES, &out, sizeof(out)));
    EXPECT_EQ(out, 16);
    ROC_TN_EXPECT_OK(roctensornet_destroy_state(s));
    ROC_TN_EXPECT_OK(roctensornet_destroy(h));
}

TEST(NetworkState, ApplyTensorRecordsGate)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    roctensornet_index_t ext[2] = {2, 2};
    roctensornet_state s = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_state(
        h, ROCTENSORNET_STATE_PURITY_PURE, 2, ext, ROCTENSORNET_C_64F, &s));
    int32_t modes[1] = {0};
    int64_t tid = -1;
    /* gate data is opaque (device or host) and not dereferenced here */
    ROC_TN_EXPECT_OK(roctensornet_state_apply_tensor_operator(
        h, s, 1, modes, /*tensor_data*/(void*)0x100,
        /*strides*/nullptr, /*immutable*/0, /*adjoint*/0, /*unitary*/1, &tid));
    EXPECT_EQ(tid, 0);
    /* update */
    ROC_TN_EXPECT_OK(roctensornet_state_update_tensor_operator(h, s, tid, (void*)0x200, 1));
    ROC_TN_EXPECT_OK(roctensornet_destroy_state(s));
    ROC_TN_EXPECT_OK(roctensornet_destroy(h));
}
