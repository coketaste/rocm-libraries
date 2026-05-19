#include "test_helpers.hpp"

TEST(Accessor, LifecycleAndAttributes)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    roctensornet_index_t ext[2] = {2, 2};
    roctensornet_state s = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_state(
        h, ROCTENSORNET_STATE_PURITY_PURE, 2, ext, ROCTENSORNET_C_64F, &s));
    int32_t pm[1] = {0};
    roctensornet_state_accessor a = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_accessor(h, s, 1, pm, nullptr, &a));
    int32_t v = 16;
    ROC_TN_EXPECT_OK(roctensornet_accessor_configure(
        h, a, ROCTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES, &v, sizeof(v)));
    int32_t out = 0;
    ROC_TN_EXPECT_OK(roctensornet_accessor_get_info(
        h, a, ROCTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES, &out, sizeof(out)));
    EXPECT_EQ(out, 16);
    /* prepare/compute are not implemented in v0.1.0; both surface NOT_SUPPORTED. */
    ROC_TN_EXPECT_NOT_SUPPORTED(roctensornet_accessor_prepare(h, a, 0, nullptr, nullptr));
    ROC_TN_EXPECT_OK(roctensornet_destroy_accessor(a));
    ROC_TN_EXPECT_OK(roctensornet_destroy_state(s));
    ROC_TN_EXPECT_OK(roctensornet_destroy(h));
}
