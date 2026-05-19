#include "test_helpers.hpp"

TEST(Sampler, LifecycleAndAttributes)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    roctensornet_index_t ext[2] = {2, 2};
    roctensornet_state s = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_state(
        h, ROCTENSORNET_STATE_PURITY_PURE, 2, ext, ROCTENSORNET_C_64F, &s));
    int32_t mts[2] = {0, 1};
    roctensornet_state_sampler sp = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_sampler(h, s, 2, mts, &sp));
    int32_t v = 32;
    ROC_TN_EXPECT_OK(roctensornet_sampler_configure(
        h, sp, ROCTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES, &v, sizeof(v)));
    int32_t out = 0;
    ROC_TN_EXPECT_OK(roctensornet_sampler_get_info(
        h, sp, ROCTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES, &out, sizeof(out)));
    EXPECT_EQ(out, 32);
    /* prepare/sample are not implemented in v0.1.0; both surface NOT_SUPPORTED. */
    ROC_TN_EXPECT_NOT_SUPPORTED(roctensornet_sampler_prepare(h, sp, 0, nullptr, nullptr));
    ROC_TN_EXPECT_OK(roctensornet_destroy_sampler(sp));
    ROC_TN_EXPECT_OK(roctensornet_destroy_state(s));
    ROC_TN_EXPECT_OK(roctensornet_destroy(h));
}
