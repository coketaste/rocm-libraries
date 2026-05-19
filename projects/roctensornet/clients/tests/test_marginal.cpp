#include "test_helpers.hpp"

TEST(Marginal, LifecycleAndAttributes)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    roctensornet_index_t ext[2] = {2, 2};
    roctensornet_state s = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_state(
        h, ROCTENSORNET_STATE_PURITY_PURE, 2, ext, ROCTENSORNET_C_64F, &s));
    int32_t mm[1] = {0};
    int32_t pm[1] = {1};
    roctensornet_state_marginal m = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_marginal(h, s, 1, mm, 1, pm, nullptr, &m));
    int32_t v = 8;
    ROC_TN_EXPECT_OK(roctensornet_marginal_configure(
        h, m, ROCTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES, &v, sizeof(v)));
    int32_t out = 0;
    ROC_TN_EXPECT_OK(roctensornet_marginal_get_info(
        h, m, ROCTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES, &out, sizeof(out)));
    EXPECT_EQ(out, 8);
    /* prepare/compute are not implemented in v0.1.0; both surface NOT_SUPPORTED. */
    ROC_TN_EXPECT_NOT_SUPPORTED(roctensornet_marginal_prepare(h, m, 0, nullptr, nullptr));
    ROC_TN_EXPECT_OK(roctensornet_destroy_marginal(m));
    ROC_TN_EXPECT_OK(roctensornet_destroy_state(s));
    ROC_TN_EXPECT_OK(roctensornet_destroy(h));
}
