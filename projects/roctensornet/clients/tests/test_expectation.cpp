#include "test_helpers.hpp"

TEST(Expectation, LifecycleAndAttributes)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    roctensornet_index_t ext[2] = {2, 2};
    roctensornet_state s = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_state(
        h, ROCTENSORNET_STATE_PURITY_PURE, 2, ext, ROCTENSORNET_C_64F, &s));
    roctensornet_network_operator op = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_network_operator(
        h, 2, ext, ROCTENSORNET_C_64F, &op));

    roctensornet_state_expectation ex = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_expectation(h, s, op, &ex));
    int32_t v = 4;
    ROC_TN_EXPECT_OK(roctensornet_expectation_configure(
        h, ex, ROCTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES, &v, sizeof(v)));
    int32_t out = 0;
    ROC_TN_EXPECT_OK(roctensornet_expectation_get_info(
        h, ex, ROCTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES, &out, sizeof(out)));
    EXPECT_EQ(out, 4);
    /* prepare/compute are not implemented in v0.1.0; both surface NOT_SUPPORTED. */
    ROC_TN_EXPECT_NOT_SUPPORTED(roctensornet_expectation_prepare(h, ex, 0, nullptr, nullptr));
    ROC_TN_EXPECT_OK(roctensornet_destroy_expectation(ex));
    ROC_TN_EXPECT_OK(roctensornet_destroy_network_operator(op));
    ROC_TN_EXPECT_OK(roctensornet_destroy_state(s));
    ROC_TN_EXPECT_OK(roctensornet_destroy(h));
}
