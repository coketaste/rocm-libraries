#include "test_helpers.hpp"

TEST(Smoke, HandleLifecycle)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create(&h));
    EXPECT_NE(h, nullptr);
    int ver = 0;
    ROC_TN_EXPECT_OK(roctensornet_get_version(&ver));
    EXPECT_GT(ver, 0);
    int hver = 0;
    ROC_TN_EXPECT_OK(roctensornet_get_hip_runtime_version(&hver));
    EXPECT_GE(hver, 0);
    ROC_TN_EXPECT_OK(roctensornet_destroy(h));
}

TEST(Smoke, ErrorString)
{
    EXPECT_STREQ(roctensornet_get_error_string(ROCTENSORNET_STATUS_SUCCESS),
                 "ROCTENSORNET_STATUS_SUCCESS");
    EXPECT_STREQ(roctensornet_get_error_string(ROCTENSORNET_STATUS_NOT_SUPPORTED),
                 "ROCTENSORNET_STATUS_NOT_SUPPORTED");
}

TEST(Smoke, StreamRoundtrip)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create(&h));
    hipStream_t s = nullptr;
    ASSERT_EQ(hipStreamCreate(&s), hipSuccess);
    ROC_TN_EXPECT_OK(roctensornet_set_stream(h, s));
    hipStream_t out = nullptr;
    ROC_TN_EXPECT_OK(roctensornet_get_stream(h, &out));
    EXPECT_EQ(out, s);
    hipStreamDestroy(s);
    ROC_TN_EXPECT_OK(roctensornet_destroy(h));
}
