#include "test_helpers.hpp"

TEST(HipTNSmoke, HandleLifecycle)
{
    if(htn_skip_if_no_gpu()) return;
    hiptensornet_handle h = nullptr;
    HTN_ASSERT_OK(hiptensornet_create(&h));
    EXPECT_NE(h, nullptr);
    int ver = 0;
    HTN_EXPECT_OK(hiptensornet_get_version(&ver));
    EXPECT_GT(ver, 0);
    int hver = 0;
    HTN_EXPECT_OK(hiptensornet_get_hip_runtime_version(&hver));
    EXPECT_GE(hver, 0);
    HTN_EXPECT_OK(hiptensornet_destroy(h));
}

TEST(HipTNSmoke, StreamRoundtrip)
{
    if(htn_skip_if_no_gpu()) return;
    hiptensornet_handle h = nullptr;
    HTN_ASSERT_OK(hiptensornet_create(&h));
    hipStream_t s = nullptr;
    ASSERT_EQ(hipStreamCreate(&s), hipSuccess);
    HTN_EXPECT_OK(hiptensornet_set_stream(h, s));
    hipStream_t out = nullptr;
    HTN_EXPECT_OK(hiptensornet_get_stream(h, &out));
    EXPECT_EQ(out, s);
    hipStreamDestroy(s);
    HTN_EXPECT_OK(hiptensornet_destroy(h));
}

TEST(HipTNSmoke, LoggerSettings)
{
    HTN_EXPECT_OK(hiptensornet_logger_set_callback(nullptr));
    HTN_EXPECT_OK(hiptensornet_logger_set_callback_data(nullptr, nullptr));
    HTN_EXPECT_OK(hiptensornet_logger_set_level(2));
}
