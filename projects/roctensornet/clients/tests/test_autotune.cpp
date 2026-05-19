#include "test_fixtures.hpp"

#include <vector>

TEST(Autotune, MatchesContraction)
{
    if(roctn_skip_if_no_gpu()) return;
    constexpr int M = 2, K = 2, N = 2;
    MatMulNetwork net; net.setup_dims(M, K, N); net.build();

    std::vector<double> hA(M*K, 1.0), hB(K*N, 0.5);
    double* dA = roctn_upload(hA);
    double* dB = roctn_upload(hB);
    double* dC = nullptr;
    ASSERT_EQ(hipMalloc(&dC, M*N*sizeof(double)), hipSuccess);
    ASSERT_EQ(hipMemset(dC, 0, M*N*sizeof(double)), hipSuccess);

    roctensornet_contraction_autotune_preference prefs = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_contraction_autotune_preference(net.h, &prefs));
    int32_t max_iter = 1;
    ROC_TN_EXPECT_OK(roctensornet_contraction_autotune_preference_set_attribute(
        net.h, prefs, ROCTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS, &max_iter, sizeof(max_iter)));
    int32_t got = 0;
    ROC_TN_EXPECT_OK(roctensornet_contraction_autotune_preference_get_attribute(
        net.h, prefs, ROCTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS, &got, sizeof(got)));
    EXPECT_EQ(got, 1);

    const void* inputs[2] = { dA, dB };
    ROC_TN_EXPECT_OK(roctensornet_contraction_autotune(
        net.h, net.plan, inputs, dC, net.ws, prefs, nullptr));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    auto hC = roctn_download(dC, M*N);
    for(auto v : hC) EXPECT_NEAR(v, K * 1.0 * 0.5, 1e-9);

    hipFree(dA); hipFree(dB); hipFree(dC);
    roctensornet_destroy_contraction_autotune_preference(prefs);
}
