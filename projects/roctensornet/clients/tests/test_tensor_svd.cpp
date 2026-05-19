#include "test_helpers.hpp"

#include <vector>

TEST(TensorSVD, FullRankReconstruction)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));

    constexpr int M = 4, N = 3, K = 3;
    /* T_{m,n}: 2-mode tensor that we split as U_{m,k} * S_{k} * V_{n,k} */
    int32_t modes_in[2] = {0, 1};
    int32_t modes_u[2]  = {0, 2}; /* row m, shared k */
    int32_t modes_v[2]  = {1, 2}; /* col n, shared k */
    roctensornet_index_t ein[2] = {M, N};
    roctensornet_index_t eu[2]  = {M, K};
    roctensornet_index_t ev[2]  = {N, K};

    roctensornet_tensor_descriptor tin = nullptr, tu = nullptr, tv = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_descriptor(
        h, 2, ein, nullptr, modes_in, ROCTENSORNET_R_64F, &tin));
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_descriptor(
        h, 2, eu, nullptr, modes_u, ROCTENSORNET_R_64F, &tu));
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_descriptor(
        h, 2, ev, nullptr, modes_v, ROCTENSORNET_R_64F, &tv));

    /* Input matrix: random-ish full-rank values */
    std::vector<double> hIn(M*N);
    for(int j = 0; j < N; ++j) for(int i = 0; i < M; ++i)
        hIn[i + j*M] = 0.5 + 0.1*i + 0.2*j;

    double* dIn = roctn_upload(hIn);
    double* dU = nullptr; double* dV = nullptr; double* dS = nullptr;
    ASSERT_EQ(hipMalloc(&dU, M*K*sizeof(double)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dV, N*K*sizeof(double)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dS, K*sizeof(double)), hipSuccess);

    roctensornet_tensor_svd_config cfg = nullptr;
    roctensornet_tensor_svd_info   info = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_svd_config(h, &cfg));
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_svd_info(h, &info));

    ROC_TN_EXPECT_OK(roctensornet_tensor_svd(
        h, tin, dIn, tu, dU, dS, tv, dV, cfg, info, nullptr, nullptr));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    auto hU = roctn_download(dU, M*K);
    auto hV = roctn_download(dV, N*K);
    auto hS = roctn_download(dS, K);

    /* Reconstruct T = U * diag(S) * V^T and compare. */
    std::vector<double> recon(M*N, 0.0);
    for(int i = 0; i < M; ++i)
        for(int j = 0; j < N; ++j)
            for(int k = 0; k < K; ++k)
                recon[i + j*M] += hU[i + k*M] * hS[k] * hV[j + k*N];
    for(int i = 0; i < M*N; ++i) EXPECT_NEAR(recon[i], hIn[i], 1e-9);

    int64_t reduced = 0;
    ROC_TN_EXPECT_OK(roctensornet_tensor_svd_info_get_attribute(
        h, info, ROCTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT, &reduced, sizeof(reduced)));
    EXPECT_EQ(reduced, K);

    hipFree(dIn); hipFree(dU); hipFree(dV); hipFree(dS);
    roctensornet_destroy_tensor_svd_info(info);
    roctensornet_destroy_tensor_svd_config(cfg);
    roctensornet_destroy_tensor_descriptor(tin);
    roctensornet_destroy_tensor_descriptor(tu);
    roctensornet_destroy_tensor_descriptor(tv);
    roctensornet_destroy(h);
}

TEST(TensorSVD, MaxExtentTruncates)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    constexpr int M = 4, N = 4, K = 4, K_TRUNC = 2;
    int32_t modes_in[2] = {0, 1}, modes_u[2] = {0, 2}, modes_v[2] = {1, 2};
    roctensornet_index_t ein[2] = {M, N};
    roctensornet_index_t eu[2]  = {M, K_TRUNC};
    roctensornet_index_t ev[2]  = {N, K_TRUNC};

    roctensornet_tensor_descriptor tin = nullptr, tu = nullptr, tv = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_descriptor(h, 2, ein, nullptr, modes_in, ROCTENSORNET_R_64F, &tin));
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_descriptor(h, 2, eu,  nullptr, modes_u,  ROCTENSORNET_R_64F, &tu));
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_descriptor(h, 2, ev,  nullptr, modes_v,  ROCTENSORNET_R_64F, &tv));

    std::vector<double> hIn(M*N);
    for(int i = 0; i < M*N; ++i) hIn[i] = 0.7 + 0.1 * (i % 5);
    double* dIn = roctn_upload(hIn);
    double* dU = nullptr; double* dV = nullptr; double* dS = nullptr;
    hipMalloc(&dU, M*K_TRUNC*sizeof(double));
    hipMalloc(&dV, N*K_TRUNC*sizeof(double));
    hipMalloc(&dS, K_TRUNC*sizeof(double));

    roctensornet_tensor_svd_config cfg = nullptr;
    roctensornet_tensor_svd_info   info = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_svd_config(h, &cfg));
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_svd_info(h, &info));
    int64_t me = K_TRUNC;
    ROC_TN_EXPECT_OK(roctensornet_tensor_svd_config_set_attribute(
        h, cfg, ROCTENSORNET_TENSOR_SVD_CONFIG_MAX_EXTENT, &me, sizeof(me)));
    ROC_TN_EXPECT_OK(roctensornet_tensor_svd(h, tin, dIn, tu, dU, dS, tv, dV, cfg, info, nullptr, nullptr));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    int64_t reduced = 0;
    ROC_TN_EXPECT_OK(roctensornet_tensor_svd_info_get_attribute(
        h, info, ROCTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT, &reduced, sizeof(reduced)));
    EXPECT_LE(reduced, (int64_t)K_TRUNC);

    hipFree(dIn); hipFree(dU); hipFree(dV); hipFree(dS);
    roctensornet_destroy_tensor_svd_info(info);
    roctensornet_destroy_tensor_svd_config(cfg);
    roctensornet_destroy_tensor_descriptor(tin);
    roctensornet_destroy_tensor_descriptor(tu);
    roctensornet_destroy_tensor_descriptor(tv);
    roctensornet_destroy(h);
}
