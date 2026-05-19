#include "test_helpers.hpp"

#include <vector>

namespace
{
/* Chain A_ij * B_jk * C_kl -> D_il */
void ref_chain(int M, int K1, int K2, int N,
               const std::vector<double>& A,
               const std::vector<double>& B,
               const std::vector<double>& C,
               std::vector<double>& D)
{
    D.assign(M * N, 0.0);
    for(int l = 0; l < N; ++l)
        for(int k = 0; k < K2; ++k)
            for(int j = 0; j < K1; ++j)
                for(int i = 0; i < M; ++i)
                    D[i + l * M] += A[i + j * M] * B[j + k * K1] * C[k + l * K2];
}
}

TEST(ContractionThreeTensor, ChainMatMul)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));

    constexpr int M = 2, K1 = 3, K2 = 2, N = 2;
    int32_t num_modes_in[3] = {2, 2, 2};
    int32_t ma[2] = {0, 1}, mb[2] = {1, 2}, mc[2] = {2, 3};
    int32_t md[2] = {0, 3};
    roctensornet_index_t ea[2] = {M, K1}, eb[2] = {K1, K2}, ec[2] = {K2, N};
    roctensornet_index_t ed[2] = {M, N};
    const int32_t* modes_in[3]                = {ma, mb, mc};
    const roctensornet_index_t* extents_in[3] = {ea, eb, ec};

    roctensornet_network_descriptor nd = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_network_descriptor(
        h, 3, num_modes_in, extents_in, nullptr, modes_in, nullptr,
        2, ed, nullptr, md, 256, ROCTENSORNET_R_64F, ROCTENSORNET_COMPUTE_64F, &nd));

    roctensornet_contraction_optimizer_config cfg = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_contraction_optimizer_config(h, &cfg));
    roctensornet_contraction_optimizer_info info = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_contraction_optimizer_info(h, nd, &info));
    ROC_TN_ASSERT_OK(roctensornet_contraction_optimize(h, nd, cfg, 1ull<<30, info));

    roctensornet_workspace_descriptor ws = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_workspace_descriptor(h, &ws));
    ROC_TN_ASSERT_OK(roctensornet_workspace_compute_contraction_sizes(h, nd, info, ws));

    roctensornet_contraction_plan plan = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_contraction_plan(h, nd, info, ws, &plan));

    std::vector<double> hA(M*K1), hB(K1*K2), hC(K2*N);
    for(size_t i = 0; i < hA.size(); ++i) hA[i] = 0.1 + 0.05 * i;
    for(size_t i = 0; i < hB.size(); ++i) hB[i] = -0.2 + 0.07 * i;
    for(size_t i = 0; i < hC.size(); ++i) hC[i] = 0.3 - 0.04 * i;
    std::vector<double> hD_ref;
    ref_chain(M, K1, K2, N, hA, hB, hC, hD_ref);

    double* dA = roctn_upload(hA);
    double* dB = roctn_upload(hB);
    double* dC = roctn_upload(hC);
    double* dD = nullptr;
    ASSERT_EQ(hipMalloc(&dD, M*N*sizeof(double)), hipSuccess);
    ASSERT_EQ(hipMemset(dD, 0, M*N*sizeof(double)), hipSuccess);
    const void* inputs[3] = { dA, dB, dC };
    ROC_TN_ASSERT_OK(roctensornet_contraction(h, plan, inputs, dD, ws, 0, nullptr));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    auto hD = roctn_download(dD, M*N);
    for(size_t i = 0; i < hD.size(); ++i)
        EXPECT_NEAR(hD[i], hD_ref[i], 1e-9) << "i=" << i;

    hipFree(dA); hipFree(dB); hipFree(dC); hipFree(dD);
    roctensornet_destroy_contraction_plan(plan);
    roctensornet_destroy_workspace_descriptor(ws);
    roctensornet_destroy_contraction_optimizer_info(info);
    roctensornet_destroy_contraction_optimizer_config(cfg);
    roctensornet_destroy_network_descriptor(nd);
    roctensornet_destroy(h);
}
