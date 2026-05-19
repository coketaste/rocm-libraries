#include "test_fixtures.hpp"

#include <vector>

namespace
{
/* Host-side reference: column-major matmul C = A * B */
void ref_matmul(const std::vector<double>& A, const std::vector<double>& B,
                std::vector<double>& C, int M, int K, int N)
{
    C.assign(M * N, 0.0);
    for(int j = 0; j < N; ++j)
        for(int k = 0; k < K; ++k)
            for(int i = 0; i < M; ++i)
                C[i + j * M] += A[i + k * M] * B[k + j * K];
}
}

TEST(ContractionTwoTensor, MatMul)
{
    if(roctn_skip_if_no_gpu()) return;
    constexpr int M = 3, K = 4, N = 2;
    MatMulNetwork net; net.setup_dims(M, K, N); net.build();

    std::vector<double> hA(M*K), hB(K*N), hC_ref;
    for(size_t i = 0; i < hA.size(); ++i) hA[i] = 0.1 * (i + 1);
    for(size_t i = 0; i < hB.size(); ++i) hB[i] = 0.2 * (i + 1) - 0.05;
    ref_matmul(hA, hB, hC_ref, M, K, N);

    double* dA = roctn_upload(hA);
    double* dB = roctn_upload(hB);
    double* dC = nullptr;
    ASSERT_EQ(hipMalloc(&dC, M * N * sizeof(double)), hipSuccess);
    ASSERT_EQ(hipMemset(dC, 0, M * N * sizeof(double)), hipSuccess);

    const void* inputs[2] = { dA, dB };
    ROC_TN_ASSERT_OK(roctensornet_contraction(
        net.h, net.plan, inputs, dC, net.ws, /*slice=*/0, /*stream=*/nullptr));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    auto hC = roctn_download(dC, M * N);
    for(size_t i = 0; i < hC.size(); ++i)
        EXPECT_NEAR(hC[i], hC_ref[i], 1e-9) << "i=" << i;

    hipFree(dA); hipFree(dB); hipFree(dC);
}
