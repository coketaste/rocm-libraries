#include "test_helpers.hpp"

#include <vector>

TEST(TensorQR, QRReconstruction)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    constexpr int M = 4, N = 3;
    int32_t modes_in[2] = {0, 1}, modes_q[2] = {0, 2}, modes_r[2] = {2, 1};
    roctensornet_index_t ein[2] = {M, N};
    roctensornet_index_t eq[2]  = {M, N};
    roctensornet_index_t er[2]  = {N, N};

    roctensornet_tensor_descriptor tin = nullptr, tq = nullptr, tr = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_descriptor(h, 2, ein, nullptr, modes_in, ROCTENSORNET_R_64F, &tin));
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_descriptor(h, 2, eq,  nullptr, modes_q,  ROCTENSORNET_R_64F, &tq));
    ROC_TN_ASSERT_OK(roctensornet_create_tensor_descriptor(h, 2, er,  nullptr, modes_r,  ROCTENSORNET_R_64F, &tr));

    std::vector<double> hIn(M*N);
    for(int j = 0; j < N; ++j) for(int i = 0; i < M; ++i)
        hIn[i + j*M] = 0.3 + 0.5*i - 0.2*j;
    double* dIn = roctn_upload(hIn);
    double* dQ = nullptr; double* dR = nullptr;
    hipMalloc(&dQ, M*N*sizeof(double));
    hipMalloc(&dR, N*N*sizeof(double));

    ROC_TN_EXPECT_OK(roctensornet_tensor_qr(h, tin, dIn, tq, dQ, tr, dR, nullptr, nullptr));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    auto hQ = roctn_download(dQ, M*N);
    auto hR = roctn_download(dR, N*N);

    /* Orthonormality: Q^T Q = I */
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
        {
            double sum = 0;
            for(int k = 0; k < M; ++k) sum += hQ[k + i*M] * hQ[k + j*M];
            EXPECT_NEAR(sum, i == j ? 1.0 : 0.0, 1e-9);
        }
    /* R upper triangular */
    for(int j = 0; j < N; ++j)
        for(int i = j + 1; i < N; ++i)
            EXPECT_NEAR(hR[i + j*N], 0.0, 1e-9);
    /* QR = A */
    for(int j = 0; j < N; ++j) for(int i = 0; i < M; ++i)
    {
        double s = 0;
        for(int k = 0; k < N; ++k) s += hQ[i + k*M] * hR[k + j*N];
        EXPECT_NEAR(s, hIn[i + j*M], 1e-9);
    }

    hipFree(dIn); hipFree(dQ); hipFree(dR);
    roctensornet_destroy_tensor_descriptor(tin);
    roctensornet_destroy_tensor_descriptor(tq);
    roctensornet_destroy_tensor_descriptor(tr);
    roctensornet_destroy(h);
}
