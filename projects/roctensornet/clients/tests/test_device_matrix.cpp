#include "test_fixtures.hpp"

#include <vector>

TEST(DeviceMatrix, ContractionUsesDeviceInputs)
{
    if(roctn_skip_if_no_gpu()) return;
    MatMulNetwork net; net.setup_dims(2, 2, 2); net.build();
    std::vector<double> hA(4, 1.0), hB(4, 2.0);
    double* dA = roctn_upload(hA);
    double* dB = roctn_upload(hB);
    double* dC = nullptr;
    hipMalloc(&dC, 4*sizeof(double));
    hipMemset(dC, 0, 4*sizeof(double));
    const void* inputs[2] = { dA, dB };
    ROC_TN_ASSERT_OK(roctensornet_contraction(net.h, net.plan, inputs, dC, net.ws, 0, nullptr));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    auto hC = roctn_download(dC, 4);
    for(auto v : hC) EXPECT_NEAR(v, 4.0, 1e-9);
    hipFree(dA); hipFree(dB); hipFree(dC);
}
