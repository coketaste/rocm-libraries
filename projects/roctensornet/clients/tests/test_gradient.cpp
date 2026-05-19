#include "test_fixtures.hpp"

#include <vector>

TEST(Gradient, ReturnsNotSupportedV010)
{
    if(roctn_skip_if_no_gpu()) return;
    MatMulNetwork net; net.setup_dims(2, 2, 2); net.build();
    std::vector<double> hA(4, 1.0), hB(4, 1.0), hC(4, 1.0);
    double* dA = roctn_upload(hA);
    double* dB = roctn_upload(hB);
    double* dC = roctn_upload(hC);
    double* dGA = nullptr; double* dGB = nullptr;
    hipMalloc(&dGA, 4*sizeof(double));
    hipMalloc(&dGB, 4*sizeof(double));
    const void* inputs[2] = { dA, dB };
    void* grads[2]        = { dGA, dGB };
    auto rc = roctensornet_compute_gradients_backward(
        net.h, net.plan, inputs, dC, grads, /*acc*/0, net.ws, nullptr);
    EXPECT_EQ(rc, ROCTENSORNET_STATUS_NOT_SUPPORTED);
    hipFree(dA); hipFree(dB); hipFree(dC); hipFree(dGA); hipFree(dGB);
}
