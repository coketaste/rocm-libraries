#include "test_fixtures.hpp"

#include <vector>

TEST(MemHandler, SetGetRoundtrip)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    tracking_allocator tr;
    auto handler = tr.handler();
    ROC_TN_EXPECT_OK(roctensornet_set_device_mem_handler(h, &handler));
    roctensornet_device_mem_handler_t got{};
    ROC_TN_EXPECT_OK(roctensornet_get_device_mem_handler(h, &got));
    EXPECT_EQ(got.ctx, handler.ctx);
    EXPECT_EQ(got.device_alloc, handler.device_alloc);
    EXPECT_EQ(got.device_free,  handler.device_free);
    ROC_TN_EXPECT_OK(roctensornet_destroy(h));
}

TEST(MemHandler, ContractionUsesHandlerWhenNoUserWorkspace)
{
    if(roctn_skip_if_no_gpu()) return;
    MatMulNetwork net; net.setup_dims(3, 3, 3); net.build();
    tracking_allocator tr;
    auto handler = tr.handler();
    ROC_TN_EXPECT_OK(roctensornet_set_device_mem_handler(net.h, &handler));

    std::vector<double> hA(9, 1.0), hB(9, 1.0);
    double* dA = roctn_upload(hA);
    double* dB = roctn_upload(hB);
    double* dC = nullptr;
    hipMalloc(&dC, 9*sizeof(double));
    hipMemset(dC, 0, 9*sizeof(double));

    const void* inputs[2] = { dA, dB };
    ROC_TN_ASSERT_OK(roctensornet_contraction(net.h, net.plan, inputs, dC, net.ws, 0, nullptr));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_GE(tr.alloc_calls.load(), 1);
    hipFree(dA); hipFree(dB); hipFree(dC);
}

TEST(MemHandler, UserWorkspaceBypassesHandler)
{
    if(roctn_skip_if_no_gpu()) return;
    MatMulNetwork net; net.setup_dims(3, 3, 3); net.build();
    tracking_allocator tr;
    auto handler = tr.handler();
    ROC_TN_EXPECT_OK(roctensornet_set_device_mem_handler(net.h, &handler));
    int64_t need = 0;
    ROC_TN_EXPECT_OK(roctensornet_workspace_get_memory_size(
        net.h, net.ws, ROCTENSORNET_WORKSIZE_PREF_RECOMMENDED,
        ROCTENSORNET_MEMSPACE_DEVICE, ROCTENSORNET_WORKSPACE_SCRATCH, &need));
    void* ws_buf = nullptr;
    ASSERT_EQ(hipMalloc(&ws_buf, (size_t)need + 4096), hipSuccess);
    ROC_TN_EXPECT_OK(roctensornet_workspace_set_memory(
        net.h, net.ws, ROCTENSORNET_MEMSPACE_DEVICE,
        ROCTENSORNET_WORKSPACE_SCRATCH, ws_buf, need + 4096));

    std::vector<double> hA(9, 1.0), hB(9, 1.0);
    double* dA = roctn_upload(hA);
    double* dB = roctn_upload(hB);
    double* dC = nullptr;
    hipMalloc(&dC, 9*sizeof(double));
    hipMemset(dC, 0, 9*sizeof(double));

    int before = tr.alloc_calls.load();
    const void* inputs[2] = { dA, dB };
    ROC_TN_ASSERT_OK(roctensornet_contraction(net.h, net.plan, inputs, dC, net.ws, 0, nullptr));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    int after = tr.alloc_calls.load();
    EXPECT_EQ(before, after) << "User-provided workspace should fully cover internal allocations";

    hipFree(dA); hipFree(dB); hipFree(dC); hipFree(ws_buf);
}
