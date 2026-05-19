/* ************************************************************************
 * cudaq_pattern_demo.cpp
 *
 * Mirrors a typical CUDA-Q tensornet integration pattern:
 *   1. Build a 2-tensor contraction (a single layer of a circuit).
 *   2. Query workspace size.
 *   3. Pre-allocate the workspace.
 *   4. Run the contraction.
 *
 * The output is a small print so the sample doubles as a smoke test.
 * ************************************************************************ */

#include "roctensornet.h"
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

int main()
{
    int n = 0;
    if(hipGetDeviceCount(&n) != hipSuccess || n <= 0) return 0;
    roctensornet_handle h = nullptr;
    roctensornet_create(&h);

    constexpr int d = 2;
    int32_t num_modes_in[2] = {2, 2};
    int32_t ma[2] = {0, 1}, mb[2] = {1, 2}, mc[2] = {0, 2};
    roctensornet_index_t ea[2] = {d, d}, eb[2] = {d, d}, ec[2] = {d, d};
    const int32_t* modes_in[2] = {ma, mb};
    const roctensornet_index_t* extents_in[2] = {ea, eb};

    roctensornet_network_descriptor nd = nullptr;
    roctensornet_create_network_descriptor(
        h, 2, num_modes_in, extents_in, nullptr, modes_in, nullptr,
        2, ec, nullptr, mc, 256,
        ROCTENSORNET_C_64F, ROCTENSORNET_COMPUTE_64F, &nd);
    roctensornet_contraction_optimizer_config cfg = nullptr;
    roctensornet_create_contraction_optimizer_config(h, &cfg);
    roctensornet_contraction_optimizer_info info = nullptr;
    roctensornet_create_contraction_optimizer_info(h, nd, &info);
    roctensornet_contraction_optimize(h, nd, cfg, 1ull<<30, info);

    roctensornet_workspace_descriptor ws = nullptr;
    roctensornet_create_workspace_descriptor(h, &ws);
    roctensornet_workspace_compute_contraction_sizes(h, nd, info, ws);
    int64_t need = 0;
    roctensornet_workspace_get_memory_size(
        h, ws, ROCTENSORNET_WORKSIZE_PREF_RECOMMENDED,
        ROCTENSORNET_MEMSPACE_DEVICE, ROCTENSORNET_WORKSPACE_SCRATCH, &need);
    std::printf("Required scratch: %lld bytes\n", (long long)need);

    void* ws_buf = nullptr;
    hipMalloc(&ws_buf, (size_t)need);
    roctensornet_workspace_set_memory(
        h, ws, ROCTENSORNET_MEMSPACE_DEVICE,
        ROCTENSORNET_WORKSPACE_SCRATCH, ws_buf, need);

    roctensornet_contraction_plan plan = nullptr;
    roctensornet_create_contraction_plan(h, nd, info, ws, &plan);

    std::vector<double> hI(2 * d * d, 0.0);
    hI[0] = 1.0; hI[6] = 1.0; /* identity-like layout */
    double* dA = nullptr; double* dB = nullptr; double* dC = nullptr;
    hipMalloc(&dA, hI.size()*sizeof(double));
    hipMalloc(&dB, hI.size()*sizeof(double));
    hipMalloc(&dC, hI.size()*sizeof(double));
    hipMemcpy(dA, hI.data(), hI.size()*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(dB, hI.data(), hI.size()*sizeof(double), hipMemcpyHostToDevice);
    hipMemset(dC, 0, hI.size()*sizeof(double));

    const void* inputs[2] = { dA, dB };
    roctensornet_contraction(h, plan, inputs, dC, ws, 0, nullptr);
    hipDeviceSynchronize();

    std::printf("Pattern run OK\n");

    hipFree(dA); hipFree(dB); hipFree(dC); hipFree(ws_buf);
    roctensornet_destroy_contraction_plan(plan);
    roctensornet_destroy_workspace_descriptor(ws);
    roctensornet_destroy_contraction_optimizer_info(info);
    roctensornet_destroy_contraction_optimizer_config(cfg);
    roctensornet_destroy_network_descriptor(nd);
    roctensornet_destroy(h);
    return 0;
}
