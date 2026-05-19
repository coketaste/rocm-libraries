/* ************************************************************************
 * tensor_contraction.cpp
 *
 * Minimal sample: contract A_ij * B_jk -> C_ik for a 3x4 * 4x5 case.
 *
 * Demonstrates the complete pipeline: handle, network descriptor,
 * optimizer, workspace, plan, contract, destroy.
 * ************************************************************************ */

#include "roctensornet.h"
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(expr) do {                                                    \
    auto _s = (expr);                                                       \
    if(_s != ROCTENSORNET_STATUS_SUCCESS) {                                 \
        std::fprintf(stderr, "%s: %s\n", #expr,                             \
                     roctensornet_get_error_string(_s));                    \
        std::exit(1);                                                       \
    }                                                                       \
} while(0)

int main()
{
    int n = 0;
    if(hipGetDeviceCount(&n) != hipSuccess || n <= 0)
    {
        std::printf("No HIP device available; skipping.\n");
        return 0;
    }

    roctensornet_handle h = nullptr;
    CHECK(roctensornet_create(&h));

    constexpr int M = 3, K = 4, N = 5;
    int32_t num_modes_in[2] = {2, 2};
    int32_t ma[2] = {0, 1}, mb[2] = {1, 2}, mc[2] = {0, 2};
    roctensornet_index_t ea[2] = {M, K}, eb[2] = {K, N}, ec[2] = {M, N};
    const int32_t* modes_in[2]                = {ma, mb};
    const roctensornet_index_t* extents_in[2] = {ea, eb};

    roctensornet_network_descriptor nd = nullptr;
    CHECK(roctensornet_create_network_descriptor(
        h, 2, num_modes_in, extents_in, nullptr, modes_in, nullptr,
        2, ec, nullptr, mc, 256,
        ROCTENSORNET_R_64F, ROCTENSORNET_COMPUTE_64F, &nd));

    roctensornet_contraction_optimizer_config cfg = nullptr;
    CHECK(roctensornet_create_contraction_optimizer_config(h, &cfg));
    roctensornet_contraction_optimizer_info info = nullptr;
    CHECK(roctensornet_create_contraction_optimizer_info(h, nd, &info));
    CHECK(roctensornet_contraction_optimize(h, nd, cfg, 1ull<<30, info));

    roctensornet_workspace_descriptor ws = nullptr;
    CHECK(roctensornet_create_workspace_descriptor(h, &ws));
    CHECK(roctensornet_workspace_compute_contraction_sizes(h, nd, info, ws));

    roctensornet_contraction_plan plan = nullptr;
    CHECK(roctensornet_create_contraction_plan(h, nd, info, ws, &plan));

    std::vector<double> hA(M*K), hB(K*N);
    for(size_t i = 0; i < hA.size(); ++i) hA[i] = 0.1 * (i + 1);
    for(size_t i = 0; i < hB.size(); ++i) hB[i] = 0.2 * (i + 1);
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    hipMalloc(&dA, hA.size() * sizeof(double));
    hipMalloc(&dB, hB.size() * sizeof(double));
    hipMalloc(&dC, M*N * sizeof(double));
    hipMemcpy(dA, hA.data(), hA.size()*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), hB.size()*sizeof(double), hipMemcpyHostToDevice);
    hipMemset(dC, 0, M*N*sizeof(double));

    const void* inputs[2] = { dA, dB };
    CHECK(roctensornet_contraction(h, plan, inputs, dC, ws, 0, nullptr));
    hipDeviceSynchronize();

    std::vector<double> hC(M*N);
    hipMemcpy(hC.data(), dC, M*N*sizeof(double), hipMemcpyDeviceToHost);
    std::printf("C[0,0] = %f\n", hC[0]);
    std::printf("C[%d,%d] = %f\n", M-1, N-1, hC[M*N - 1]);

    hipFree(dA); hipFree(dB); hipFree(dC);
    roctensornet_destroy_contraction_plan(plan);
    roctensornet_destroy_workspace_descriptor(ws);
    roctensornet_destroy_contraction_optimizer_info(info);
    roctensornet_destroy_contraction_optimizer_config(cfg);
    roctensornet_destroy_network_descriptor(nd);
    roctensornet_destroy(h);
    return 0;
}
