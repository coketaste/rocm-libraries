/* ************************************************************************
 * mps_sample.cpp
 *
 * Sketch: build a tiny 3-site MPS as a 3-tensor contraction
 * (A1_{p1,b1} * A2_{b1,p2,b2} * A3_{b2,p3}) and contract it down to
 * the full state vector psi_{p1,p2,p3}.
 *
 * Demonstrates the 3-tensor pipeline with mixed-rank tensors.
 * ************************************************************************ */

#include "roctensornet.h"
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

int main()
{
    int n = 0;
    if(hipGetDeviceCount(&n) != hipSuccess || n <= 0) return 0;
    /* Physical dim 2, bond dim 2. */
    constexpr int d = 2, b = 2;
    roctensornet_handle h = nullptr;
    if(roctensornet_create(&h) != ROCTENSORNET_STATUS_SUCCESS) return 1;

    int32_t modes_a1[2] = {0 /*p1*/, 4 /*b1*/};
    int32_t modes_a2[3] = {4 /*b1*/, 1 /*p2*/, 5 /*b2*/};
    int32_t modes_a3[2] = {5 /*b2*/, 2 /*p3*/};
    int32_t modes_out[3] = {0, 1, 2};
    int32_t num_modes_in[3] = {2, 3, 2};
    roctensornet_index_t e_a1[2] = {d, b};
    roctensornet_index_t e_a2[3] = {b, d, b};
    roctensornet_index_t e_a3[2] = {b, d};
    roctensornet_index_t e_out[3] = {d, d, d};
    const int32_t* modes_in[3]                = {modes_a1, modes_a2, modes_a3};
    const roctensornet_index_t* extents_in[3] = {e_a1,     e_a2,     e_a3};

    roctensornet_network_descriptor nd = nullptr;
    roctensornet_create_network_descriptor(
        h, 3, num_modes_in, extents_in, nullptr, modes_in, nullptr,
        3, e_out, nullptr, modes_out, 256,
        ROCTENSORNET_R_64F, ROCTENSORNET_COMPUTE_64F, &nd);

    roctensornet_contraction_optimizer_config cfg = nullptr;
    roctensornet_contraction_optimizer_info   info = nullptr;
    roctensornet_create_contraction_optimizer_config(h, &cfg);
    roctensornet_create_contraction_optimizer_info(h, nd, &info);
    roctensornet_contraction_optimize(h, nd, cfg, 1ull<<30, info);

    roctensornet_workspace_descriptor ws = nullptr;
    roctensornet_create_workspace_descriptor(h, &ws);
    roctensornet_workspace_compute_contraction_sizes(h, nd, info, ws);

    roctensornet_contraction_plan plan = nullptr;
    roctensornet_create_contraction_plan(h, nd, info, ws, &plan);

    /* Fill A_i with simple values; the resulting psi just demonstrates the path. */
    std::vector<double> hA1(d*b, 0.1), hA2(b*d*b, 0.2), hA3(b*d, 0.3);
    double *dA1=nullptr,*dA2=nullptr,*dA3=nullptr,*dOut=nullptr;
    hipMalloc(&dA1, hA1.size()*sizeof(double));
    hipMalloc(&dA2, hA2.size()*sizeof(double));
    hipMalloc(&dA3, hA3.size()*sizeof(double));
    hipMalloc(&dOut, d*d*d*sizeof(double));
    hipMemcpy(dA1, hA1.data(), hA1.size()*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(dA2, hA2.data(), hA2.size()*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(dA3, hA3.data(), hA3.size()*sizeof(double), hipMemcpyHostToDevice);
    hipMemset(dOut, 0, d*d*d*sizeof(double));

    const void* inputs[3] = { dA1, dA2, dA3 };
    roctensornet_contraction(h, plan, inputs, dOut, ws, 0, nullptr);
    hipDeviceSynchronize();
    std::vector<double> hOut(d*d*d);
    hipMemcpy(hOut.data(), dOut, hOut.size()*sizeof(double), hipMemcpyDeviceToHost);
    std::printf("psi[0,0,0] = %f\n", hOut[0]);

    hipFree(dA1); hipFree(dA2); hipFree(dA3); hipFree(dOut);
    roctensornet_destroy_contraction_plan(plan);
    roctensornet_destroy_workspace_descriptor(ws);
    roctensornet_destroy_contraction_optimizer_info(info);
    roctensornet_destroy_contraction_optimizer_config(cfg);
    roctensornet_destroy_network_descriptor(nd);
    roctensornet_destroy(h);
    return 0;
}
