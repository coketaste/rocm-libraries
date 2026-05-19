/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Minimal sample: 4x4 double-precision matrix-matrix contraction via
 * the portable hipTENSORNET API. Mirrors the rocTENSORNET matmul sample
 * to demonstrate API equivalence.
 * ************************************************************************ */

#include "hiptensornet.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { auto _r = (x); if(_r != HIPTENSORNET_STATUS_SUCCESS) { \
    std::fprintf(stderr, "hiptensornet error %d at %s:%d\n", _r, __FILE__, __LINE__); \
    return 1; } } while(0)

int main()
{
    int n_dev = 0;
    if(hipGetDeviceCount(&n_dev) != hipSuccess || n_dev <= 0)
    {
        std::printf("no HIP device, skipping sample\n");
        return 0;
    }

    hiptensornet_handle h = nullptr;
    CK(hiptensornet_create(&h));

    constexpr int N = 4;
    hiptensornet_index_t ea[2] = {N, N};
    hiptensornet_index_t eb[2] = {N, N};
    int32_t              ma[2] = {0, 1};
    int32_t              mb[2] = {1, 2};
    int32_t              n_in[2] = {2, 2};
    uint32_t             align[2] = {64, 64};
    const hiptensornet_index_t* ext_in[2] = {ea, eb};
    const hiptensornet_index_t* str_in[2] = {nullptr, nullptr};
    const int32_t*              mod_in[2] = {ma, mb};

    hiptensornet_index_t ec[2] = {N, N};
    int32_t              mc[2] = {0, 2};

    hiptensornet_network_descriptor d = nullptr;
    CK(hiptensornet_create_network_descriptor(
        h, 2, n_in, ext_in, str_in, mod_in, align,
        2, ec, nullptr, mc, 64,
        HIPTENSORNET_R_64F, HIPTENSORNET_COMPUTE_DEFAULT, &d));

    hiptensornet_contraction_optimizer_config cfg = nullptr;
    hiptensornet_contraction_optimizer_info   info = nullptr;
    CK(hiptensornet_create_contraction_optimizer_config(h, &cfg));
    CK(hiptensornet_create_contraction_optimizer_info(h, d, &info));
    CK(hiptensornet_contraction_optimize(h, d, cfg, 0, info));

    hiptensornet_workspace_descriptor ws = nullptr;
    CK(hiptensornet_create_workspace_descriptor(h, &ws));
    CK(hiptensornet_workspace_compute_contraction_sizes(h, d, info, ws));

    hiptensornet_contraction_plan plan = nullptr;
    CK(hiptensornet_create_contraction_plan(h, d, info, ws, &plan));

    std::vector<double> ha(N * N, 1.0);
    std::vector<double> hb(N * N, 1.0);
    double *da = nullptr, *db = nullptr, *dc = nullptr;
    hipMalloc(&da, ha.size() * sizeof(double));
    hipMalloc(&db, hb.size() * sizeof(double));
    hipMalloc(&dc, N * N * sizeof(double));
    hipMemcpy(da, ha.data(), ha.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(db, hb.data(), hb.size() * sizeof(double), hipMemcpyHostToDevice);

    const void* ins[2] = {da, db};
    CK(hiptensornet_contraction(h, plan, ins, dc, ws, 0, 0));
    hipDeviceSynchronize();

    std::vector<double> out(N * N, 0.0);
    hipMemcpy(out.data(), dc, out.size() * sizeof(double), hipMemcpyDeviceToHost);
    std::printf("hiptensornet matmul sample: C[0,0] = %.1f (expected %d)\n", out[0], N);

    hipFree(da); hipFree(db); hipFree(dc);
    hiptensornet_destroy_contraction_plan(plan);
    hiptensornet_destroy_workspace_descriptor(ws);
    hiptensornet_destroy_contraction_optimizer_info(info);
    hiptensornet_destroy_contraction_optimizer_config(cfg);
    hiptensornet_destroy_network_descriptor(d);
    hiptensornet_destroy(h);
    return 0;
}
