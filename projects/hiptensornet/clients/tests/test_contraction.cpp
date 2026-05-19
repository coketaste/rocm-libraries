#include "test_helpers.hpp"

#include <cmath>

TEST(HipTNContraction, MatrixMatrixDouble)
{
    if(htn_skip_if_no_gpu()) return;
    hiptensornet_handle h = nullptr;
    HTN_ASSERT_OK(hiptensornet_create(&h));

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
    HTN_ASSERT_OK(hiptensornet_create_network_descriptor(
        h, 2, n_in, ext_in, str_in, mod_in, align,
        2, ec, nullptr, mc, 64,
        HIPTENSORNET_R_64F, HIPTENSORNET_COMPUTE_DEFAULT, &d));

    hiptensornet_contraction_optimizer_config cfg = nullptr;
    hiptensornet_contraction_optimizer_info   info = nullptr;
    HTN_ASSERT_OK(hiptensornet_create_contraction_optimizer_config(h, &cfg));
    HTN_ASSERT_OK(hiptensornet_create_contraction_optimizer_info(h, d, &info));
    HTN_ASSERT_OK(hiptensornet_contraction_optimize(h, d, cfg, 0, info));

    hiptensornet_workspace_descriptor ws = nullptr;
    HTN_ASSERT_OK(hiptensornet_create_workspace_descriptor(h, &ws));
    HTN_ASSERT_OK(hiptensornet_workspace_compute_contraction_sizes(h, d, info, ws));

    hiptensornet_contraction_plan plan = nullptr;
    HTN_ASSERT_OK(hiptensornet_create_contraction_plan(h, d, info, ws, &plan));

    /* a = [N, N] all 1s; b = [N, N] all 1s; c = N */
    std::vector<double> ha(N * N, 1.0);
    std::vector<double> hb(N * N, 1.0);
    std::vector<double> hc(N * N, 0.0);
    double* da = htn_upload(ha);
    double* db = htn_upload(hb);
    double* dc = htn_upload(hc);
    const void* ins[2] = {da, db};

    HTN_EXPECT_OK(hiptensornet_contraction(h, plan, ins, dc, ws, 0, 0));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    auto out = htn_download(dc, N * N);
    for(double v : out) EXPECT_NEAR(v, double(N), 1e-12);

    hipFree(da); hipFree(db); hipFree(dc);
    HTN_EXPECT_OK(hiptensornet_destroy_contraction_plan(plan));
    HTN_EXPECT_OK(hiptensornet_destroy_workspace_descriptor(ws));
    HTN_EXPECT_OK(hiptensornet_destroy_contraction_optimizer_info(info));
    HTN_EXPECT_OK(hiptensornet_destroy_contraction_optimizer_config(cfg));
    HTN_EXPECT_OK(hiptensornet_destroy_network_descriptor(d));
    HTN_EXPECT_OK(hiptensornet_destroy(h));
}
