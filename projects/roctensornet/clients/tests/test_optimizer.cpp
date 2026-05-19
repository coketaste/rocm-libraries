#include "test_helpers.hpp"

#include <vector>

namespace
{
roctensornet_network_descriptor build_2tensor(roctensornet_handle h)
{
    int32_t num_modes_in[2] = {2, 2};
    int32_t modes_a[2] = {0, 1};
    int32_t modes_b[2] = {1, 2};
    int32_t modes_c[2] = {0, 2};
    roctensornet_index_t ea[2] = {3, 4};
    roctensornet_index_t eb[2] = {4, 5};
    roctensornet_index_t ec[2] = {3, 5};
    const int32_t* modes_in[2]                = {modes_a, modes_b};
    const roctensornet_index_t* extents_in[2] = {ea, eb};
    roctensornet_network_descriptor nd = nullptr;
    EXPECT_EQ(roctensornet_create_network_descriptor(
        h, 2, num_modes_in, extents_in, nullptr, modes_in, nullptr,
        2, ec, nullptr, modes_c, 256,
        ROCTENSORNET_R_64F, ROCTENSORNET_COMPUTE_64F, &nd),
        ROCTENSORNET_STATUS_SUCCESS);
    return nd;
}
}

TEST(Optimizer, ConfigLifecycleAndAttributes)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    roctensornet_contraction_optimizer_config cfg = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_contraction_optimizer_config(h, &cfg));
    int32_t v = 99;
    ROC_TN_EXPECT_OK(roctensornet_contraction_optimizer_config_set_attribute(
        h, cfg, ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED, &v, sizeof(v)));
    int32_t out = 0;
    ROC_TN_EXPECT_OK(roctensornet_contraction_optimizer_config_get_attribute(
        h, cfg, ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED, &out, sizeof(out)));
    EXPECT_EQ(out, 99);
    ROC_TN_EXPECT_OK(roctensornet_destroy_contraction_optimizer_config(cfg));
    ROC_TN_EXPECT_OK(roctensornet_destroy(h));
}

TEST(Optimizer, RunOptimize2Tensor)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    auto nd = build_2tensor(h);
    roctensornet_contraction_optimizer_config cfg = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_contraction_optimizer_config(h, &cfg));
    roctensornet_contraction_optimizer_info info = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_contraction_optimizer_info(h, nd, &info));
    ROC_TN_EXPECT_OK(roctensornet_contraction_optimize(h, nd, cfg, /*ws=*/1<<30, info));
    size_t flops = 0;
    ROC_TN_EXPECT_OK(roctensornet_contraction_optimizer_info_get_attribute(
        h, info, ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT, &flops, sizeof(flops)));
    EXPECT_GT(flops, 0u);
    roctensornet_destroy_contraction_optimizer_info(info);
    roctensornet_destroy_contraction_optimizer_config(cfg);
    roctensornet_destroy_network_descriptor(nd);
    roctensornet_destroy(h);
}

TEST(Optimizer, PackUnpackRoundtrip)
{
    if(roctn_skip_if_no_gpu()) return;
    roctensornet_handle h = nullptr; ROC_TN_ASSERT_OK(roctensornet_create(&h));
    auto nd = build_2tensor(h);
    roctensornet_contraction_optimizer_config cfg = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_contraction_optimizer_config(h, &cfg));
    roctensornet_contraction_optimizer_info info = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_contraction_optimizer_info(h, nd, &info));
    ROC_TN_EXPECT_OK(roctensornet_contraction_optimize(h, nd, cfg, 1<<30, info));

    size_t sz = 0;
    ROC_TN_ASSERT_OK(roctensornet_contraction_optimizer_info_get_packed_size(h, info, &sz));
    EXPECT_GT(sz, 0u);
    std::vector<unsigned char> buf(sz);
    ROC_TN_ASSERT_OK(roctensornet_contraction_optimizer_info_pack_data(h, info, buf.data(), sz));

    roctensornet_contraction_optimizer_info info2 = nullptr;
    ROC_TN_ASSERT_OK(roctensornet_create_contraction_optimizer_info_from_packed_data(
        h, nd, buf.data(), sz, &info2));

    size_t f1 = 0, f2 = 0;
    ROC_TN_EXPECT_OK(roctensornet_contraction_optimizer_info_get_attribute(
        h, info, ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT, &f1, sizeof(f1)));
    ROC_TN_EXPECT_OK(roctensornet_contraction_optimizer_info_get_attribute(
        h, info2, ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT, &f2, sizeof(f2)));
    EXPECT_EQ(f1, f2);

    roctensornet_destroy_contraction_optimizer_info(info2);
    roctensornet_destroy_contraction_optimizer_info(info);
    roctensornet_destroy_contraction_optimizer_config(cfg);
    roctensornet_destroy_network_descriptor(nd);
    roctensornet_destroy(h);
}
