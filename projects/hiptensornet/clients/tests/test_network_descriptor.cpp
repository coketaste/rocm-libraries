#include "test_helpers.hpp"

TEST(HipTNNetwork, CreateAndQueryAttributes)
{
    if(htn_skip_if_no_gpu()) return;
    hiptensornet_handle h = nullptr;
    HTN_ASSERT_OK(hiptensornet_create(&h));

    /* Two 4x4 matrices contracting on the middle index. */
    hiptensornet_index_t ea[2] = {4, 4};
    hiptensornet_index_t eb[2] = {4, 4};
    int32_t              ma[2] = {0, 1};
    int32_t              mb[2] = {1, 2};
    int32_t              n_in[2] = {2, 2};
    uint32_t             align[2] = {64, 64};
    const hiptensornet_index_t* ext_in[2] = {ea, eb};
    const hiptensornet_index_t* str_in[2] = {nullptr, nullptr};
    const int32_t*              mod_in[2] = {ma, mb};

    hiptensornet_index_t ec[2] = {4, 4};
    int32_t              mc[2] = {0, 2};

    hiptensornet_network_descriptor d = nullptr;
    HTN_ASSERT_OK(hiptensornet_create_network_descriptor(
        h, 2, n_in, ext_in, str_in, mod_in, align,
        2, ec, nullptr, mc, 64,
        HIPTENSORNET_R_64F, HIPTENSORNET_COMPUTE_DEFAULT, &d));

    int32_t n_inputs = 0;
    HTN_EXPECT_OK(hiptensornet_network_get_attribute(
        h, d, HIPTENSORNET_NETWORK_INPUT_TENSORS_NUM, &n_inputs, sizeof(n_inputs)));
    EXPECT_EQ(n_inputs, 2);

    int32_t n_out_modes = 0;
    HTN_EXPECT_OK(hiptensornet_network_get_attribute(
        h, d, HIPTENSORNET_NETWORK_OUTPUT_TENSOR_NUM_MODES,
        &n_out_modes, sizeof(n_out_modes)));
    EXPECT_EQ(n_out_modes, 2);

    HTN_EXPECT_OK(hiptensornet_destroy_network_descriptor(d));
    HTN_EXPECT_OK(hiptensornet_destroy(h));
}

TEST(HipTNNetwork, TensorDescriptorDetails)
{
    if(htn_skip_if_no_gpu()) return;
    hiptensornet_handle h = nullptr;
    HTN_ASSERT_OK(hiptensornet_create(&h));

    hiptensornet_index_t ext[3] = {2, 3, 4};
    int32_t              mods[3] = {7, 8, 9};
    hiptensornet_tensor_descriptor t = nullptr;
    HTN_ASSERT_OK(hiptensornet_create_tensor_descriptor(
        h, 3, ext, nullptr, mods, HIPTENSORNET_C_64F, &t));

    int32_t              q_nm = 0;
    size_t               q_bytes = 0;
    hiptensornet_data_type q_dt;
    int32_t              q_mods[8];
    hiptensornet_index_t q_ext[8];
    hiptensornet_index_t q_str[8];
    HTN_EXPECT_OK(hiptensornet_get_tensor_details(
        h, t, &q_nm, &q_bytes, &q_dt, q_mods, q_ext, q_str));
    EXPECT_EQ(q_nm, 3);
    EXPECT_EQ(q_bytes, 2u * 3u * 4u * 16u);
    EXPECT_EQ(q_dt, HIPTENSORNET_C_64F);
    EXPECT_EQ(q_mods[0], 7);
    EXPECT_EQ(q_mods[2], 9);
    EXPECT_EQ(q_ext[1], 3);

    HTN_EXPECT_OK(hiptensornet_destroy_tensor_descriptor(t));
    HTN_EXPECT_OK(hiptensornet_destroy(h));
}
