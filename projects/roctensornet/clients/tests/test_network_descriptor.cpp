#include "test_helpers.hpp"

namespace
{
struct TwoTensorNetwork
{
    roctensornet_handle h = nullptr;
    roctensornet_network_descriptor nd = nullptr;
    static constexpr int32_t num_in = 2;
    int32_t num_modes_in[num_in] = {2, 2};
    /* A_ij  B_jk  -> C_ik */
    int32_t modes_a[2]              = {0, 1};
    int32_t modes_b[2]              = {1, 2};
    int32_t modes_c[2]              = {0, 2};
    roctensornet_index_t extents_a[2] = {3, 4};
    roctensornet_index_t extents_b[2] = {4, 5};
    roctensornet_index_t extents_c[2] = {3, 5};
    const int32_t* modes_in[2]                = {modes_a, modes_b};
    const roctensornet_index_t* extents_in[2] = {extents_a, extents_b};

    TwoTensorNetwork() { ROC_TN_ASSERT_OK(roctensornet_create(&h)); }
    ~TwoTensorNetwork()
    {
        if(nd) roctensornet_destroy_network_descriptor(nd);
        if(h)  roctensornet_destroy(h);
    }
    void build()
    {
        ROC_TN_ASSERT_OK(roctensornet_create_network_descriptor(
            h, num_in, num_modes_in, extents_in, nullptr, modes_in, nullptr,
            /*num_modes_out*/2, extents_c, nullptr, modes_c, 256,
            ROCTENSORNET_R_64F, ROCTENSORNET_COMPUTE_64F, &nd));
    }
};
}

TEST(NetworkDescriptor, Lifecycle)
{
    if(roctn_skip_if_no_gpu()) return;
    TwoTensorNetwork t; t.build();
    EXPECT_NE(t.nd, nullptr);
}

TEST(NetworkDescriptor, OutputTensorDescriptor)
{
    if(roctn_skip_if_no_gpu()) return;
    TwoTensorNetwork t; t.build();
    roctensornet_tensor_descriptor out = nullptr;
    ROC_TN_EXPECT_OK(roctensornet_get_output_tensor_descriptor(t.h, t.nd, &out));
    int32_t nm = 0;
    ROC_TN_EXPECT_OK(roctensornet_get_tensor_details(t.h, out, &nm, nullptr, nullptr, nullptr, nullptr, nullptr));
    EXPECT_EQ(nm, 2);
    ROC_TN_EXPECT_OK(roctensornet_destroy_tensor_descriptor(out));
}

TEST(NetworkDescriptor, AttributeRoundtrip)
{
    if(roctn_skip_if_no_gpu()) return;
    TwoTensorNetwork t; t.build();
    int32_t n = 0;
    ROC_TN_EXPECT_OK(roctensornet_network_get_attribute(
        t.h, t.nd, ROCTENSORNET_NETWORK_INPUT_TENSORS_NUM, &n, sizeof(n)));
    EXPECT_EQ(n, 2);
    roctensornet_compute_type ct = ROCTENSORNET_COMPUTE_DEFAULT;
    ROC_TN_EXPECT_OK(roctensornet_network_get_attribute(
        t.h, t.nd, ROCTENSORNET_NETWORK_COMPUTE_TYPE, &ct, sizeof(ct)));
    EXPECT_EQ(ct, ROCTENSORNET_COMPUTE_64F);
}
