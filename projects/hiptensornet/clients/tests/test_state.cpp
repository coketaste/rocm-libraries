#include "test_helpers.hpp"

#include <vector>

TEST(HipTNState, ApplyGateLifecycle)
{
    if(htn_skip_if_no_gpu()) return;
    hiptensornet_handle h = nullptr;
    HTN_ASSERT_OK(hiptensornet_create(&h));

    constexpr int  n_qubits  = 3;
    hiptensornet_index_t dims[n_qubits] = {2, 2, 2};
    hiptensornet_state s = nullptr;
    HTN_ASSERT_OK(hiptensornet_create_state(h, HIPTENSORNET_STATE_PURITY_PURE,
                                            n_qubits, dims, HIPTENSORNET_C_64F, &s));

    /* A 4x4 'gate' (host doubles 0..15 for shape only). */
    std::vector<double> gate(32, 0.0);
    int32_t modes[2] = {0, 1};
    int64_t tid = -1;
    HTN_EXPECT_OK(hiptensornet_state_apply_tensor_operator(
        h, s, 2, modes, gate.data(), nullptr, 0, 0, 0, &tid));
    EXPECT_GE(tid, 0);

    HTN_EXPECT_OK(hiptensornet_destroy_state(s));
    HTN_EXPECT_OK(hiptensornet_destroy(h));
}

TEST(HipTNState, SamplerAndAccessorLifecycle)
{
    if(htn_skip_if_no_gpu()) return;
    hiptensornet_handle h = nullptr;
    HTN_ASSERT_OK(hiptensornet_create(&h));

    constexpr int n = 2;
    hiptensornet_index_t dims[n] = {2, 2};
    hiptensornet_state s = nullptr;
    HTN_ASSERT_OK(hiptensornet_create_state(h, HIPTENSORNET_STATE_PURITY_PURE,
                                            n, dims, HIPTENSORNET_C_64F, &s));
    int32_t modes[n] = {0, 1};
    hiptensornet_state_sampler smp = nullptr;
    HTN_EXPECT_OK(hiptensornet_create_sampler(h, s, n, modes, &smp));
    HTN_EXPECT_OK(hiptensornet_destroy_sampler(smp));
    hiptensornet_state_accessor acc = nullptr;
    HTN_EXPECT_OK(hiptensornet_create_accessor(h, s, n, modes, nullptr, &acc));
    HTN_EXPECT_OK(hiptensornet_destroy_accessor(acc));
    HTN_EXPECT_OK(hiptensornet_destroy_state(s));
    HTN_EXPECT_OK(hiptensornet_destroy(h));
}
