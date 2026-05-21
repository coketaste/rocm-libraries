/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Apply X to |0> on a 2-d system: expect |1>. compute_action returns
 * -i H |psi>; with H = X and coefficient 1.0 we expect (0, -i).
 * ************************************************************************ */

#include "test_helpers.hpp"

TEST(ApplyPureTest, XOnZeroProducesOne)
{
    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    int64_t shape[1] = {2};

    rocdensitymat_elementary_operator x = nullptr;
    ROC_DM_OK(rocdensitymat_create_elementary_operator(
        h, 1, shape, ROCDENSITYMAT_ELEMENTARY_PAULI_X,
        ROCDENSITYMAT_C_64F, nullptr, nullptr, &x));
    rocdensitymat_operator_term t = nullptr;
    ROC_DM_OK(rocdensitymat_create_operator_term(h, 1, shape, &t));
    int32_t modes[1] = {0};
    int32_t dual[1]  = {ROCDENSITYMAT_DUALITY_KET};
    rocdensitymat_complex_double one{1,0};
    ROC_DM_OK(rocdensitymat_operator_term_append_elementary_product(
        h, t, 1, &x, modes, dual, one, nullptr));
    rocdensitymat_operator H = nullptr;
    ROC_DM_OK(rocdensitymat_create_operator(h, 1, shape, &H));
    ROC_DM_OK(rocdensitymat_operator_append_term(h, H, t, 0, one, nullptr));

    rocdensitymat_state in_s = nullptr, out_s = nullptr;
    ROC_DM_OK(rocdensitymat_create_state(
        h, ROCDENSITYMAT_STATE_PURITY_PURE, 1, shape, 1,
        ROCDENSITYMAT_C_64F, &in_s));
    ROC_DM_OK(rocdensitymat_create_state(
        h, ROCDENSITYMAT_STATE_PURITY_PURE, 1, shape, 1,
        ROCDENSITYMAT_C_64F, &out_s));
    void* in_buf  = nullptr;
    void* out_buf = nullptr;
    HIP_OK(hipMalloc(&in_buf,  2 * sizeof(hipDoubleComplex)));
    HIP_OK(hipMalloc(&out_buf, 2 * sizeof(hipDoubleComplex)));
    ROC_DM_OK(rocdensitymat_state_attach_component_buffer(
        h, in_s, 0, in_buf, 2 * sizeof(hipDoubleComplex)));
    ROC_DM_OK(rocdensitymat_state_attach_component_buffer(
        h, out_s, 0, out_buf, 2 * sizeof(hipDoubleComplex)));
    int64_t basis[1] = {0};
    ROC_DM_OK(rocdensitymat_state_initialize_basis(h, in_s, basis));

    rocdensitymat_workspace_descriptor ws = nullptr;
    ROC_DM_OK(rocdensitymat_create_workspace(h, &ws));
    ROC_DM_OK(rocdensitymat_operator_prepare_action(
        h, H, in_s, out_s, ROCDENSITYMAT_COMPUTE_64F, 0, ws));
    size_t need = 0;
    ROC_DM_OK(rocdensitymat_workspace_get_memory_size(
        h, ws, ROCDENSITYMAT_MEMSPACE_DEVICE,
        ROCDENSITYMAT_WORKSPACE_SCRATCH, &need));
    void* d_ws = nullptr;
    HIP_OK(hipMalloc(&d_ws, need));
    ROC_DM_OK(rocdensitymat_workspace_set_memory(
        h, ws, ROCDENSITYMAT_MEMSPACE_DEVICE,
        ROCDENSITYMAT_WORKSPACE_SCRATCH, d_ws, need));
    ROC_DM_OK(rocdensitymat_operator_compute_action(
        h, H, 0.0, 0, nullptr, in_s, out_s, ws));
    HIP_OK(hipDeviceSynchronize());

    auto host = download_c64(static_cast<hipDoubleComplex*>(out_buf), 2);
    EXPECT_NEAR(host[0].real(), 0.0, 1e-9);
    EXPECT_NEAR(host[0].imag(), 0.0, 1e-9);
    EXPECT_NEAR(host[1].real(), 0.0, 1e-9);
    EXPECT_NEAR(host[1].imag(), -1.0, 1e-9);

    HIP_OK(hipFree(in_buf));
    HIP_OK(hipFree(out_buf));
    HIP_OK(hipFree(d_ws));
    ROC_DM_OK(rocdensitymat_destroy_workspace(ws));
    ROC_DM_OK(rocdensitymat_destroy_state(in_s));
    ROC_DM_OK(rocdensitymat_destroy_state(out_s));
    ROC_DM_OK(rocdensitymat_destroy_operator(H));
    ROC_DM_OK(rocdensitymat_destroy_operator_term(t));
    ROC_DM_OK(rocdensitymat_destroy_elementary_operator(x));
    ROC_DM_OK(rocdensitymat_destroy(h));
}
