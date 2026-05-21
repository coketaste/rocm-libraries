/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Single-qubit Rabi oscillator under H = (Omega/2) X. Integrate with RK4
 * for n_steps and compare <Z(t)> to the analytic cos(Omega t) within a
 * tight tolerance.
 * ************************************************************************ */

#include "test_helpers.hpp"

TEST(RK4Test, RabiHalfCycleFlipsZ)
{
    constexpr double  Omega = 1.0;
    constexpr double  dt    = 1e-3;
    constexpr int64_t N     = static_cast<int64_t>(M_PI / Omega / dt);

    rocdensitymat_handle h = nullptr;
    ROC_DM_OK(rocdensitymat_create(&h));
    int64_t shape[1] = {2};

    rocdensitymat_elementary_operator x = nullptr, z = nullptr;
    ROC_DM_OK(rocdensitymat_create_elementary_operator(
        h, 1, shape, ROCDENSITYMAT_ELEMENTARY_PAULI_X,
        ROCDENSITYMAT_C_64F, nullptr, nullptr, &x));
    ROC_DM_OK(rocdensitymat_create_elementary_operator(
        h, 1, shape, ROCDENSITYMAT_ELEMENTARY_PAULI_Z,
        ROCDENSITYMAT_C_64F, nullptr, nullptr, &z));

    int32_t modes[1] = {0};
    int32_t dual[1]  = {ROCDENSITYMAT_DUALITY_KET};
    rocdensitymat_complex_double one{1,0};
    rocdensitymat_complex_double half_omega{0.5 * Omega, 0.0};

    rocdensitymat_operator_term hterm = nullptr;
    ROC_DM_OK(rocdensitymat_create_operator_term(h, 1, shape, &hterm));
    ROC_DM_OK(rocdensitymat_operator_term_append_elementary_product(
        h, hterm, 1, &x, modes, dual, half_omega, nullptr));
    rocdensitymat_operator H = nullptr;
    ROC_DM_OK(rocdensitymat_create_operator(h, 1, shape, &H));
    ROC_DM_OK(rocdensitymat_operator_append_term(h, H, hterm, 0, one, nullptr));

    rocdensitymat_operator_term zterm = nullptr;
    ROC_DM_OK(rocdensitymat_create_operator_term(h, 1, shape, &zterm));
    ROC_DM_OK(rocdensitymat_operator_term_append_elementary_product(
        h, zterm, 1, &z, modes, dual, one, nullptr));
    rocdensitymat_operator Zobs = nullptr;
    ROC_DM_OK(rocdensitymat_create_operator(h, 1, shape, &Zobs));
    ROC_DM_OK(rocdensitymat_operator_append_term(h, Zobs, zterm, 0, one, nullptr));

    rocdensitymat_state s = nullptr;
    ROC_DM_OK(rocdensitymat_create_state(
        h, ROCDENSITYMAT_STATE_PURITY_PURE, 1, shape, 1,
        ROCDENSITYMAT_C_64F, &s));
    void* d = nullptr;
    HIP_OK(hipMalloc(&d, 2 * sizeof(hipDoubleComplex)));
    ROC_DM_OK(rocdensitymat_state_attach_component_buffer(
        h, s, 0, d, 2 * sizeof(hipDoubleComplex)));
    int64_t basis[1] = {0};
    ROC_DM_OK(rocdensitymat_state_initialize_basis(h, s, basis));

    rocdensitymat_master_equation_solver solver = nullptr;
    ROC_DM_OK(rocdensitymat_create_master_equation_solver(
        h, H, ROCDENSITYMAT_SOLVER_RK4, &solver));
    rocdensitymat_workspace_descriptor ws = nullptr;
    ROC_DM_OK(rocdensitymat_create_workspace(h, &ws));
    ROC_DM_OK(rocdensitymat_master_equation_solver_prepare(
        h, solver, s, ROCDENSITYMAT_COMPUTE_64F, 0, ws));
    size_t need = 0;
    ROC_DM_OK(rocdensitymat_workspace_get_memory_size(
        h, ws, ROCDENSITYMAT_MEMSPACE_DEVICE,
        ROCDENSITYMAT_WORKSPACE_SCRATCH, &need));
    void* d_ws = nullptr;
    HIP_OK(hipMalloc(&d_ws, need));
    ROC_DM_OK(rocdensitymat_workspace_set_memory(
        h, ws, ROCDENSITYMAT_MEMSPACE_DEVICE,
        ROCDENSITYMAT_WORKSPACE_SCRATCH, d_ws, need));
    ROC_DM_OK(rocdensitymat_master_equation_step_n(
        h, solver, 0.0, dt, N, 0, nullptr, s, ws));

    rocdensitymat_workspace_descriptor expws = nullptr;
    ROC_DM_OK(rocdensitymat_create_workspace(h, &expws));
    void* d_expws = nullptr;
    HIP_OK(hipMalloc(&d_expws, 6 * sizeof(hipDoubleComplex)));
    ROC_DM_OK(rocdensitymat_workspace_set_memory(
        h, expws, ROCDENSITYMAT_MEMSPACE_DEVICE,
        ROCDENSITYMAT_WORKSPACE_SCRATCH, d_expws, 6 * sizeof(hipDoubleComplex)));
    rocdensitymat_complex_double zv{0,0};
    ROC_DM_OK(rocdensitymat_operator_compute_expectation(
        h, Zobs, 0.0, 0, nullptr, s, expws, &zv));

    EXPECT_NEAR(zv.x, std::cos(Omega * N * dt), 1e-2);
    EXPECT_NEAR(zv.y, 0.0, 1e-2);

    HIP_OK(hipFree(d));
    HIP_OK(hipFree(d_ws));
    HIP_OK(hipFree(d_expws));
    ROC_DM_OK(rocdensitymat_destroy_workspace(expws));
    ROC_DM_OK(rocdensitymat_destroy_workspace(ws));
    ROC_DM_OK(rocdensitymat_destroy_master_equation_solver(solver));
    ROC_DM_OK(rocdensitymat_destroy_state(s));
    ROC_DM_OK(rocdensitymat_destroy_operator(Zobs));
    ROC_DM_OK(rocdensitymat_destroy_operator(H));
    ROC_DM_OK(rocdensitymat_destroy_operator_term(zterm));
    ROC_DM_OK(rocdensitymat_destroy_operator_term(hterm));
    ROC_DM_OK(rocdensitymat_destroy_elementary_operator(z));
    ROC_DM_OK(rocdensitymat_destroy_elementary_operator(x));
    ROC_DM_OK(rocdensitymat_destroy(h));
}
