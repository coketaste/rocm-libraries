/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Driven truncated quantum harmonic oscillator (4-level Fock space) under
 *   H(t) = omega n + 2 lambda cos(omega_d t) (a + a^\dagger),
 * where n = a^\dagger a is the number operator and the drive amplitude
 * is supplied through a host-side time-dependent callback.
 *
 * The sample demonstrates:
 *   - dense elementary operator construction (n, a + a^\dagger),
 *   - host-side time-dependent scalar callback wired into the OperatorTerm,
 *   - RK4 integration over O(N) steps,
 *   - <n> readout via OperatorComputeExpectation.
 *
 * v0.1.0 success criterion is "doesn't blow up" (norm preserved within
 * 1e-3) plus a non-zero <n> after the drive period.
 * ************************************************************************ */

#include <rocdensitymat.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call)                                                             \
    do {                                                                        \
        rocdensitymat_status _s = (call);                                       \
        if(_s != ROCDENSITYMAT_STATUS_SUCCESS)                                  \
        {                                                                       \
            std::fprintf(stderr, "%s -> %s\n", #call,                           \
                         rocdensitymat_get_error_string(_s));                   \
            std::exit(1);                                                       \
        }                                                                       \
    } while(0)

#define HIP_CHECK(call)                                                         \
    do { if((call) != hipSuccess) std::exit(2); } while(0)

namespace
{
    // Drive callback: 2 lambda cos(omega_d t).
    rocdensitymat_complex_double drive_callback(double t,
                                                int32_t /*np*/,
                                                const double* /*p*/)
    {
        // omega_d = 1.0, lambda = 0.05 fixed.
        rocdensitymat_complex_double v;
        v.x = 0.1 * std::cos(t);
        v.y = 0.0;
        return v;
    }
}

int main()
{
    constexpr int64_t fock_dim = 4;
    constexpr double  omega    = 1.0;
    constexpr double  dt       = 5e-3;
    constexpr int64_t n_steps  = 200;

    rocdensitymat_handle handle = nullptr;
    CHECK(rocdensitymat_create(&handle));

    // Build n = diag(0,1,2,3) as a diagonal elementary operator.
    std::vector<std::complex<double>> n_diag = {{0,0},{1,0},{2,0},{3,0}};
    rocdensitymat_elementary_operator n_op = nullptr;
    int64_t per_mode = fock_dim;
    CHECK(rocdensitymat_create_elementary_operator(
        handle, 1, &per_mode, ROCDENSITYMAT_ELEMENTARY_DIAGONAL,
        ROCDENSITYMAT_C_64F, n_diag.data(), nullptr, &n_op));

    // Build (a + a^\dagger) as a dense (4x4) tridiagonal real-symmetric
    // matrix with sqrt(k) on the (k, k-1) and (k-1, k) entries.
    std::vector<std::complex<double>> apd(fock_dim * fock_dim, {0,0});
    for(int64_t k = 1; k < fock_dim; ++k)
    {
        double sk = std::sqrt(static_cast<double>(k));
        apd[k * fock_dim + (k - 1)] = std::complex<double>(sk, 0);
        apd[(k - 1) * fock_dim + k] = std::complex<double>(sk, 0);
    }
    rocdensitymat_elementary_operator apd_op = nullptr;
    CHECK(rocdensitymat_create_elementary_operator(
        handle, 1, &per_mode, ROCDENSITYMAT_ELEMENTARY_DENSE,
        ROCDENSITYMAT_C_64F, apd.data(), nullptr, &apd_op));

    int64_t mode_extents[1] = {fock_dim};
    int32_t modes[1]   = {0};
    int32_t duality[1] = {ROCDENSITYMAT_DUALITY_KET};
    rocdensitymat_complex_double one_c{1.0, 0.0};

    // omega n term.
    rocdensitymat_operator_term n_term = nullptr;
    CHECK(rocdensitymat_create_operator_term(handle, 1, mode_extents, &n_term));
    rocdensitymat_complex_double w_c{omega, 0.0};
    CHECK(rocdensitymat_operator_term_append_elementary_product(
        handle, n_term, 1, &n_op, modes, duality, w_c, nullptr));

    // 2 lambda cos(omega_d t) (a + a^\dagger) term — drive amplitude
    // delivered through the time-dependent scalar callback.
    rocdensitymat_operator_term drive_term = nullptr;
    CHECK(rocdensitymat_create_operator_term(handle, 1, mode_extents, &drive_term));
    CHECK(rocdensitymat_operator_term_append_elementary_product(
        handle, drive_term, 1, &apd_op, modes, duality, one_c, drive_callback));

    rocdensitymat_operator H = nullptr;
    CHECK(rocdensitymat_create_operator(handle, 1, mode_extents, &H));
    CHECK(rocdensitymat_operator_append_term(handle, H, n_term, 0, one_c, nullptr));
    CHECK(rocdensitymat_operator_append_term(handle, H, drive_term, 0, one_c, nullptr));

    rocdensitymat_state state = nullptr;
    CHECK(rocdensitymat_create_state(
        handle, ROCDENSITYMAT_STATE_PURITY_PURE, 1, mode_extents,
        1, ROCDENSITYMAT_C_64F, &state));
    void* d_state = nullptr;
    HIP_CHECK(hipMalloc(&d_state, fock_dim * sizeof(hipDoubleComplex)));
    CHECK(rocdensitymat_state_attach_component_buffer(
        handle, state, 0, d_state, fock_dim * sizeof(hipDoubleComplex)));
    int64_t basis[1] = {0};
    CHECK(rocdensitymat_state_initialize_basis(handle, state, basis));

    rocdensitymat_master_equation_solver solver = nullptr;
    CHECK(rocdensitymat_create_master_equation_solver(
        handle, H, ROCDENSITYMAT_SOLVER_RK4, &solver));

    rocdensitymat_workspace_descriptor ws = nullptr;
    CHECK(rocdensitymat_create_workspace(handle, &ws));
    CHECK(rocdensitymat_master_equation_solver_prepare(
        handle, solver, state, ROCDENSITYMAT_COMPUTE_64F, 0, ws));
    size_t ws_bytes = 0;
    CHECK(rocdensitymat_workspace_get_memory_size(
        handle, ws, ROCDENSITYMAT_MEMSPACE_DEVICE,
        ROCDENSITYMAT_WORKSPACE_SCRATCH, &ws_bytes));
    void* d_ws = nullptr;
    HIP_CHECK(hipMalloc(&d_ws, ws_bytes));
    CHECK(rocdensitymat_workspace_set_memory(
        handle, ws, ROCDENSITYMAT_MEMSPACE_DEVICE,
        ROCDENSITYMAT_WORKSPACE_SCRATCH, d_ws, ws_bytes));

    CHECK(rocdensitymat_master_equation_step_n(
        handle, solver, 0.0, dt, n_steps, 0, nullptr, state, ws));

    // Build the <n> observable from the same diagonal n_op.
    rocdensitymat_operator n_obs = nullptr;
    CHECK(rocdensitymat_create_operator(handle, 1, mode_extents, &n_obs));
    CHECK(rocdensitymat_operator_append_term(handle, n_obs, n_term, 0, one_c, nullptr));

    rocdensitymat_workspace_descriptor expws = nullptr;
    CHECK(rocdensitymat_create_workspace(handle, &expws));
    size_t exp_need = 3 * fock_dim * sizeof(hipDoubleComplex);
    void* d_expws = nullptr;
    HIP_CHECK(hipMalloc(&d_expws, exp_need));
    CHECK(rocdensitymat_workspace_set_memory(
        handle, expws, ROCDENSITYMAT_MEMSPACE_DEVICE,
        ROCDENSITYMAT_WORKSPACE_SCRATCH, d_expws, exp_need));
    rocdensitymat_complex_double n_val{0,0};
    CHECK(rocdensitymat_operator_compute_expectation(
        handle, n_obs, n_steps * dt, 0, nullptr, state, expws, &n_val));

    double norm = 0;
    CHECK(rocdensitymat_state_compute_norm(handle, state, expws, &norm));
    std::printf("driven oscillator <n> = (%g, %g), norm = %g (expected ≈ 1)\n",
                n_val.x, n_val.y, norm);

    HIP_CHECK(hipFree(d_state));
    HIP_CHECK(hipFree(d_ws));
    HIP_CHECK(hipFree(d_expws));
    CHECK(rocdensitymat_destroy_workspace(expws));
    CHECK(rocdensitymat_destroy_workspace(ws));
    CHECK(rocdensitymat_destroy_master_equation_solver(solver));
    CHECK(rocdensitymat_destroy_state(state));
    CHECK(rocdensitymat_destroy_operator(n_obs));
    CHECK(rocdensitymat_destroy_operator(H));
    CHECK(rocdensitymat_destroy_operator_term(drive_term));
    CHECK(rocdensitymat_destroy_operator_term(n_term));
    CHECK(rocdensitymat_destroy_elementary_operator(apd_op));
    CHECK(rocdensitymat_destroy_elementary_operator(n_op));
    CHECK(rocdensitymat_destroy(handle));

    return std::abs(norm - 1.0) < 1e-3 ? 0 : 3;
}
