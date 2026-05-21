/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Single-qubit Rabi oscillator under H = (Omega/2) sigma_x.
 *
 * Starting from |0>, after a time t the population in |1> is sin^2(Omega t/2).
 * We integrate the Schrodinger equation with the rocDENSITYMAT RK4 stepper
 * for t in [0, pi/Omega] (one full pi-pulse) and verify that the final
 * <Z> matches -1 to within 1e-2 with dt = 1e-3.
 * ************************************************************************ */

#include <rocdensitymat.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>

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
    do {                                                                        \
        hipError_t _e = (call);                                                 \
        if(_e != hipSuccess)                                                    \
        {                                                                       \
            std::fprintf(stderr, "%s -> %s\n", #call, hipGetErrorString(_e));   \
            std::exit(2);                                                       \
        }                                                                       \
    } while(0)

int main()
{
    constexpr int64_t hilbert_dim = 2;
    constexpr double  Omega       = 1.0;
    constexpr double  dt          = 1e-3;
    constexpr int64_t n_steps     = static_cast<int64_t>(M_PI / Omega / dt);

    rocdensitymat_handle handle = nullptr;
    CHECK(rocdensitymat_create(&handle));

    rocdensitymat_elementary_operator sx = nullptr;
    int64_t mode_extents[1] = {hilbert_dim};
    CHECK(rocdensitymat_create_elementary_operator(
        handle, 1, mode_extents,
        ROCDENSITYMAT_ELEMENTARY_PAULI_X,
        ROCDENSITYMAT_C_64F, nullptr, nullptr, &sx));

    rocdensitymat_elementary_operator sz = nullptr;
    CHECK(rocdensitymat_create_elementary_operator(
        handle, 1, mode_extents,
        ROCDENSITYMAT_ELEMENTARY_PAULI_Z,
        ROCDENSITYMAT_C_64F, nullptr, nullptr, &sz));

    rocdensitymat_operator_term hterm = nullptr;
    CHECK(rocdensitymat_create_operator_term(handle, 1, mode_extents, &hterm));
    int32_t modes[1]   = {0};
    int32_t duality[1] = {ROCDENSITYMAT_DUALITY_KET};
    rocdensitymat_complex_double half_omega{0.5 * Omega, 0.0};
    CHECK(rocdensitymat_operator_term_append_elementary_product(
        handle, hterm, 1, &sx, modes, duality, half_omega, nullptr));

    rocdensitymat_operator hamiltonian = nullptr;
    CHECK(rocdensitymat_create_operator(handle, 1, mode_extents, &hamiltonian));
    rocdensitymat_complex_double one_c{1.0, 0.0};
    CHECK(rocdensitymat_operator_append_term(
        handle, hamiltonian, hterm, /*duality_offset*/0, one_c, nullptr));

    // Z observable.
    rocdensitymat_operator_term zterm = nullptr;
    CHECK(rocdensitymat_create_operator_term(handle, 1, mode_extents, &zterm));
    CHECK(rocdensitymat_operator_term_append_elementary_product(
        handle, zterm, 1, &sz, modes, duality, one_c, nullptr));
    rocdensitymat_operator z_obs = nullptr;
    CHECK(rocdensitymat_create_operator(handle, 1, mode_extents, &z_obs));
    CHECK(rocdensitymat_operator_append_term(
        handle, z_obs, zterm, 0, one_c, nullptr));

    rocdensitymat_state state = nullptr;
    CHECK(rocdensitymat_create_state(
        handle, ROCDENSITYMAT_STATE_PURITY_PURE, 1, mode_extents,
        /*batch*/1, ROCDENSITYMAT_C_64F, &state));

    void* d_state = nullptr;
    HIP_CHECK(hipMalloc(&d_state, hilbert_dim * sizeof(hipDoubleComplex)));
    CHECK(rocdensitymat_state_attach_component_buffer(
        handle, state, 0, d_state, hilbert_dim * sizeof(hipDoubleComplex)));
    int64_t basis[1] = {0};
    CHECK(rocdensitymat_state_initialize_basis(handle, state, basis));

    rocdensitymat_master_equation_solver solver = nullptr;
    CHECK(rocdensitymat_create_master_equation_solver(
        handle, hamiltonian, ROCDENSITYMAT_SOLVER_RK4, &solver));

    rocdensitymat_workspace_descriptor ws = nullptr;
    CHECK(rocdensitymat_create_workspace(handle, &ws));
    CHECK(rocdensitymat_master_equation_solver_prepare(
        handle, solver, state, ROCDENSITYMAT_COMPUTE_64F, /*ws limit*/0, ws));
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
        handle, solver, /*t0*/0.0, dt, n_steps, /*nparams*/0, nullptr,
        state, ws));

    // Read out <Z>. Allocate a small workspace for expectation.
    rocdensitymat_workspace_descriptor expws = nullptr;
    CHECK(rocdensitymat_create_workspace(handle, &expws));
    // Generous over-allocation; compute_expectation will refresh
    // `required_device_scratch_bytes` on the first call.
    size_t exp_need = 3 * static_cast<size_t>(hilbert_dim) * sizeof(hipDoubleComplex);
    void* d_expws = nullptr;
    HIP_CHECK(hipMalloc(&d_expws, exp_need));
    CHECK(rocdensitymat_workspace_set_memory(
        handle, expws, ROCDENSITYMAT_MEMSPACE_DEVICE,
        ROCDENSITYMAT_WORKSPACE_SCRATCH, d_expws, exp_need));

    rocdensitymat_complex_double z_val{0.0, 0.0};
    CHECK(rocdensitymat_operator_compute_expectation(
        handle, z_obs, /*t*/0.0, /*nparams*/0, nullptr,
        state, expws, &z_val));
    std::printf("rabi pi-pulse <Z> = (%g, %g) (expected ≈ -1)\n", z_val.x, z_val.y);

    HIP_CHECK(hipFree(d_state));
    HIP_CHECK(hipFree(d_ws));
    HIP_CHECK(hipFree(d_expws));
    CHECK(rocdensitymat_destroy_workspace(expws));
    CHECK(rocdensitymat_destroy_workspace(ws));
    CHECK(rocdensitymat_destroy_master_equation_solver(solver));
    CHECK(rocdensitymat_destroy_state(state));
    CHECK(rocdensitymat_destroy_operator(z_obs));
    CHECK(rocdensitymat_destroy_operator(hamiltonian));
    CHECK(rocdensitymat_destroy_operator_term(zterm));
    CHECK(rocdensitymat_destroy_operator_term(hterm));
    CHECK(rocdensitymat_destroy_elementary_operator(sz));
    CHECK(rocdensitymat_destroy_elementary_operator(sx));
    CHECK(rocdensitymat_destroy(handle));

    return std::abs(z_val.x + 1.0) < 1e-2 ? 0 : 3;
}
