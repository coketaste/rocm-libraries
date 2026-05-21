/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Single-qubit Rabi oscillator using the hipDENSITYMAT camelCase API.
 * Mirrors the rocDENSITYMAT sample exactly; runs on AMD ROCm via the
 * rocDENSITYMAT pass-through.
 *
 * Starting from |0>, integrate H = (Omega/2) sigma_x for one pi-pulse;
 * <Z> at the end should be approximately -1.
 * ************************************************************************ */

#include <hipdensitymat.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>

#define CHECK(call)                                                              \
    do {                                                                         \
        hipdensitymatStatus_t _s = (call);                                       \
        if(_s != HIPDENSITYMAT_STATUS_SUCCESS)                                   \
        {                                                                        \
            std::fprintf(stderr, "%s -> %s\n", #call,                            \
                         hipdensitymatGetErrorString(_s));                       \
            std::exit(1);                                                        \
        }                                                                        \
    } while(0)

#define HIP_CHECK(call)                                                          \
    do {                                                                         \
        hipError_t _e = (call);                                                  \
        if(_e != hipSuccess)                                                     \
        {                                                                        \
            std::fprintf(stderr, "%s -> %s\n", #call, hipGetErrorString(_e));    \
            std::exit(2);                                                        \
        }                                                                        \
    } while(0)

int main()
{
    constexpr int64_t hilbert_dim = 2;
    constexpr double  Omega       = 1.0;
    constexpr double  dt          = 1e-3;
    constexpr int64_t n_steps     = static_cast<int64_t>(M_PI / Omega / dt);

    hipdensitymatHandle_t handle = nullptr;
    CHECK(hipdensitymatCreate(&handle));

    int64_t mode_extents[1] = {hilbert_dim};

    hipdensitymatElementaryOperator_t sx = nullptr;
    CHECK(hipdensitymatCreateElementaryOperator(
        handle, 1, mode_extents,
        HIPDENSITYMAT_ELEMENTARY_PAULI_X,
        HIPDENSITYMAT_C_64F, nullptr, nullptr, &sx));

    hipdensitymatElementaryOperator_t sz = nullptr;
    CHECK(hipdensitymatCreateElementaryOperator(
        handle, 1, mode_extents,
        HIPDENSITYMAT_ELEMENTARY_PAULI_Z,
        HIPDENSITYMAT_C_64F, nullptr, nullptr, &sz));

    hipdensitymatOperatorTerm_t hterm = nullptr;
    CHECK(hipdensitymatCreateOperatorTerm(handle, 1, mode_extents, &hterm));
    int32_t modes[1]   = {0};
    int32_t duality[1] = {HIPDENSITYMAT_DUALITY_KET};
    hipdensitymatComplexDouble_t half_omega{0.5 * Omega, 0.0};
    CHECK(hipdensitymatOperatorTermAppendElementaryProduct(
        handle, hterm, 1, &sx, modes, duality, half_omega, nullptr));

    hipdensitymatOperator_t hamiltonian = nullptr;
    CHECK(hipdensitymatCreateOperator(handle, 1, mode_extents, &hamiltonian));
    hipdensitymatComplexDouble_t one_c{1.0, 0.0};
    CHECK(hipdensitymatOperatorAppendTerm(handle, hamiltonian, hterm, 0, one_c, nullptr));

    hipdensitymatOperatorTerm_t zterm = nullptr;
    CHECK(hipdensitymatCreateOperatorTerm(handle, 1, mode_extents, &zterm));
    CHECK(hipdensitymatOperatorTermAppendElementaryProduct(
        handle, zterm, 1, &sz, modes, duality, one_c, nullptr));
    hipdensitymatOperator_t z_obs = nullptr;
    CHECK(hipdensitymatCreateOperator(handle, 1, mode_extents, &z_obs));
    CHECK(hipdensitymatOperatorAppendTerm(handle, z_obs, zterm, 0, one_c, nullptr));

    hipdensitymatState_t state = nullptr;
    CHECK(hipdensitymatCreateState(
        handle, HIPDENSITYMAT_STATE_PURITY_PURE, 1, mode_extents,
        1, HIPDENSITYMAT_C_64F, &state));

    void* d_state = nullptr;
    HIP_CHECK(hipMalloc(&d_state, hilbert_dim * sizeof(hipDoubleComplex)));
    CHECK(hipdensitymatStateAttachComponentBuffer(
        handle, state, 0, d_state, hilbert_dim * sizeof(hipDoubleComplex)));
    int64_t basis[1] = {0};
    CHECK(hipdensitymatStateInitializeBasis(handle, state, basis));

    hipdensitymatMasterEquationSolver_t solver = nullptr;
    CHECK(hipdensitymatCreateMasterEquationSolver(
        handle, hamiltonian, HIPDENSITYMAT_SOLVER_RK4, &solver));

    hipdensitymatWorkspaceDescriptor_t ws = nullptr;
    CHECK(hipdensitymatCreateWorkspace(handle, &ws));
    CHECK(hipdensitymatMasterEquationSolverPrepare(
        handle, solver, state, HIPDENSITYMAT_COMPUTE_64F, 0, ws));
    size_t ws_bytes = 0;
    CHECK(hipdensitymatWorkspaceGetMemorySize(
        handle, ws, HIPDENSITYMAT_MEMSPACE_DEVICE,
        HIPDENSITYMAT_WORKSPACE_SCRATCH, &ws_bytes));
    void* d_ws = nullptr;
    HIP_CHECK(hipMalloc(&d_ws, ws_bytes));
    CHECK(hipdensitymatWorkspaceSetMemory(
        handle, ws, HIPDENSITYMAT_MEMSPACE_DEVICE,
        HIPDENSITYMAT_WORKSPACE_SCRATCH, d_ws, ws_bytes));

    CHECK(hipdensitymatMasterEquationStepN(
        handle, solver, 0.0, dt, n_steps, 0, nullptr, state, ws));

    hipdensitymatWorkspaceDescriptor_t expws = nullptr;
    CHECK(hipdensitymatCreateWorkspace(handle, &expws));
    size_t exp_need = 3 * static_cast<size_t>(hilbert_dim) * sizeof(hipDoubleComplex);
    void* d_expws = nullptr;
    HIP_CHECK(hipMalloc(&d_expws, exp_need));
    CHECK(hipdensitymatWorkspaceSetMemory(
        handle, expws, HIPDENSITYMAT_MEMSPACE_DEVICE,
        HIPDENSITYMAT_WORKSPACE_SCRATCH, d_expws, exp_need));

    hipdensitymatComplexDouble_t z_val{0.0, 0.0};
    CHECK(hipdensitymatOperatorComputeExpectation(
        handle, z_obs, 0.0, 0, nullptr, state, expws, &z_val));
    std::printf("hipdensitymat rabi pi-pulse <Z> = (%g, %g) (expected ~ -1)\n",
                z_val.x, z_val.y);

    HIP_CHECK(hipFree(d_state));
    HIP_CHECK(hipFree(d_ws));
    HIP_CHECK(hipFree(d_expws));
    CHECK(hipdensitymatDestroyWorkspace(expws));
    CHECK(hipdensitymatDestroyWorkspace(ws));
    CHECK(hipdensitymatDestroyMasterEquationSolver(solver));
    CHECK(hipdensitymatDestroyState(state));
    CHECK(hipdensitymatDestroyOperator(z_obs));
    CHECK(hipdensitymatDestroyOperator(hamiltonian));
    CHECK(hipdensitymatDestroyOperatorTerm(zterm));
    CHECK(hipdensitymatDestroyOperatorTerm(hterm));
    CHECK(hipdensitymatDestroyElementaryOperator(sz));
    CHECK(hipdensitymatDestroyElementaryOperator(sx));
    CHECK(hipdensitymatDestroy(handle));

    return std::abs(z_val.x + 1.0) < 1e-2 ? 0 : 3;
}
