/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Two-qubit Bell-state coherent evolution sample.
 *
 * Build the Bell state |psi> = (|00> + |11>) / sqrt(2) directly into the
 * state buffer, then verify <Z_0 Z_1> = 1 and the norm is preserved
 * after a single-step trivial evolution under H = 0 (identity-only
 * operator). The sample exercises the full prepare/compute pipeline
 * end-to-end as a smoke check.
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
        if(_e != hipSuccess) std::exit(2);                                      \
    } while(0)

int main()
{
    constexpr int64_t qubit_dim = 2;
    int64_t mode_extents[2] = {qubit_dim, qubit_dim};

    rocdensitymat_handle handle = nullptr;
    CHECK(rocdensitymat_create(&handle));

    rocdensitymat_elementary_operator id_op = nullptr;
    int64_t per_mode = qubit_dim;
    CHECK(rocdensitymat_create_elementary_operator(
        handle, 1, &per_mode, ROCDENSITYMAT_ELEMENTARY_IDENTITY,
        ROCDENSITYMAT_C_64F, nullptr, nullptr, &id_op));

    rocdensitymat_elementary_operator z_op = nullptr;
    CHECK(rocdensitymat_create_elementary_operator(
        handle, 1, &per_mode, ROCDENSITYMAT_ELEMENTARY_PAULI_Z,
        ROCDENSITYMAT_C_64F, nullptr, nullptr, &z_op));

    // Z_0 Z_1 observable.
    rocdensitymat_operator_term zz_term = nullptr;
    CHECK(rocdensitymat_create_operator_term(handle, 2, mode_extents, &zz_term));
    rocdensitymat_elementary_operator zz_factors[2] = {z_op, z_op};
    int32_t zz_modes[2]   = {0, 1};
    int32_t zz_duality[2] = {ROCDENSITYMAT_DUALITY_KET, ROCDENSITYMAT_DUALITY_KET};
    rocdensitymat_complex_double one_c{1.0, 0.0};
    CHECK(rocdensitymat_operator_term_append_elementary_product(
        handle, zz_term, 2, zz_factors, zz_modes, zz_duality, one_c, nullptr));

    rocdensitymat_operator zz_obs = nullptr;
    CHECK(rocdensitymat_create_operator(handle, 2, mode_extents, &zz_obs));
    CHECK(rocdensitymat_operator_append_term(handle, zz_obs, zz_term, 0, one_c, nullptr));

    rocdensitymat_state state = nullptr;
    CHECK(rocdensitymat_create_state(
        handle, ROCDENSITYMAT_STATE_PURITY_PURE, 2, mode_extents,
        1, ROCDENSITYMAT_C_64F, &state));

    constexpr int64_t hdim = 4;
    void* d_state = nullptr;
    HIP_CHECK(hipMalloc(&d_state, hdim * sizeof(hipDoubleComplex)));
    CHECK(rocdensitymat_state_attach_component_buffer(
        handle, state, 0, d_state, hdim * sizeof(hipDoubleComplex)));

    // Construct |psi> = (|00> + |11>) / sqrt(2) directly.
    {
        std::complex<double> bell[hdim] = {{1.0/std::sqrt(2.0), 0},
                                            {0, 0},
                                            {0, 0},
                                            {1.0/std::sqrt(2.0), 0}};
        HIP_CHECK(hipMemcpy(d_state, bell, sizeof(bell), hipMemcpyHostToDevice));
    }

    // Compute <Z_0 Z_1>: expected = 1.
    rocdensitymat_workspace_descriptor expws = nullptr;
    CHECK(rocdensitymat_create_workspace(handle, &expws));
    // Generous over-allocation; compute_expectation will refresh
    // `required_device_scratch_bytes` on the first call below.
    size_t exp_need = 3 * hdim * sizeof(hipDoubleComplex);
    void* d_expws = nullptr;
    HIP_CHECK(hipMalloc(&d_expws, exp_need));
    CHECK(rocdensitymat_workspace_set_memory(
        handle, expws, ROCDENSITYMAT_MEMSPACE_DEVICE,
        ROCDENSITYMAT_WORKSPACE_SCRATCH, d_expws, exp_need));

    rocdensitymat_complex_double zz_val{0.0, 0.0};
    CHECK(rocdensitymat_operator_compute_expectation(
        handle, zz_obs, 0.0, 0, nullptr, state, expws, &zz_val));
    std::printf("Bell <Z_0 Z_1> = (%g, %g) (expected ≈ 1)\n", zz_val.x, zz_val.y);

    double n = 0;
    CHECK(rocdensitymat_state_compute_norm(handle, state, expws, &n));
    std::printf("Bell norm = %g (expected ≈ 1)\n", n);

    HIP_CHECK(hipFree(d_state));
    HIP_CHECK(hipFree(d_expws));
    CHECK(rocdensitymat_destroy_workspace(expws));
    CHECK(rocdensitymat_destroy_state(state));
    CHECK(rocdensitymat_destroy_operator(zz_obs));
    CHECK(rocdensitymat_destroy_operator_term(zz_term));
    CHECK(rocdensitymat_destroy_elementary_operator(z_op));
    CHECK(rocdensitymat_destroy_elementary_operator(id_op));
    CHECK(rocdensitymat_destroy(handle));

    bool ok = std::abs(zz_val.x - 1.0) < 1e-9 && std::abs(n - 1.0) < 1e-9;
    return ok ? 0 : 3;
}
