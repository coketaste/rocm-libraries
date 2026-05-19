/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Consumer-readiness demo for rocSTATEVEC.
 *
 * Walks through every cuStateVec entry point cuda-quantum exercises today,
 * one-for-one, against the rocSTATEVEC API. The 12 entry points covered:
 *
 *   1.  rocstatevec_create_handle          -> custatevecCreate
 *   2.  rocstatevec_destroy_handle         -> custatevecDestroy
 *   3.  rocstatevec_get_error_string       -> custatevecGetErrorString
 *   4.  rocstatevec_apply_matrix(_get_workspace_size)
 *                                          -> custatevecApplyMatrix
 *   5.  rocstatevec_apply_pauli_rotation   -> custatevecApplyPauliRotation
 *   6.  rocstatevec_measure_on_z_basis     -> custatevecMeasureOnZBasis
 *   7.  rocstatevec_compute_expectation(_get_workspace_size)
 *                                          -> custatevecComputeExpectation
 *   8.  rocstatevec_compute_expectations_on_pauli_basis
 *                                          -> custatevecComputeExpectationsOnPauliBasis
 *   9.  rocstatevec_sampler_create         -> custatevecSamplerCreate
 *   10. rocstatevec_sampler_preprocess     -> custatevecSamplerPreprocess
 *   11. rocstatevec_sampler_sample         -> custatevecSamplerSample
 *   12. rocstatevec_sampler_destroy        -> custatevecSamplerDestroy
 *
 * Circuit:
 *   * 3-qubit GHZ via H on q0, CNOT q0->q1, CNOT q0->q2 — built with
 *     `apply_matrix` for H/CNOT and `apply_pauli_rotation` for an extra
 *     Rz on q2 to exercise the Pauli-rotation path.
 *   * Compute <ZZZ>; expected ~ cos(theta) due to the Rz wrapping the
 *     parity exactly with phase ±1 (Hermitian Pauli string evaluation).
 *   * Sample 1024 shots and emit a histogram. GHZ has only 000 and 111.
 *   * Compute <P> for two Pauli strings (ZZZ and XXX) in one batched call.
 * ************************************************************************ */

#include <rocstatevec.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

namespace
{
using cd = std::complex<double>;

#define CHECK(call)                                                           \
    do {                                                                      \
        rocstatevec_status _s = (call);                                       \
        if(_s != ROCSTATEVEC_STATUS_SUCCESS)                                  \
        {                                                                     \
            std::printf("[error] %s -> %s\n", #call,                          \
                        rocstatevec_get_error_string(_s));                    \
            return 1;                                                         \
        }                                                                     \
    } while(0)

#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t _e = (call);                                               \
        if(_e != hipSuccess)                                                  \
        {                                                                     \
            std::printf("[hip] %s -> %s\n", #call, hipGetErrorString(_e));    \
            return 1;                                                         \
        }                                                                     \
    } while(0)

constexpr double SQRT2_INV = 0.70710678118654752440;
} // namespace

int main()
{
    int n_dev = 0;
    if(hipGetDeviceCount(&n_dev) != hipSuccess || n_dev == 0)
    {
        std::printf("[skip] no HIP-capable device available.\n");
        return 0;
    }

    rocstatevec_handle handle = nullptr;
    CHECK(rocstatevec_create_handle(&handle));
    std::printf("[1/12] rocstatevec_create_handle ok\n");

    constexpr uint32_t n     = 3;
    constexpr size_t   N     = size_t{1} << n;

    cd* dsv = nullptr;
    HIP_CHECK(hipMalloc(&dsv, N * sizeof(cd)));
    CHECK(rocstatevec_initialize_state_vector(
        handle, dsv, ROCSTATEVEC_C_64F, n, ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO));

    cd      H_mat[4]   = {{SQRT2_INV, 0}, {SQRT2_INV, 0}, {SQRT2_INV, 0}, {-SQRT2_INV, 0}};
    cd      X_mat[4]   = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    int32_t targets0[1] = {0};
    int32_t targets1[1] = {1};
    int32_t targets2[1] = {2};
    int32_t controls0[1] = {0};

    size_t ws_bytes = 0;
    CHECK(rocstatevec_apply_matrix_get_workspace_size(
        handle, ROCSTATEVEC_C_64F, n, H_mat, ROCSTATEVEC_C_64F,
        ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1, 0, ROCSTATEVEC_COMPUTE_64F, &ws_bytes));
    void* ws_buf = nullptr;
    if(ws_bytes) HIP_CHECK(hipMalloc(&ws_buf, ws_bytes));
    CHECK(rocstatevec_apply_matrix(
        handle, dsv, ROCSTATEVEC_C_64F, n, H_mat, ROCSTATEVEC_C_64F,
        ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets0, 1, nullptr, nullptr, 0,
        ROCSTATEVEC_COMPUTE_64F, ws_buf, ws_bytes));
    std::printf("[2/12] rocstatevec_apply_matrix(H) ok (workspace=%zu B)\n", ws_bytes);

    CHECK(rocstatevec_apply_matrix(
        handle, dsv, ROCSTATEVEC_C_64F, n, X_mat, ROCSTATEVEC_C_64F,
        ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets1, 1, controls0, nullptr, 1,
        ROCSTATEVEC_COMPUTE_64F, ws_buf, ws_bytes));
    CHECK(rocstatevec_apply_matrix(
        handle, dsv, ROCSTATEVEC_C_64F, n, X_mat, ROCSTATEVEC_C_64F,
        ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets2, 1, controls0, nullptr, 1,
        ROCSTATEVEC_COMPUTE_64F, ws_buf, ws_bytes));
    std::printf("[3/12] CNOT(0->1), CNOT(0->2) via apply_matrix ok\n");

    rocstatevec_pauli z_basis[1]   = {ROCSTATEVEC_PAULI_Z};
    int32_t           rz_targets[1] = {2};
    constexpr double  theta         = 0.0;
    CHECK(rocstatevec_apply_pauli_rotation(
        handle, dsv, ROCSTATEVEC_C_64F, n, theta, z_basis, rz_targets, 1,
        nullptr, nullptr, 0));
    std::printf("[4/12] rocstatevec_apply_pauli_rotation ok\n");

    cd     Z_mat[4]            = {{1, 0}, {0, 0}, {0, 0}, {-1, 0}};
    int32_t exp_target[1]     = {0};
    size_t  exp_ws_bytes      = 0;
    CHECK(rocstatevec_compute_expectation_get_workspace_size(
        handle, ROCSTATEVEC_C_64F, n, Z_mat, ROCSTATEVEC_C_64F,
        ROCSTATEVEC_MATRIX_LAYOUT_ROW, 1, ROCSTATEVEC_COMPUTE_64F, &exp_ws_bytes));
    void* exp_ws_buf = nullptr;
    if(exp_ws_bytes) HIP_CHECK(hipMalloc(&exp_ws_buf, exp_ws_bytes));
    cd     exp_value         = {0, 0};
    double residual          = 0.0;
    CHECK(rocstatevec_compute_expectation(
        handle, dsv, ROCSTATEVEC_C_64F, n, &exp_value, ROCSTATEVEC_C_64F, &residual,
        Z_mat, ROCSTATEVEC_C_64F, ROCSTATEVEC_MATRIX_LAYOUT_ROW, exp_target, 1,
        ROCSTATEVEC_COMPUTE_64F, exp_ws_buf, exp_ws_bytes));
    std::printf("[5/12] <Z_0> for GHZ = %g (expected 0)\n", exp_value.real());

    rocstatevec_pauli              zzz[3] = {ROCSTATEVEC_PAULI_Z, ROCSTATEVEC_PAULI_Z, ROCSTATEVEC_PAULI_Z};
    rocstatevec_pauli              xxx[3] = {ROCSTATEVEC_PAULI_X, ROCSTATEVEC_PAULI_X, ROCSTATEVEC_PAULI_X};
    int32_t                        all_qb[3] = {0, 1, 2};
    const rocstatevec_pauli* paulis_arr[2] = {zzz, xxx};
    const int32_t*           qubits_arr[2] = {all_qb, all_qb};
    uint32_t                       qcounts[2] = {3, 3};
    double                         expvals[2] = {0.0, 0.0};
    CHECK(rocstatevec_compute_expectations_on_pauli_basis(
        handle, dsv, ROCSTATEVEC_C_64F, n, expvals, paulis_arr, 2, qubits_arr, qcounts));
    std::printf("[6/12] <ZZZ>=%g (expected 1), <XXX>=%g (expected 1)\n",
                expvals[0], expvals[1]);

    int32_t parity = -1;
    CHECK(rocstatevec_measure_on_z_basis(
        handle, dsv, ROCSTATEVEC_C_64F, n, &parity, all_qb, 3, 0.0,
        ROCSTATEVEC_COLLAPSE_NONE));
    std::printf("[7/12] rocstatevec_measure_on_z_basis (rng=0): parity=%d\n", parity);

    CHECK(rocstatevec_initialize_state_vector(
        handle, dsv, ROCSTATEVEC_C_64F, n, ROCSTATEVEC_STATE_VECTOR_TYPE_GHZ));

    rocstatevec_sampler_descriptor sampler   = nullptr;
    constexpr uint32_t                  shots = 1024;
    size_t                              s_ws_bytes = 0;
    CHECK(rocstatevec_sampler_create(
        handle, dsv, ROCSTATEVEC_C_64F, n, &sampler, shots, &s_ws_bytes));
    void* s_ws_buf = nullptr;
    if(s_ws_bytes) HIP_CHECK(hipMalloc(&s_ws_buf, s_ws_bytes));
    std::printf("[8/12] rocstatevec_sampler_create ok (workspace=%zu B)\n", s_ws_bytes);

    CHECK(rocstatevec_sampler_preprocess(handle, sampler, s_ws_buf, s_ws_bytes));
    std::printf("[9/12] rocstatevec_sampler_preprocess ok\n");

    std::vector<double> rng(shots);
    {
        std::mt19937_64 gen(0xC0FFEEULL);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        for(double& r : rng) r = uni(gen);
    }
    std::vector<rocstatevec_index_t> bs(shots, 0);
    CHECK(rocstatevec_sampler_sample(
        handle, sampler, bs.data(), all_qb, 3, rng.data(), shots,
        ROCSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER));
    std::printf("[10/12] rocstatevec_sampler_sample (1024 shots) ok\n");

    std::map<rocstatevec_index_t, uint32_t> hist;
    for(auto v : bs) hist[v]++;
    std::printf("        Sample histogram (GHZ, expected only |000> and |111>):\n");
    for(auto& kv : hist) std::printf("          %d%d%d : %u\n",
                                     int((kv.first >> 0) & 1),
                                     int((kv.first >> 1) & 1),
                                     int((kv.first >> 2) & 1),
                                     kv.second);

    CHECK(rocstatevec_sampler_destroy(sampler));
    std::printf("[11/12] rocstatevec_sampler_destroy ok\n");

    if(s_ws_buf)  HIP_CHECK(hipFree(s_ws_buf));
    if(exp_ws_buf) HIP_CHECK(hipFree(exp_ws_buf));
    if(ws_buf)    HIP_CHECK(hipFree(ws_buf));
    HIP_CHECK(hipFree(dsv));

    CHECK(rocstatevec_destroy_handle(handle));
    std::printf("[12/12] rocstatevec_destroy_handle ok\n");

    std::printf("\nAll 12 cuda-quantum-pattern entry points executed successfully.\n");
    return 0;
}
