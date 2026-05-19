/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * GHZ-state sampling demo: builds an N-qubit GHZ via the named preset and
 * draws shots through the sampler API.
 * ************************************************************************ */

#include <rocstatevec.h>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

#define HIPCHK(expr) do { hipError_t e = (expr); if(e != hipSuccess) { std::fprintf(stderr, "HIP error %d at %s:%d\n", (int)e, __FILE__, __LINE__); return 1; } } while(0)
#define SVCHK(expr)  do { rocstatevec_status s = (expr); if(s != ROCSTATEVEC_STATUS_SUCCESS) { std::fprintf(stderr, "rocSTATEVEC error %s: %s\n", rocstatevec_get_error_name(s), rocstatevec_get_error_string(s)); return 1; } } while(0)

int main()
{
    constexpr uint32_t n_qubits = 5;
    constexpr uint32_t n_shots  = 1024;

    rocstatevec_handle h = nullptr;
    SVCHK(rocstatevec_create_handle(&h));

    void* dsv = nullptr;
    HIPCHK(hipMalloc(&dsv, (size_t{1} << n_qubits) * 16));
    SVCHK(rocstatevec_initialize_state_vector(h, dsv, ROCSTATEVEC_C_64F, n_qubits,
                                              ROCSTATEVEC_STATE_VECTOR_TYPE_GHZ));

    rocstatevec_sampler_descriptor sampler = nullptr;
    size_t ws_bytes = 0;
    SVCHK(rocstatevec_sampler_create(h, dsv, ROCSTATEVEC_C_64F, n_qubits, &sampler, n_shots, &ws_bytes));
    void* ws = nullptr;
    if(ws_bytes) HIPCHK(hipMalloc(&ws, ws_bytes));
    SVCHK(rocstatevec_sampler_preprocess(h, sampler, ws, ws_bytes));

    std::vector<int32_t>  ordering(n_qubits);
    for(uint32_t i = 0; i < n_qubits; ++i) ordering[i] = i;
    std::vector<rocstatevec_index_t> samples(n_shots);

    std::mt19937_64                rng(42);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double>            randnums(n_shots);
    for(auto& r : randnums) r = u(rng);

    SVCHK(rocstatevec_sampler_sample(h, sampler, samples.data(),
                                     ordering.data(), n_qubits, randnums.data(),
                                     n_shots, ROCSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER));
    SVCHK(rocstatevec_sampler_destroy(sampler));

    size_t zeros = 0, all_ones = 0;
    for(auto s : samples)
    {
        if(s == 0)                       ++zeros;
        if(s == ((rocstatevec_index_t{1} << n_qubits) - 1)) ++all_ones;
    }
    std::printf("GHZ shots: |0...0>=%zu  |1...1>=%zu  other=%zu\n",
                zeros, all_ones, size_t{n_shots} - zeros - all_ones);

    if(ws) HIPCHK(hipFree(ws));
    HIPCHK(hipFree(dsv));
    SVCHK(rocstatevec_destroy_handle(h));
    return 0;
}
