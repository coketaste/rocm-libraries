/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Phase 5 — Sampler + BatchMeasure + Collapse + Abs2SumArray.
 * ************************************************************************ */

#include "test_helpers.hpp"

#include <gtest/gtest.h>

#include <random>

using namespace rocstatevec::test;
using cd = std::complex<double>;

TEST(Sampler, ghz_samples_are_all_zeros_or_all_ones)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n_qubits = 4;
    constexpr uint32_t n_shots  = 4096;

    device_buffer<cd> dsv(size_t{1} << n_qubits);
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_initialize_state_vector(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n_qubits,
                                                  ROCSTATEVEC_STATE_VECTOR_TYPE_GHZ));

    rocstatevec_sampler_descriptor sampler = nullptr;
    size_t ws = 0;
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_sampler_create(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n_qubits,
                                         &sampler, n_shots, &ws));
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_sampler_preprocess(h.h, sampler, nullptr, 0));

    double norm = 0;
    EXPECT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_sampler_get_squared_norm(h.h, sampler, &norm));
    EXPECT_NEAR(norm, 1.0, tol_fp64);

    std::vector<int32_t> ord(n_qubits);
    for(uint32_t i = 0; i < n_qubits; ++i) ord[i] = i;

    std::mt19937_64                        rng(123);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double>                    rs(n_shots);
    for(auto& r : rs) r = u(rng);

    std::vector<rocstatevec_index_t> samples(n_shots);
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_sampler_sample(h.h, sampler, samples.data(),
                                         ord.data(), n_qubits, rs.data(),
                                         n_shots, ROCSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER));

    rocstatevec_sampler_destroy(sampler);

    rocstatevec_index_t all_one = (rocstatevec_index_t{1} << n_qubits) - 1;
    size_t z = 0, o = 0;
    for(auto s : samples)
    {
        EXPECT_TRUE(s == 0 || s == all_one) << "unexpected sample 0x" << std::hex << s;
        if(s == 0)        ++z;
        if(s == all_one)  ++o;
    }
    double frac = double(z) / double(n_shots);
    EXPECT_GT(frac, 0.40);
    EXPECT_LT(frac, 0.60);
    EXPECT_EQ(z + o, n_shots);
}

TEST(Sampler, batch_measure_collapses_to_outcome)
{
    if(skip_if_no_gpu()) return;
    handle_guard h;
    constexpr uint32_t n = 3;
    constexpr size_t   N = size_t{1} << n;
    device_buffer<cd> dsv(N);
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_initialize_state_vector(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                                  ROCSTATEVEC_STATE_VECTOR_TYPE_UNIFORM));

    int32_t ord[3] = {0, 1, 2};
    int32_t bs[3]  = {-1, -1, -1};
    ASSERT_EQ(ROCSTATEVEC_STATUS_SUCCESS,
              rocstatevec_batch_measure(h.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                        bs, ord, 3, 0.5,
                                        ROCSTATEVEC_COLLAPSE_NORMALIZE_AS_SPECIFIED));
    for(int b : bs) EXPECT_TRUE(b == 0 || b == 1);

    std::vector<cd> hv(N);
    copy_to_host(hv.data(), dsv.ptr, N);
    EXPECT_NEAR(l2_norm_squared(hv), 1.0, tol_fp64);

    int    nonzero = 0;
    size_t which   = ~size_t{0};
    for(size_t i = 0; i < N; ++i)
        if(std::abs(hv[i]) > 0.5)
        {
            ++nonzero;
            which = i;
        }
    EXPECT_EQ(nonzero, 1);
    size_t expected = size_t(bs[0]) | (size_t(bs[1]) << 1) | (size_t(bs[2]) << 2);
    EXPECT_EQ(which, expected);
}
