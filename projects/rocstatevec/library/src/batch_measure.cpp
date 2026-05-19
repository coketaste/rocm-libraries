/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * `batch_measure` — full Z-basis projective measurement on the qubits
 * named by `bit_ordering`. Returns the resulting bit-string and
 * optionally collapses + renormalizes the state vector.
 *
 * Algorithm:
 *   1. Use `abs2_sum_array` to get the probability mass for each of the
 *      2^bs_len possible outcomes.
 *   2. Build cumulative distribution on the host (bs_len <= ~30 in
 *      realistic use; the array fits in cache).
 *   3. Pick the smallest k with cum[k] >= randnum * total_norm.
 *   4. Decompose k into bits and write into bit_string.
 *   5. If collapse requested, call `collapse_by_bit_string`.
 * ************************************************************************ */

#include "rocstatevec_internal.hpp"

#include <vector>

extern "C" rocstatevec_status rocstatevec_abs2_sum_array(
    rocstatevec_handle, const void*, rocstatevec_data_type, uint32_t, double*,
    const int32_t*, uint32_t, const int32_t*, const int32_t*, uint32_t);
extern "C" rocstatevec_status rocstatevec_collapse_by_bit_string(
    rocstatevec_handle, void*, rocstatevec_data_type, uint32_t,
    const int32_t*, const int32_t*, uint32_t, double);

static rocstatevec_status batch_core(
    rocstatevec_handle h, void* dsv, rocstatevec_data_type dtype, uint32_t n,
    int32_t* bit_string, const int32_t* bit_ordering, uint32_t bs_len,
    double randnum, rocstatevec_collapse_op collapse,
    double offset, double explicit_total)
{
    if(bs_len == 0)                   return ROCSTATEVEC_STATUS_INVALID_VALUE;
    if(bs_len > 30)                   return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
    if(randnum < 0.0 || randnum >= 1.0) return ROCSTATEVEC_STATUS_INVALID_VALUE;

    size_t sz = size_t{1} << bs_len;
    std::vector<double> probs(sz, 0.0);
    auto rc = rocstatevec_abs2_sum_array(h, dsv, dtype, n,
                                         probs.data(), bit_ordering, bs_len,
                                         nullptr, nullptr, 0);
    if(rc != ROCSTATEVEC_STATUS_SUCCESS) return rc;

    double total = 0.0;
    for(double p : probs) total += p;
    double total_norm = (explicit_total > 0.0) ? explicit_total : total;
    if(total_norm <= 0.0) return ROCSTATEVEC_STATUS_INTERNAL_ERROR;

    double target = (randnum - offset) * total_norm;
    if(target < 0.0)        target = 0.0;
    if(target > total_norm) target = total_norm;

    double  acc = 0.0;
    size_t  pick = sz - 1;
    for(size_t i = 0; i < sz; ++i)
    {
        acc += probs[i];
        if(acc >= target) { pick = i; break; }
    }

    for(uint32_t b = 0; b < bs_len; ++b)
        bit_string[b] = (pick >> b) & 1;

    if(collapse == ROCSTATEVEC_COLLAPSE_NORMALIZE_AS_SPECIFIED)
    {
        double norm_pick = probs[pick];
        if(norm_pick <= 0.0) return ROCSTATEVEC_STATUS_INTERNAL_ERROR;
        rc = rocstatevec_collapse_by_bit_string(h, dsv, dtype, n,
                                                bit_string, bit_ordering, bs_len,
                                                norm_pick);
        if(rc != ROCSTATEVEC_STATUS_SUCCESS) return rc;
    }
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_batch_measure(
    rocstatevec_handle h, void* dsv, rocstatevec_data_type dtype, uint32_t n,
    int32_t* bit_string, const int32_t* bit_ordering, uint32_t bs_len,
    double randnum, rocstatevec_collapse_op collapse)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(dsv); ROCSTATEVEC_CHECK_PTR(bit_string); ROCSTATEVEC_CHECK_PTR(bit_ordering);
    return batch_core(h, dsv, dtype, n, bit_string, bit_ordering, bs_len,
                      randnum, collapse, 0.0, 0.0);
}

extern "C" rocstatevec_status rocstatevec_batch_measure_with_offset(
    rocstatevec_handle h, void* dsv, rocstatevec_data_type dtype, uint32_t n,
    int32_t* bit_string, const int32_t* bit_ordering, uint32_t bs_len,
    double randnum, rocstatevec_collapse_op collapse,
    double offset, double abs2_sum)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(dsv); ROCSTATEVEC_CHECK_PTR(bit_string); ROCSTATEVEC_CHECK_PTR(bit_ordering);
    return batch_core(h, dsv, dtype, n, bit_string, bit_ordering, bs_len,
                      randnum, collapse, offset, abs2_sum);
}
