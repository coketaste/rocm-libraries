/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Z-basis projective measurement.
 *
 * Algorithm:
 *   1. Compute s0 = sum_{parity(i & mask)==0} |amp[i]|^2 and s1 similarly.
 *   2. norm = s0 + s1; p0 = s0 / norm. If randnum < p0, parity = 0,
 *      otherwise parity = 1.
 *   3. If collapse == NORMALIZE_AS_SPECIFIED, run `collapse_on_z_basis`
 *      to zero amplitudes whose parity differs from the chosen value
 *      and renormalize the rest by 1/sqrt(s_chosen).
 * ************************************************************************ */

#include "rocstatevec_kernels.hpp"

extern "C" rocstatevec_status rocstatevec_abs2_sum_on_z_basis(
    rocstatevec_handle, const void*, rocstatevec_data_type, uint32_t,
    double*, double*, const int32_t*, uint32_t);

extern "C" rocstatevec_status rocstatevec_collapse_on_z_basis(
    rocstatevec_handle, void*, rocstatevec_data_type, uint32_t, int32_t,
    const int32_t*, uint32_t, double);

extern "C" rocstatevec_status rocstatevec_measure_on_z_basis(
    rocstatevec_handle h,
    void*                   state_vector,
    rocstatevec_data_type   dtype,
    uint32_t                n_index_bits,
    int32_t*                parity,
    const int32_t*          basis_bits,
    uint32_t                n_basis_bits,
    double                  randnum,
    rocstatevec_collapse_op collapse)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(state_vector);
    ROCSTATEVEC_CHECK_PTR(parity);
    if(n_basis_bits == 0) return ROCSTATEVEC_STATUS_INVALID_VALUE;
    ROCSTATEVEC_CHECK_PTR(basis_bits);
    if(randnum < 0.0 || randnum >= 1.0) return ROCSTATEVEC_STATUS_INVALID_VALUE;

    double s0 = 0, s1 = 0;
    auto rc = rocstatevec_abs2_sum_on_z_basis(h, state_vector, dtype, n_index_bits,
                                              &s0, &s1, basis_bits, n_basis_bits);
    if(rc != ROCSTATEVEC_STATUS_SUCCESS) return rc;

    double norm = s0 + s1;
    if(norm <= 0.0) return ROCSTATEVEC_STATUS_INTERNAL_ERROR;
    double p0   = s0 / norm;
    int    sel  = (randnum < p0) ? 0 : 1;
    *parity     = sel;

    if(collapse == ROCSTATEVEC_COLLAPSE_NORMALIZE_AS_SPECIFIED)
    {
        double s_sel = (sel == 0) ? s0 : s1;
        if(s_sel <= 0.0) return ROCSTATEVEC_STATUS_INTERNAL_ERROR;
        rc = rocstatevec_collapse_on_z_basis(h, state_vector, dtype, n_index_bits,
                                             sel, basis_bits, n_basis_bits, s_sel);
        if(rc != ROCSTATEVEC_STATUS_SUCCESS) return rc;
    }
    return ROCSTATEVEC_STATUS_SUCCESS;
}
