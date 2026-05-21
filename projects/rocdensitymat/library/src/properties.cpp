/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "rocdensitymat_descriptors.hpp"

#include <cmath>
#include <vector>

namespace
{

template <typename Cmplx>
rocdensitymat_status copy_state_to_host(hipStream_t stream,
                                        rocdensitymat_state state,
                                        std::vector<Cmplx>& dst)
{
    dst.resize(static_cast<size_t>(state->component_elems));
    if(hipMemcpyAsync(dst.data(),
                      state->component_buffer,
                      state->component_bytes,
                      hipMemcpyDeviceToHost,
                      stream) != hipSuccess)
        return ROCDENSITYMAT_STATUS_HIP_ERROR;
    if(hipStreamSynchronize(stream) != hipSuccess)
        return ROCDENSITYMAT_STATUS_HIP_ERROR;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

template <typename Cmplx>
double l2_norm_host(const std::vector<Cmplx>& v)
{
    double acc = 0;
    for(auto& z : v)
    {
        double r = static_cast<double>(z.x);
        double i = static_cast<double>(z.y);
        acc += r * r + i * i;
    }
    return std::sqrt(acc);
}

template <typename Cmplx>
rocdensitymat_complex_double trace_host(const std::vector<Cmplx>& v, int64_t D)
{
    rocdensitymat_complex_double t;
    t.x = 0;
    t.y = 0;
    for(int64_t i = 0; i < D; ++i)
    {
        t.x += static_cast<double>(v[i * D + i].x);
        t.y += static_cast<double>(v[i * D + i].y);
    }
    return t;
}

template <typename Cmplx>
rocdensitymat_complex_double overlap_host(const std::vector<Cmplx>& a,
                                          const std::vector<Cmplx>& b)
{
    rocdensitymat_complex_double o;
    o.x = 0;
    o.y = 0;
    for(size_t i = 0; i < a.size(); ++i)
    {
        // conj(a) * b
        double ar = static_cast<double>(a[i].x);
        double ai = static_cast<double>(a[i].y);
        double br = static_cast<double>(b[i].x);
        double bi = static_cast<double>(b[i].y);
        o.x += ar * br + ai * bi;
        o.y += ar * bi - ai * br;
    }
    return o;
}

} // namespace

extern "C" {

rocdensitymat_status rocdensitymat_state_compute_norm(
    rocdensitymat_handle handle, rocdensitymat_state state,
    rocdensitymat_workspace_descriptor workspace, double* norm)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(state);
    ROCDENSITYMAT_CHECK_PTR(norm);
    (void)workspace; // host reduction in v0.1.0; rocBLAS path lands in v0.2.

    if(state->component_buffer == nullptr)
        return ROCDENSITYMAT_STATUS_NOT_INITIALIZED;

    if(state->data_type == ROCDENSITYMAT_C_64F)
    {
        std::vector<hipDoubleComplex> hv;
        ROCDENSITYMAT_RC_CHECK(copy_state_to_host(handle->stream, state, hv));
        *norm = l2_norm_host(hv);
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    std::vector<hipFloatComplex> hv;
    ROCDENSITYMAT_RC_CHECK(copy_state_to_host(handle->stream, state, hv));
    *norm = l2_norm_host(hv);
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_state_compute_trace(
    rocdensitymat_handle handle, rocdensitymat_state state,
    rocdensitymat_workspace_descriptor workspace,
    rocdensitymat_complex_double* trace)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(state);
    ROCDENSITYMAT_CHECK_PTR(trace);
    (void)workspace;

    if(state->purity != ROCDENSITYMAT_STATE_PURITY_MIXED)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    if(state->component_buffer == nullptr)
        return ROCDENSITYMAT_STATUS_NOT_INITIALIZED;

    int64_t D = state->hilbert_dim();
    if(state->data_type == ROCDENSITYMAT_C_64F)
    {
        std::vector<hipDoubleComplex> hv;
        ROCDENSITYMAT_RC_CHECK(copy_state_to_host(handle->stream, state, hv));
        *trace = trace_host(hv, D);
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    std::vector<hipFloatComplex> hv;
    ROCDENSITYMAT_RC_CHECK(copy_state_to_host(handle->stream, state, hv));
    *trace = trace_host(hv, D);
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_state_compute_overlap(
    rocdensitymat_handle handle, rocdensitymat_state lhs,
    rocdensitymat_state rhs,
    rocdensitymat_workspace_descriptor workspace,
    rocdensitymat_complex_double* overlap)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(lhs);
    ROCDENSITYMAT_CHECK_PTR(rhs);
    ROCDENSITYMAT_CHECK_PTR(overlap);
    (void)workspace;

    if(lhs->purity != rhs->purity || lhs->data_type != rhs->data_type
       || lhs->component_elems != rhs->component_elems)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    if(lhs->component_buffer == nullptr || rhs->component_buffer == nullptr)
        return ROCDENSITYMAT_STATUS_NOT_INITIALIZED;

    if(lhs->data_type == ROCDENSITYMAT_C_64F)
    {
        std::vector<hipDoubleComplex> ha, hb;
        ROCDENSITYMAT_RC_CHECK(copy_state_to_host(handle->stream, lhs, ha));
        ROCDENSITYMAT_RC_CHECK(copy_state_to_host(handle->stream, rhs, hb));
        *overlap = overlap_host(ha, hb);
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    std::vector<hipFloatComplex> ha, hb;
    ROCDENSITYMAT_RC_CHECK(copy_state_to_host(handle->stream, lhs, ha));
    ROCDENSITYMAT_RC_CHECK(copy_state_to_host(handle->stream, rhs, hb));
    *overlap = overlap_host(ha, hb);
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_state_get_data_type(
    rocdensitymat_handle handle, rocdensitymat_state state,
    rocdensitymat_data_type* data_type)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(state);
    ROCDENSITYMAT_CHECK_PTR(data_type);
    *data_type = state->data_type;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_state_get_purity(
    rocdensitymat_handle handle, rocdensitymat_state state,
    rocdensitymat_state_purity* purity)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(state);
    ROCDENSITYMAT_CHECK_PTR(purity);
    *purity = state->purity;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_state_get_space_shape(
    rocdensitymat_handle handle, rocdensitymat_state state,
    int32_t* num_space_modes, int64_t* space_shape)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(state);
    if(num_space_modes != nullptr) *num_space_modes = state->num_modes;
    if(space_shape != nullptr)
    {
        for(int32_t i = 0; i < state->num_modes; ++i)
            space_shape[i] = state->space_shape[i];
    }
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

} // extern "C"
