/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * <O> for pure / mixed states. Pure: <psi|O|psi>. Mixed: Tr(O rho).
 *
 * Implementation reuses apply_op_pure_into to compute y = O|x> (pure)
 * or a column of (O rho) (mixed); the pure-state expectation closes
 * with a host-side conj(x) . y dot product, the mixed-state expectation
 * sums the diagonal of (O rho) on the host.
 *
 * The host-side reduction is fine for v0.1.0's small Hilbert spaces
 * (< 256 amplitudes in the smoke samples). v0.2 swaps it for a rocBLAS
 * complex dot.
 * ************************************************************************ */

#include "rocdensitymat_descriptors.hpp"
#include "rocdensitymat_kernels.hpp"

#include <complex>
#include <vector>

namespace rocdensitymat
{

template <typename Cmplx>
rocdensitymat_status
apply_op_pure_into(handle_impl* h,
                   rocdensitymat_operator op,
                   double t,
                   int32_t num_params,
                   const double* params,
                   const Cmplx* x,
                   Cmplx* y,
                   bool accumulate,
                   Cmplx* buf_a,
                   Cmplx* buf_b,
                   int64_t n_elems);

namespace
{

template <typename Cmplx>
rocdensitymat_status host_conj_dot(hipStream_t stream,
                                   const Cmplx* d_a,
                                   const Cmplx* d_b,
                                   int64_t n,
                                   rocdensitymat_complex_double* out)
{
    std::vector<Cmplx> ha(n), hb(n);
    if(hipMemcpyAsync(ha.data(), d_a, n * sizeof(Cmplx),
                      hipMemcpyDeviceToHost, stream) != hipSuccess)
        return ROCDENSITYMAT_STATUS_HIP_ERROR;
    if(hipMemcpyAsync(hb.data(), d_b, n * sizeof(Cmplx),
                      hipMemcpyDeviceToHost, stream) != hipSuccess)
        return ROCDENSITYMAT_STATUS_HIP_ERROR;
    if(hipStreamSynchronize(stream) != hipSuccess)
        return ROCDENSITYMAT_STATUS_HIP_ERROR;

    double re = 0;
    double im = 0;
    for(int64_t i = 0; i < n; ++i)
    {
        // conj(a) * b
        re += static_cast<double>(ha[i].x) * static_cast<double>(hb[i].x)
            + static_cast<double>(ha[i].y) * static_cast<double>(hb[i].y);
        im += static_cast<double>(ha[i].x) * static_cast<double>(hb[i].y)
            - static_cast<double>(ha[i].y) * static_cast<double>(hb[i].x);
    }
    out->x = re;
    out->y = im;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

template <typename Cmplx>
rocdensitymat_status host_diag_sum(hipStream_t stream,
                                   const Cmplx* d_v,
                                   int64_t D,
                                   rocdensitymat_complex_double* out)
{
    std::vector<Cmplx> hv(D * D);
    if(hipMemcpyAsync(hv.data(), d_v, D * D * sizeof(Cmplx),
                      hipMemcpyDeviceToHost, stream) != hipSuccess)
        return ROCDENSITYMAT_STATUS_HIP_ERROR;
    if(hipStreamSynchronize(stream) != hipSuccess)
        return ROCDENSITYMAT_STATUS_HIP_ERROR;
    double re = 0;
    double im = 0;
    for(int64_t i = 0; i < D; ++i)
    {
        re += static_cast<double>(hv[i * D + i].x);
        im += static_cast<double>(hv[i * D + i].y);
    }
    out->x = re;
    out->y = im;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

} // namespace

} // namespace rocdensitymat

extern "C" {

rocdensitymat_status rocdensitymat_operator_compute_expectation(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    double t,
    int32_t num_params,
    const double* params,
    rocdensitymat_state state,
    rocdensitymat_workspace_descriptor workspace,
    rocdensitymat_complex_double* expectation)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(op);
    ROCDENSITYMAT_CHECK_PTR(state);
    ROCDENSITYMAT_CHECK_PTR(workspace);
    ROCDENSITYMAT_CHECK_PTR(expectation);
    if(state->component_buffer == nullptr)
        return ROCDENSITYMAT_STATUS_NOT_INITIALIZED;
    if(op->has_collapse_term())
        return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;
    if(state->purity == ROCDENSITYMAT_STATE_PURITY_MPS)
        return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;

    auto* hi = static_cast<rocdensitymat::handle_impl*>(handle);

    if(state->purity == ROCDENSITYMAT_STATE_PURITY_PURE)
    {
        int64_t n = state->component_elems;
        // Need 3 D-sized scratch slots: ping-pong (2 D) + output y (1 D).
        size_t need = 3 * static_cast<size_t>(n)
                    * rocdensitymat::element_size_bytes(state->data_type);
        if(workspace->device_scratch_bytes < need
           || workspace->device_scratch_ptr == nullptr)
        {
            workspace->required_device_scratch_bytes = need;
            return ROCDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE;
        }
        if(state->data_type == ROCDENSITYMAT_C_64F)
        {
            auto* base = static_cast<hipDoubleComplex*>(workspace->device_scratch_ptr);
            auto* sa = base;
            auto* sb = sa + n;
            auto* y  = sb + n;
            auto  rc = rocdensitymat::apply_op_pure_into<hipDoubleComplex>(
                hi, op, t, num_params, params,
                static_cast<const hipDoubleComplex*>(state->component_buffer),
                y, /*accumulate*/false, sa, sb, n);
            if(rc != ROCDENSITYMAT_STATUS_SUCCESS) return rc;
            return rocdensitymat::host_conj_dot<hipDoubleComplex>(
                hi->stream,
                static_cast<const hipDoubleComplex*>(state->component_buffer),
                y, n, expectation);
        }
        auto* base = static_cast<hipFloatComplex*>(workspace->device_scratch_ptr);
        auto* sa = base;
        auto* sb = sa + n;
        auto* y  = sb + n;
        auto  rc = rocdensitymat::apply_op_pure_into<hipFloatComplex>(
            hi, op, t, num_params, params,
            static_cast<const hipFloatComplex*>(state->component_buffer),
            y, /*accumulate*/false, sa, sb, n);
        if(rc != ROCDENSITYMAT_STATUS_SUCCESS) return rc;
        return rocdensitymat::host_conj_dot<hipFloatComplex>(
            hi->stream,
            static_cast<const hipFloatComplex*>(state->component_buffer),
            y, n, expectation);
    }

    // Mixed: compute O rho via column-wise pure apply, then sum diagonal.
    int64_t D    = state->hilbert_dim();
    int64_t N    = D * D;
    size_t  el   = rocdensitymat::element_size_bytes(state->data_type);
    size_t  need = (2 * static_cast<size_t>(D) + static_cast<size_t>(N)) * el;
    if(workspace->device_scratch_bytes < need
       || workspace->device_scratch_ptr == nullptr)
    {
        workspace->required_device_scratch_bytes = need;
        return ROCDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE;
    }

    if(state->data_type == ROCDENSITYMAT_C_64F)
    {
        auto* base = static_cast<hipDoubleComplex*>(workspace->device_scratch_ptr);
        auto* sa = base;
        auto* sb = sa + D;
        auto* v  = sb + D;
        // Zero v.
        hipDoubleComplex zero; zero.x = 0; zero.y = 0;
        int     bs = rocdensitymat::default_threads_per_block;
        int64_t gx = rocdensitymat::ceil_div<int64_t>(N, bs);
        hipLaunchKernelGGL(rocdensitymat::fill_kernel<hipDoubleComplex>,
                           dim3(static_cast<unsigned int>(gx)),
                           dim3(bs), 0, hi->stream, v, zero, N);
        if(hipGetLastError() != hipSuccess)
            return ROCDENSITYMAT_STATUS_EXECUTION_FAILED;

        auto* rho = static_cast<const hipDoubleComplex*>(state->component_buffer);
        for(int64_t j = 0; j < D; ++j)
        {
            auto rc = rocdensitymat::apply_op_pure_into<hipDoubleComplex>(
                hi, op, t, num_params, params,
                rho + j * D, v + j * D, /*accumulate*/true, sa, sb, D);
            if(rc != ROCDENSITYMAT_STATUS_SUCCESS) return rc;
        }
        return rocdensitymat::host_diag_sum<hipDoubleComplex>(
            hi->stream, v, D, expectation);
    }
    auto* base = static_cast<hipFloatComplex*>(workspace->device_scratch_ptr);
    auto* sa = base;
    auto* sb = sa + D;
    auto* v  = sb + D;
    hipFloatComplex zero; zero.x = 0; zero.y = 0;
    int     bs = rocdensitymat::default_threads_per_block;
    int64_t gx = rocdensitymat::ceil_div<int64_t>(N, bs);
    hipLaunchKernelGGL(rocdensitymat::fill_kernel<hipFloatComplex>,
                       dim3(static_cast<unsigned int>(gx)),
                       dim3(bs), 0, hi->stream, v, zero, N);
    if(hipGetLastError() != hipSuccess)
        return ROCDENSITYMAT_STATUS_EXECUTION_FAILED;
    auto* rho = static_cast<const hipFloatComplex*>(state->component_buffer);
    for(int64_t j = 0; j < D; ++j)
    {
        auto rc = rocdensitymat::apply_op_pure_into<hipFloatComplex>(
            hi, op, t, num_params, params,
            rho + j * D, v + j * D, /*accumulate*/true, sa, sb, D);
        if(rc != ROCDENSITYMAT_STATUS_SUCCESS) return rc;
    }
    return rocdensitymat::host_diag_sum<hipFloatComplex>(
        hi->stream, v, D, expectation);
}

} // extern "C"
