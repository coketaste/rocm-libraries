/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Pure-state operator-action implementation. The kernel pattern reuses
 * the rocSTATEVEC tensor-product apply: split the flat state into
 * (high, mode_block, low) strides and apply the elementary operator on
 * the mode block via a small dense GEMV-style kernel.
 *
 * v0.1.0 only ships single-mode elementary operators; the OperatorTerm
 * layer rejects multi-mode factors with NOT_SUPPORTED already.
 * ************************************************************************ */

#include "rocdensitymat_descriptors.hpp"
#include "rocdensitymat_kernels.hpp"

#include <cmath>
#include <complex>

namespace rocdensitymat
{

namespace
{

template <typename Cmplx>
rocdensitymat_status upload_elementary_data(handle_impl* h,
                                            rocdensitymat_elementary_operator e,
                                            void** dev_ptr_out)
{
    if(e->device_data != nullptr)
    {
        *dev_ptr_out = e->device_data;
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    size_t bytes = e->host_data.size() * sizeof(Cmplx);
    void*  dptr  = nullptr;
    auto   rc    = dev_alloc(h, &dptr, bytes, h->stream);
    if(rc != ROCDENSITYMAT_STATUS_SUCCESS) return rc;

    std::vector<Cmplx> tmp(e->host_data.size());
    for(size_t i = 0; i < e->host_data.size(); ++i)
    {
        tmp[i].x = static_cast<decltype(tmp[i].x)>(e->host_data[i].real());
        tmp[i].y = static_cast<decltype(tmp[i].y)>(e->host_data[i].imag());
    }
    if(hipMemcpyAsync(dptr, tmp.data(), bytes, hipMemcpyHostToDevice, h->stream)
       != hipSuccess)
    {
        (void)dev_free(h, dptr, bytes, h->stream);
        return ROCDENSITYMAT_STATUS_HIP_ERROR;
    }
    if(hipStreamSynchronize(h->stream) != hipSuccess)
    {
        (void)dev_free(h, dptr, bytes, h->stream);
        return ROCDENSITYMAT_STATUS_HIP_ERROR;
    }
    e->device_data        = dptr;
    e->device_data_bytes  = bytes;
    e->device_data_stream = h->stream;
    *dev_ptr_out          = dptr;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

template <typename Cmplx>
rocdensitymat_status apply_single_factor(handle_impl* h,
                                         const Cmplx* x,
                                         Cmplx* y,
                                         const std::vector<int64_t>& space_shape,
                                         int32_t mode,
                                         rocdensitymat_elementary_operator e)
{
    int64_t mat_dim  = space_shape[mode];
    int64_t low_dim  = 1;
    int64_t high_dim = 1;
    for(int32_t i = static_cast<int32_t>(space_shape.size()) - 1; i > mode; --i)
        low_dim *= space_shape[i];
    for(int32_t i = 0; i < mode; ++i)
        high_dim *= space_shape[i];

    void* dptr = nullptr;
    ROCDENSITYMAT_RC_CHECK(upload_elementary_data<Cmplx>(h, e, &dptr));

    int   bs = 8;
    int   ts = 4;
    dim3  block(bs, ts, ts);
    dim3  grid(static_cast<unsigned int>(ceil_div<int64_t>(low_dim, bs)),
              static_cast<unsigned int>(ceil_div<int64_t>(high_dim, ts)),
              static_cast<unsigned int>(ceil_div<int64_t>(mat_dim, ts)));

    if(e->kind == ROCDENSITYMAT_ELEMENTARY_DIAGONAL)
    {
        hipLaunchKernelGGL(apply_diagonal_block_kernel<Cmplx>,
                           grid, block, 0, h->stream,
                           reinterpret_cast<const Cmplx*>(dptr), mat_dim,
                           x, y, high_dim, low_dim);
    }
    else
    {
        hipLaunchKernelGGL(apply_dense_block_kernel<Cmplx>,
                           grid, block, 0, h->stream,
                           reinterpret_cast<const Cmplx*>(dptr), mat_dim,
                           x, y, high_dim, low_dim);
    }
    if(hipGetLastError() != hipSuccess)
        return ROCDENSITYMAT_STATUS_EXECUTION_FAILED;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

} // namespace

/*! \brief Per-component scratch requirement of apply_op_pure_into. */
size_t apply_op_pure_scratch_bytes_per_component(rocdensitymat_state state)
{
    return 2 * state->component_bytes;
}

/*! \brief y_out = sum_t coeff_t * O_t |x>. Pure-state action of the entire
 *  operator on a state vector. The caller supplies two device-side
 *  ping-pong buffers each at least n*sizeof(Cmplx) bytes. When `accumulate`
 *  is false the caller's y is zero-filled before accumulation. */
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
                   int64_t n_elems)
{
    int     bs = default_threads_per_block;
    int64_t gx = ceil_div<int64_t>(n_elems, bs);

    if(!accumulate)
    {
        Cmplx zero;
        zero.x = 0; zero.y = 0;
        hipLaunchKernelGGL(fill_kernel<Cmplx>,
                           dim3(static_cast<unsigned int>(gx)),
                           dim3(bs), 0, h->stream, y, zero, n_elems);
        if(hipGetLastError() != hipSuccess)
            return ROCDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    for(auto& te : op->terms)
    {
        if(te.duality_offset != 0)
            return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;

        std::complex<double> term_coeff(te.coefficient.x, te.coefficient.y);
        if(te.tdep_callback != nullptr)
        {
            auto cb_val = te.tdep_callback(t, num_params, params);
            term_coeff *= std::complex<double>(cb_val.x, cb_val.y);
        }
        for(auto& p : te.term->products)
        {
            std::complex<double> prod_coeff(p.coefficient.x, p.coefficient.y);
            if(p.tdep_callback != nullptr)
            {
                auto cb_val = p.tdep_callback(t, num_params, params);
                prod_coeff *= std::complex<double>(cb_val.x, cb_val.y);
            }
            std::complex<double> total = term_coeff * prod_coeff;

            // Apply the chain of single-mode factors, ping-ponging buf_a / buf_b.
            const Cmplx* in_ptr  = x;
            Cmplx*       out_ptr = buf_a;
            for(size_t f = 0; f < p.operators.size(); ++f)
            {
                if(p.mode_action_duality[f] != ROCDENSITYMAT_DUALITY_KET)
                    return ROCDENSITYMAT_STATUS_INVALID_VALUE;
                ROCDENSITYMAT_RC_CHECK(
                    apply_single_factor<Cmplx>(h, in_ptr, out_ptr,
                                               op->space_shape,
                                               p.state_modes[f],
                                               p.operators[f]));
                in_ptr  = out_ptr;
                out_ptr = (out_ptr == buf_a) ? buf_b : buf_a;
            }

            const Cmplx* last_result = (p.operators.empty()) ? x : in_ptr;
            Cmplx alpha;
            alpha.x = static_cast<decltype(alpha.x)>(total.real());
            alpha.y = static_cast<decltype(alpha.y)>(total.imag());
            Cmplx one;
            one.x = 1; one.y = 0;
            hipLaunchKernelGGL(axpby_kernel<Cmplx>,
                               dim3(static_cast<unsigned int>(gx)),
                               dim3(bs), 0, h->stream,
                               n_elems, alpha, last_result, one, y);
            if(hipGetLastError() != hipSuccess)
                return ROCDENSITYMAT_STATUS_EXECUTION_FAILED;
        }
    }
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

template rocdensitymat_status apply_op_pure_into<hipDoubleComplex>(
    handle_impl*, rocdensitymat_operator, double, int32_t, const double*,
    const hipDoubleComplex*, hipDoubleComplex*, bool,
    hipDoubleComplex*, hipDoubleComplex*, int64_t);
template rocdensitymat_status apply_op_pure_into<hipFloatComplex>(
    handle_impl*, rocdensitymat_operator, double, int32_t, const double*,
    const hipFloatComplex*, hipFloatComplex*, bool,
    hipFloatComplex*, hipFloatComplex*, int64_t);

} // namespace rocdensitymat
