/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Mixed-state operator-action implementation. The Liouvillian
 *   L[rho] = -i [H, rho] = -i (H rho - rho H)
 * is decomposed into two pure-side applies. The density matrix is stored
 * column-major as a (D x D) tensor flattened to length D^2; the row-side
 * apply is exactly the pure-state apply on each column of rho, and the
 * column-side apply uses (rho H)^T = H^T rho^T which we compute by
 * transposing rho, applying the operator column-wise, then transposing
 * back.
 * ************************************************************************ */

#include "rocdensitymat_descriptors.hpp"
#include "rocdensitymat_kernels.hpp"

#include <complex>

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
__global__ void transpose_dxd_kernel(const Cmplx* __restrict__ in,
                                     Cmplx*       __restrict__ out,
                                     int64_t      D)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t j = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= D || j >= D) return;
    out[j * D + i] = in[i * D + j];
}

template <typename Cmplx>
__global__ void sub_axpy_kernel(int64_t n,
                                Cmplx alpha,
                                const Cmplx* __restrict__ x,
                                Cmplx* __restrict__ y)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(tid >= n) return;
    Cmplx xv = x[tid];
    Cmplx ax;
    ax.x = alpha.x * xv.x - alpha.y * xv.y;
    ax.y = alpha.x * xv.y + alpha.y * xv.x;
    y[tid].x -= ax.x;
    y[tid].y -= ax.y;
}

} // namespace

/*! \brief Per-state scratch requirement of apply_op_mixed_into.
 *  Two D-sized ping-pong buffers (used for each pure-side apply) plus
 *  two D^2-sized buffers (for the transposed copy and for the row/column
 *  side accumulator before subtraction).
 */
size_t apply_op_mixed_scratch_bytes(rocdensitymat_state state)
{
    int64_t D = state->hilbert_dim();
    size_t  e = element_size_bytes(state->data_type);
    return 2 * static_cast<size_t>(D) * e
         + 2 * static_cast<size_t>(D) * static_cast<size_t>(D) * e;
}

/*! \brief Compute rho_out = -i [H, rho_in]. Caller supplies two D-sized
 *  ping-pong buffers (buf_a / buf_b) and two D*D-sized buffers
 *  (rho_T / col_acc). */
template <typename Cmplx>
rocdensitymat_status
apply_op_mixed_into(handle_impl* h,
                    rocdensitymat_operator op,
                    double t,
                    int32_t num_params,
                    const double* params,
                    const Cmplx* rho_in,
                    Cmplx* rho_out,
                    Cmplx* buf_a,
                    Cmplx* buf_b,
                    Cmplx* rho_T,
                    Cmplx* col_acc,
                    int64_t D)
{
    int64_t N  = D * D;
    int     bs = default_threads_per_block;
    int64_t gx = ceil_div<int64_t>(N, bs);
    int64_t gd = ceil_div<int64_t>(D, bs);

    Cmplx zero; zero.x = 0; zero.y = 0;
    hipLaunchKernelGGL(fill_kernel<Cmplx>,
                       dim3(static_cast<unsigned int>(gx)),
                       dim3(bs), 0, h->stream, rho_out, zero, N);
    if(hipGetLastError() != hipSuccess)
        return ROCDENSITYMAT_STATUS_EXECUTION_FAILED;

    // Row side: rho_out[:, j] += H * rho_in[:, j] for each j.
    for(int64_t j = 0; j < D; ++j)
    {
        const Cmplx* col_in  = rho_in  + j * D;
        Cmplx*       col_out = rho_out + j * D;
        ROCDENSITYMAT_RC_CHECK(
            apply_op_pure_into<Cmplx>(h, op, t, num_params, params,
                                      col_in, col_out, /*accumulate*/true,
                                      buf_a, buf_b, D));
    }

    // Column side: H rho^T columnwise → rho^T H stored as (rho H)^T.
    {
        dim3 block(16, 16);
        dim3 grid(static_cast<unsigned int>(ceil_div<int64_t>(D, 16)),
                  static_cast<unsigned int>(ceil_div<int64_t>(D, 16)));
        hipLaunchKernelGGL(transpose_dxd_kernel<Cmplx>,
                           grid, block, 0, h->stream, rho_in, rho_T, D);
        if(hipGetLastError() != hipSuccess)
            return ROCDENSITYMAT_STATUS_EXECUTION_FAILED;
    }
    hipLaunchKernelGGL(fill_kernel<Cmplx>,
                       dim3(static_cast<unsigned int>(gx)),
                       dim3(bs), 0, h->stream, col_acc, zero, N);
    if(hipGetLastError() != hipSuccess)
        return ROCDENSITYMAT_STATUS_EXECUTION_FAILED;

    for(int64_t j = 0; j < D; ++j)
    {
        const Cmplx* col_in  = rho_T   + j * D;
        Cmplx*       col_out = col_acc + j * D;
        ROCDENSITYMAT_RC_CHECK(
            apply_op_pure_into<Cmplx>(h, op, t, num_params, params,
                                      col_in, col_out, /*accumulate*/true,
                                      buf_a, buf_b, D));
    }

    // col_acc holds H * rho^T = (rho H)^T; transpose back into rho_T then
    // subtract from rho_out.
    {
        dim3 block(16, 16);
        dim3 grid(static_cast<unsigned int>(ceil_div<int64_t>(D, 16)),
                  static_cast<unsigned int>(ceil_div<int64_t>(D, 16)));
        hipLaunchKernelGGL(transpose_dxd_kernel<Cmplx>,
                           grid, block, 0, h->stream, col_acc, rho_T, D);
        if(hipGetLastError() != hipSuccess)
            return ROCDENSITYMAT_STATUS_EXECUTION_FAILED;
    }
    {
        Cmplx one; one.x = 1; one.y = 0;
        hipLaunchKernelGGL(sub_axpy_kernel<Cmplx>,
                           dim3(static_cast<unsigned int>(gx)),
                           dim3(bs), 0, h->stream,
                           N, one, rho_T, rho_out);
        if(hipGetLastError() != hipSuccess)
            return ROCDENSITYMAT_STATUS_EXECUTION_FAILED;
    }
    {
        Cmplx neg_i; neg_i.x = 0; neg_i.y = -1;
        hipLaunchKernelGGL(scale_kernel<Cmplx>,
                           dim3(static_cast<unsigned int>(gx)),
                           dim3(bs), 0, h->stream,
                           N, neg_i, rho_out);
        if(hipGetLastError() != hipSuccess)
            return ROCDENSITYMAT_STATUS_EXECUTION_FAILED;
    }
    (void)gd;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

template rocdensitymat_status apply_op_mixed_into<hipDoubleComplex>(
    handle_impl*, rocdensitymat_operator, double, int32_t, const double*,
    const hipDoubleComplex*, hipDoubleComplex*,
    hipDoubleComplex*, hipDoubleComplex*,
    hipDoubleComplex*, hipDoubleComplex*, int64_t);
template rocdensitymat_status apply_op_mixed_into<hipFloatComplex>(
    handle_impl*, rocdensitymat_operator, double, int32_t, const double*,
    const hipFloatComplex*, hipFloatComplex*,
    hipFloatComplex*, hipFloatComplex*,
    hipFloatComplex*, hipFloatComplex*, int64_t);

} // namespace rocdensitymat
