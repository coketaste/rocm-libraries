/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Internal HIP kernel templates shared across rocDENSITYMAT translation
 * units. The apply / axpby / scale / fill kernels live here so the apply
 * path can specialize on element type without copy-pasting kernel bodies.
 * ************************************************************************ */

#ifndef ROCDENSITYMAT_KERNELS_HPP
#define ROCDENSITYMAT_KERNELS_HPP

#include "rocdensitymat_internal.hpp"

namespace rocdensitymat
{

/*! \brief Fill a contiguous device buffer with a scalar (broadcast). */
template <typename Cmplx>
__global__ void fill_kernel(Cmplx* dst, Cmplx value, int64_t n)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(tid >= n) return;
    dst[tid] = value;
}

/*! \brief y <- alpha * x + beta * y. */
template <typename Cmplx>
__global__ void axpby_kernel(int64_t n,
                             Cmplx   alpha,
                             const Cmplx* __restrict__ x,
                             Cmplx   beta,
                             Cmplx* __restrict__ y)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(tid >= n) return;
    Cmplx xv = x[tid];
    Cmplx yv = y[tid];
    Cmplx ax;
    ax.x  = alpha.x * xv.x - alpha.y * xv.y;
    ax.y  = alpha.x * xv.y + alpha.y * xv.x;
    Cmplx by;
    by.x  = beta.x * yv.x - beta.y * yv.y;
    by.y  = beta.x * yv.y + beta.y * yv.x;
    y[tid].x = ax.x + by.x;
    y[tid].y = ax.y + by.y;
}

/*! \brief y <- alpha * x. */
template <typename Cmplx>
__global__ void scale_kernel(int64_t n, Cmplx alpha, Cmplx* __restrict__ y)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(tid >= n) return;
    Cmplx yv = y[tid];
    Cmplx out;
    out.x = alpha.x * yv.x - alpha.y * yv.y;
    out.y = alpha.x * yv.y + alpha.y * yv.x;
    y[tid] = out;
}

/*! \brief Apply a dense ``mat_dim``-by-``mat_dim`` matrix on the contiguous
 *  mode block ``[mode_lo, mode_hi)`` of a flattened state vector of length
 *  ``total_dim``. Splits the index as ``high * mat_dim * low + r * low + low_idx``.
 *
 *  The state is interpreted as a tensor with shape
 *  (high_dim, mat_dim, low_dim). For each fiber (high, low) of length
 *  ``mat_dim`` the kernel computes y[high, r, low] = sum_c mat[r, c] *
 *  x[high, c, low]. ``mat`` is assumed row-major (mat[r, c] at
 *  ``mat[r * mat_dim + c]``). */
template <typename Cmplx>
__global__ void apply_dense_block_kernel(const Cmplx* __restrict__ mat,
                                         int64_t mat_dim,
                                         const Cmplx* __restrict__ x,
                                         Cmplx*       __restrict__ y,
                                         int64_t high_dim,
                                         int64_t low_dim)
{
    int64_t low_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t high    = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    int64_t r       = static_cast<int64_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if(low_idx >= low_dim || high >= high_dim || r >= mat_dim) return;

    Cmplx acc;
    acc.x = 0;
    acc.y = 0;
    int64_t base = high * mat_dim * low_dim + low_idx;
    for(int64_t c = 0; c < mat_dim; ++c)
    {
        Cmplx xv = x[base + c * low_dim];
        Cmplx mv = mat[r * mat_dim + c];
        acc.x += mv.x * xv.x - mv.y * xv.y;
        acc.y += mv.x * xv.y + mv.y * xv.x;
    }
    y[high * mat_dim * low_dim + r * low_dim + low_idx] = acc;
}

/*! \brief Apply a length-``mat_dim`` diagonal operator on the contiguous
 *  mode block ``[mode_lo, mode_hi)`` of a flattened state vector. */
template <typename Cmplx>
__global__ void apply_diagonal_block_kernel(const Cmplx* __restrict__ diag,
                                            int64_t mat_dim,
                                            const Cmplx* __restrict__ x,
                                            Cmplx*       __restrict__ y,
                                            int64_t high_dim,
                                            int64_t low_dim)
{
    int64_t low_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t high    = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    int64_t r       = static_cast<int64_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if(low_idx >= low_dim || high >= high_dim || r >= mat_dim) return;

    int64_t idx = high * mat_dim * low_dim + r * low_dim + low_idx;
    Cmplx   xv  = x[idx];
    Cmplx   d   = diag[r];
    y[idx].x    = d.x * xv.x - d.y * xv.y;
    y[idx].y    = d.x * xv.y + d.y * xv.x;
}

/*! \brief Compute conj(a) dot b accumulated to host memory. Pure-state
 *  overlap is hipBLAS-backed in production builds; this kernel covers
 *  smoke tests where rocBLAS is not yet wired through CMake. */
template <typename Cmplx>
__global__ void
conj_dot_partial_kernel(const Cmplx* __restrict__ a,
                        const Cmplx* __restrict__ b,
                        Cmplx*       __restrict__ partials,
                        int64_t      n)
{
    __shared__ double s_re[default_threads_per_block];
    __shared__ double s_im[default_threads_per_block];
    int64_t  tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int      lid = threadIdx.x;
    double   re = 0;
    double   im = 0;
    if(tid < n)
    {
        Cmplx av = a[tid];
        Cmplx bv = b[tid];
        // conj(a) * b = (a.x - i a.y) * (b.x + i b.y)
        re = av.x * bv.x + av.y * bv.y;
        im = av.x * bv.y - av.y * bv.x;
    }
    s_re[lid] = re;
    s_im[lid] = im;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(lid < s)
        {
            s_re[lid] += s_re[lid + s];
            s_im[lid] += s_im[lid + s];
        }
        __syncthreads();
    }
    if(lid == 0)
    {
        partials[blockIdx.x].x = s_re[0];
        partials[blockIdx.x].y = s_im[0];
    }
}

} // namespace rocdensitymat

#endif /* ROCDENSITYMAT_KERNELS_HPP */
