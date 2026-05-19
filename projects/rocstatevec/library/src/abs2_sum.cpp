/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Abs2-sum probability reductions:
 *   * `abs2_sum_on_z_basis` — splits |amp|^2 into parity-0 / parity-1 of
 *     the basis-bit subset.
 *   * `abs2_sum_array`      — generalizes that into an arbitrary
 *     bit-ordering with optional masked-bit projection.
 * ************************************************************************ */

#include "rocstatevec_kernels.hpp"

namespace rocstatevec
{

template <typename C>
__global__ void k_abs2sum_z_basis(const C* sv, rocstatevec_index_t N,
                                  rocstatevec_index_t mask,
                                  double* s0, double* s1)
{
    using T = complex_traits<C>;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    double v = static_cast<double>(T::abs2(sv[i]));
    int    p = parity64(i & mask);
    if(p == 0) atomicAdd(s0, v);
    else       atomicAdd(s1, v);
}

template <typename C>
static rocstatevec_status z_basis_dispatch(rocstatevec_handle h_in, const void* dsv, uint32_t n,
                                           double* s0, double* s1, rocstatevec_index_t mask)
{
    auto*               hh   = reinterpret_cast<handle*>(h_in);
    rocstatevec_index_t N    = state_vector_length(n);
    hipStream_t         s    = hh->stream;
    int                 tpb  = default_threads_per_block;
    int                 gpc  = static_cast<int>(ceil_div<rocstatevec_index_t>(N, tpb));

    dev_arena arena(hh, s);
    void*     d_acc_v = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc(&d_acc_v, 2 * sizeof(double)));
    auto* d_acc = reinterpret_cast<double*>(d_acc_v);
    ROCSTATEVEC_HIP_CHECK(hipMemsetAsync(d_acc, 0, 2 * sizeof(double), s));

    hipLaunchKernelGGL(k_abs2sum_z_basis<C>, dim3(gpc), dim3(tpb), 0, s,
                       reinterpret_cast<const C*>(dsv), N, mask, d_acc, d_acc + 1);

    double host[2] = {0, 0};
    ROCSTATEVEC_HIP_CHECK(hipMemcpyAsync(host, d_acc, 2 * sizeof(double),
                                         hipMemcpyDeviceToHost, s));
    ROCSTATEVEC_HIP_CHECK(hipStreamSynchronize(s));

    *s0 = host[0];
    *s1 = host[1];
    return ROCSTATEVEC_STATUS_SUCCESS;
}

template <typename C>
__global__ void k_abs2sum_array(const C* sv, rocstatevec_index_t N,
                                const int32_t* bit_ordering, uint32_t bo_len,
                                rocstatevec_index_t mask_pat, rocstatevec_index_t mask_match,
                                double* out)
{
    using T = complex_traits<C>;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    if((i & mask_pat) != mask_match) return;
    rocstatevec_index_t key = 0;
    for(uint32_t b = 0; b < bo_len; ++b)
        key |= (((i >> bit_ordering[b]) & 1) << b);
    atomicAdd(out + key, static_cast<double>(T::abs2(sv[i])));
}

template <typename C>
static rocstatevec_status array_dispatch(rocstatevec_handle h_in, const void* dsv, uint32_t n,
                                         double* out, const int32_t* bit_ordering, uint32_t bo_len,
                                         const int32_t* mask_bit_string,
                                         const int32_t* mask_ordering, uint32_t mask_len)
{
    auto*               hh    = reinterpret_cast<handle*>(h_in);
    rocstatevec_index_t N      = state_vector_length(n);
    rocstatevec_index_t out_n  = rocstatevec_index_t{1} << bo_len;
    hipStream_t         s      = hh->stream;
    int                 tpb    = default_threads_per_block;
    int                 gpc    = static_cast<int>(ceil_div<rocstatevec_index_t>(N, tpb));

    rocstatevec_index_t mask_pat = 0, mask_match = 0;
    for(uint32_t i = 0; i < mask_len; ++i)
    {
        rocstatevec_index_t bit = rocstatevec_index_t{1} << mask_ordering[i];
        mask_pat   |= bit;
        if(mask_bit_string && mask_bit_string[i] != 0) mask_match |= bit;
    }

    dev_arena arena(hh, s);

    int32_t* d_bo = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc_and_copy(reinterpret_cast<void**>(&d_bo),
                                              bit_ordering, bo_len * sizeof(int32_t)));

    void* d_out_v = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc(&d_out_v, out_n * sizeof(double)));
    auto* d_out = reinterpret_cast<double*>(d_out_v);
    ROCSTATEVEC_HIP_CHECK(hipMemsetAsync(d_out, 0, out_n * sizeof(double), s));

    hipLaunchKernelGGL(k_abs2sum_array<C>, dim3(gpc), dim3(tpb), 0, s,
                       reinterpret_cast<const C*>(dsv), N, d_bo, bo_len,
                       mask_pat, mask_match, d_out);

    ROCSTATEVEC_HIP_CHECK(hipMemcpyAsync(out, d_out, out_n * sizeof(double),
                                         hipMemcpyDeviceToHost, s));
    ROCSTATEVEC_HIP_CHECK(hipStreamSynchronize(s));
    return ROCSTATEVEC_STATUS_SUCCESS;
}

} // namespace rocstatevec

extern "C" rocstatevec_status rocstatevec_abs2_sum_on_z_basis(
    rocstatevec_handle h, const void* dsv, rocstatevec_data_type dtype, uint32_t n,
    double* s0, double* s1, const int32_t* basis_bits, uint32_t n_basis_bits)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(dsv); ROCSTATEVEC_CHECK_PTR(s0); ROCSTATEVEC_CHECK_PTR(s1);
    if(n_basis_bits == 0) { *s0 = 0; *s1 = 0; return ROCSTATEVEC_STATUS_INVALID_VALUE; }
    ROCSTATEVEC_CHECK_PTR(basis_bits);

    rocstatevec_index_t mask = qubit_mask(basis_bits, n_basis_bits);

    if(dtype == ROCSTATEVEC_C_64F) return z_basis_dispatch<c64>(h, dsv, n, s0, s1, mask);
    if(dtype == ROCSTATEVEC_C_32F) return z_basis_dispatch<c32>(h, dsv, n, s0, s1, mask);
    return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
}

extern "C" rocstatevec_status rocstatevec_abs2_sum_array(
    rocstatevec_handle h, const void* dsv, rocstatevec_data_type dtype, uint32_t n,
    double* out, const int32_t* bit_ordering, uint32_t bo_len,
    const int32_t* mask_bit_string, const int32_t* mask_ordering, uint32_t mask_len)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(dsv); ROCSTATEVEC_CHECK_PTR(out);
    if(bo_len == 0)             return ROCSTATEVEC_STATUS_INVALID_VALUE;
    ROCSTATEVEC_CHECK_PTR(bit_ordering);
    if(mask_len > 0)
    {
        ROCSTATEVEC_CHECK_PTR(mask_bit_string);
        ROCSTATEVEC_CHECK_PTR(mask_ordering);
    }

    if(dtype == ROCSTATEVEC_C_64F)
        return array_dispatch<c64>(h, dsv, n, out, bit_ordering, bo_len,
                                   mask_bit_string, mask_ordering, mask_len);
    if(dtype == ROCSTATEVEC_C_32F)
        return array_dispatch<c32>(h, dsv, n, out, bit_ordering, bo_len,
                                   mask_bit_string, mask_ordering, mask_len);
    return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
}
