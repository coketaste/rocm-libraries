/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * State-vector collapse onto a Z-basis parity outcome or onto a
 * specific bit-string outcome on a chosen ordering.
 * ************************************************************************ */

#include "rocstatevec_kernels.hpp"

#include <cmath>

namespace rocstatevec
{

template <typename C>
__global__ void k_collapse_z_basis(C* sv, rocstatevec_index_t N, rocstatevec_index_t mask,
                                   int32_t parity_keep,
                                   typename complex_traits<C>::real_t inv_sqrt_norm)
{
    using T = complex_traits<C>;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int p = parity64(i & mask);
    if(p == parity_keep) sv[i] = T::scale(sv[i], inv_sqrt_norm);
    else                 sv[i] = T::zero();
}

template <typename C>
__global__ void k_collapse_bit_string(C* sv, rocstatevec_index_t N,
                                      rocstatevec_index_t mask_pat, rocstatevec_index_t mask_match,
                                      typename complex_traits<C>::real_t inv_sqrt_norm)
{
    using T = complex_traits<C>;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    if((i & mask_pat) == mask_match) sv[i] = T::scale(sv[i], inv_sqrt_norm);
    else                             sv[i] = T::zero();
}

template <typename C>
static rocstatevec_status z_collapse_dispatch(rocstatevec_handle h, void* dsv, uint32_t n,
                                              rocstatevec_index_t mask, int32_t parity_keep,
                                              double norm)
{
    using TR = typename complex_traits<C>::real_t;
    rocstatevec_index_t N    = state_vector_length(n);
    hipStream_t         s    = handle_stream(h);
    int                 tpb  = default_threads_per_block;
    int                 gpc  = static_cast<int>(ceil_div<rocstatevec_index_t>(N, tpb));
    TR  inv_sqrt = static_cast<TR>(1.0 / std::sqrt(norm));
    hipLaunchKernelGGL(k_collapse_z_basis<C>, dim3(gpc), dim3(tpb), 0, s,
                       reinterpret_cast<C*>(dsv), N, mask, parity_keep, inv_sqrt);
    return ROCSTATEVEC_STATUS_SUCCESS;
}

template <typename C>
static rocstatevec_status bs_collapse_dispatch(rocstatevec_handle h, void* dsv, uint32_t n,
                                               rocstatevec_index_t mask_pat,
                                               rocstatevec_index_t mask_match,
                                               double norm)
{
    using TR = typename complex_traits<C>::real_t;
    rocstatevec_index_t N    = state_vector_length(n);
    hipStream_t         s    = handle_stream(h);
    int                 tpb  = default_threads_per_block;
    int                 gpc  = static_cast<int>(ceil_div<rocstatevec_index_t>(N, tpb));
    TR  inv_sqrt = static_cast<TR>(1.0 / std::sqrt(norm));
    hipLaunchKernelGGL(k_collapse_bit_string<C>, dim3(gpc), dim3(tpb), 0, s,
                       reinterpret_cast<C*>(dsv), N, mask_pat, mask_match, inv_sqrt);
    return ROCSTATEVEC_STATUS_SUCCESS;
}

} // namespace rocstatevec

extern "C" rocstatevec_status rocstatevec_collapse_on_z_basis(
    rocstatevec_handle h, void* dsv, rocstatevec_data_type dtype, uint32_t n,
    int32_t parity_keep, const int32_t* basis_bits, uint32_t n_basis_bits, double norm)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(dsv);
    if(n_basis_bits == 0)               return ROCSTATEVEC_STATUS_INVALID_VALUE;
    ROCSTATEVEC_CHECK_PTR(basis_bits);
    if(parity_keep != 0 && parity_keep != 1) return ROCSTATEVEC_STATUS_INVALID_VALUE;
    if(norm <= 0.0)                     return ROCSTATEVEC_STATUS_INVALID_VALUE;

    rocstatevec_index_t mask = qubit_mask(basis_bits, n_basis_bits);
    if(dtype == ROCSTATEVEC_C_64F) return z_collapse_dispatch<c64>(h, dsv, n, mask, parity_keep, norm);
    if(dtype == ROCSTATEVEC_C_32F) return z_collapse_dispatch<c32>(h, dsv, n, mask, parity_keep, norm);
    return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
}

extern "C" rocstatevec_status rocstatevec_collapse_by_bit_string(
    rocstatevec_handle h, void* dsv, rocstatevec_data_type dtype, uint32_t n,
    const int32_t* bit_string, const int32_t* bit_ordering, uint32_t bs_len, double norm)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(dsv);
    if(bs_len == 0)                  return ROCSTATEVEC_STATUS_INVALID_VALUE;
    ROCSTATEVEC_CHECK_PTR(bit_string);
    ROCSTATEVEC_CHECK_PTR(bit_ordering);
    if(norm <= 0.0)                  return ROCSTATEVEC_STATUS_INVALID_VALUE;

    rocstatevec_index_t mask_pat = 0, mask_match = 0;
    for(uint32_t i = 0; i < bs_len; ++i)
    {
        rocstatevec_index_t bit = rocstatevec_index_t{1} << bit_ordering[i];
        mask_pat |= bit;
        if(bit_string[i] != 0) mask_match |= bit;
    }
    if(dtype == ROCSTATEVEC_C_64F) return bs_collapse_dispatch<c64>(h, dsv, n, mask_pat, mask_match, norm);
    if(dtype == ROCSTATEVEC_C_32F) return bs_collapse_dispatch<c32>(h, dsv, n, mask_pat, mask_match, norm);
    return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
}
