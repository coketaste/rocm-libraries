/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * `swap_index_bits` — apply a list of qubit-pair index swaps to the
 * state vector simultaneously. For each pair (x, y) in `bit_swaps`,
 * exchange the bit positions x and y of the state-vector index. With
 * an optional mask, only amplitudes matching the mask participate.
 *
 * Implementation: out-of-place gather. Per amplitude `i`, compute the
 * source index `j` by exchanging the named bit pairs in `i`, then write
 * `out[i] = sv[j]`. If `i` does not match the mask, `out[i] = sv[i]`.
 * ************************************************************************ */

#include "rocstatevec_kernels.hpp"

namespace rocstatevec
{

template <typename C>
__global__ void k_swap_bits(
    const C* in, C* out, rocstatevec_index_t N,
    const rocstatevec_index_pair_t* swaps, uint32_t n_swaps,
    rocstatevec_index_t mask_pat, rocstatevec_index_t mask_match,
    int has_mask)
{
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    if(has_mask && (i & mask_pat) != mask_match) { out[i] = in[i]; return; }

    rocstatevec_index_t j = i;
    for(uint32_t s = 0; s < n_swaps; ++s)
    {
        int32_t x = swaps[s].x;
        int32_t y = swaps[s].y;
        rocstatevec_index_t bx = (j >> x) & 1;
        rocstatevec_index_t by = (j >> y) & 1;
        if(bx != by)
        {
            j ^= (rocstatevec_index_t{1} << x);
            j ^= (rocstatevec_index_t{1} << y);
        }
    }
    out[i] = in[j];
}

template <typename C>
static rocstatevec_status swap_dispatch(
    rocstatevec_handle h_in, void* dsv, uint32_t n,
    const rocstatevec_index_pair_t* swaps, uint32_t n_swaps,
    const int32_t* mask_bit_string, const int32_t* mask_ordering, uint32_t mask_len)
{
    auto*               hh   = reinterpret_cast<handle*>(h_in);
    rocstatevec_index_t N    = state_vector_length(n);
    hipStream_t         s    = hh->stream;
    int                 tpb  = default_threads_per_block;
    int                 gpc  = static_cast<int>(ceil_div<rocstatevec_index_t>(N, tpb));

    dev_arena arena(hh, s);

    rocstatevec_index_pair_t* d_swaps = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc_and_copy(reinterpret_cast<void**>(&d_swaps),
                                              swaps,
                                              n_swaps * sizeof(rocstatevec_index_pair_t)));

    rocstatevec_index_t mask_pat = 0, mask_match = 0;
    int has_mask = (mask_len > 0) ? 1 : 0;
    for(uint32_t i = 0; i < mask_len; ++i)
    {
        rocstatevec_index_t bit = rocstatevec_index_t{1} << mask_ordering[i];
        mask_pat |= bit;
        if(mask_bit_string && mask_bit_string[i] != 0) mask_match |= bit;
    }

    void* scratch_v = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc(&scratch_v, size_t(N) * sizeof(C)));
    auto* d_scratch = reinterpret_cast<C*>(scratch_v);
    ROCSTATEVEC_HIP_CHECK(hipMemcpyAsync(d_scratch, dsv, size_t(N) * sizeof(C),
                                         hipMemcpyDeviceToDevice, s));

    hipLaunchKernelGGL(k_swap_bits<C>, dim3(gpc), dim3(tpb), 0, s,
                       d_scratch, reinterpret_cast<C*>(dsv), N,
                       d_swaps, n_swaps, mask_pat, mask_match, has_mask);
    return ROCSTATEVEC_STATUS_SUCCESS;
}

} // namespace rocstatevec

extern "C" rocstatevec_status rocstatevec_swap_index_bits(
    rocstatevec_handle h, void* state_vector, rocstatevec_data_type dtype, uint32_t n,
    const rocstatevec_index_pair_t* bit_swaps, uint32_t n_bit_swaps,
    const int32_t* mask_bit_string, const int32_t* mask_ordering, uint32_t mask_len)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(state_vector);
    if(n_bit_swaps == 0) return ROCSTATEVEC_STATUS_INVALID_VALUE;
    ROCSTATEVEC_CHECK_PTR(bit_swaps);
    if(mask_len > 0)
    {
        ROCSTATEVEC_CHECK_PTR(mask_bit_string);
        ROCSTATEVEC_CHECK_PTR(mask_ordering);
    }

    if(dtype == ROCSTATEVEC_C_64F)
        return swap_dispatch<c64>(h, state_vector, n, bit_swaps, n_bit_swaps,
                                  mask_bit_string, mask_ordering, mask_len);
    if(dtype == ROCSTATEVEC_C_32F)
        return swap_dispatch<c32>(h, state_vector, n, bit_swaps, n_bit_swaps,
                                  mask_bit_string, mask_ordering, mask_len);
    return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
}
