/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Accessor API.
 *
 * An accessor exposes a "logical" sub-view of a state vector ordered by
 * `bit_ordering`, optionally constrained by a mask. Logical-index `j`
 * (0 <= j < 2^bit_ordering_len) maps to a physical index by:
 *
 *     phys(j) = mask_match
 *             | sum over b in [0, bit_ordering_len) of
 *                 (((j >> b) & 1) << bit_ordering[b])
 *
 * `accessor_get` copies amplitudes [begin, end) of the logical window
 * into the user host buffer. `accessor_set` does the opposite. Both run
 * on the bound stream and synchronize before returning so the host
 * buffer is valid on exit.
 * ************************************************************************ */

#include "rocstatevec_kernels.hpp"

#include <vector>

namespace rocstatevec
{

struct accessor_descriptor
{
    void*                 sv          = nullptr;
    bool                  read_only    = false;
    rocstatevec_data_type dtype       = ROCSTATEVEC_C_64F;
    uint32_t              n_index_bits = 0;

    std::vector<int32_t>  bit_ordering;
    rocstatevec_index_t   mask_match  = 0;
    rocstatevec_index_t   mask_pat    = 0;
};

template <typename C>
__global__ void k_accessor_get(
    const C* sv, C* out, rocstatevec_index_t begin, rocstatevec_index_t count,
    const int32_t* bit_ordering, uint32_t bo_len, rocstatevec_index_t mask_match)
{
    rocstatevec_index_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k >= count) return;
    rocstatevec_index_t j   = begin + k;
    rocstatevec_index_t idx = mask_match;
    for(uint32_t b = 0; b < bo_len; ++b)
        if(((j >> b) & 1) != 0) idx |= (rocstatevec_index_t{1} << bit_ordering[b]);
    out[k] = sv[idx];
}

template <typename C>
__global__ void k_accessor_set(
    C* sv, const C* in, rocstatevec_index_t begin, rocstatevec_index_t count,
    const int32_t* bit_ordering, uint32_t bo_len, rocstatevec_index_t mask_match)
{
    rocstatevec_index_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k >= count) return;
    rocstatevec_index_t j   = begin + k;
    rocstatevec_index_t idx = mask_match;
    for(uint32_t b = 0; b < bo_len; ++b)
        if(((j >> b) & 1) != 0) idx |= (rocstatevec_index_t{1} << bit_ordering[b]);
    sv[idx] = in[k];
}

template <typename C>
static rocstatevec_status do_get(rocstatevec_handle h_in, accessor_descriptor& a,
                                  void* host_buf, rocstatevec_index_t begin, rocstatevec_index_t end)
{
    rocstatevec_index_t count = end - begin;
    if(count <= 0) return ROCSTATEVEC_STATUS_INVALID_VALUE;
    auto*       hh  = reinterpret_cast<handle*>(h_in);
    hipStream_t s   = hh->stream;
    int         tpb = default_threads_per_block;
    int         gpc = static_cast<int>(ceil_div<rocstatevec_index_t>(count, tpb));

    dev_arena arena(hh, s);

    int32_t* d_bo = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc_and_copy(reinterpret_cast<void**>(&d_bo),
                                              a.bit_ordering.data(),
                                              a.bit_ordering.size() * sizeof(int32_t)));
    void* d_buf_v = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc(&d_buf_v, size_t(count) * sizeof(C)));
    auto* d_buf = reinterpret_cast<C*>(d_buf_v);

    hipLaunchKernelGGL(k_accessor_get<C>, dim3(gpc), dim3(tpb), 0, s,
                       reinterpret_cast<const C*>(a.sv), d_buf, begin, count,
                       d_bo, static_cast<uint32_t>(a.bit_ordering.size()), a.mask_match);

    ROCSTATEVEC_HIP_CHECK(hipMemcpyAsync(host_buf, d_buf, size_t(count) * sizeof(C),
                                         hipMemcpyDeviceToHost, s));
    ROCSTATEVEC_HIP_CHECK(hipStreamSynchronize(s));
    return ROCSTATEVEC_STATUS_SUCCESS;
}

template <typename C>
static rocstatevec_status do_set(rocstatevec_handle h_in, accessor_descriptor& a,
                                  const void* host_buf, rocstatevec_index_t begin, rocstatevec_index_t end)
{
    if(a.read_only) return ROCSTATEVEC_STATUS_INVALID_VALUE;
    rocstatevec_index_t count = end - begin;
    if(count <= 0) return ROCSTATEVEC_STATUS_INVALID_VALUE;
    auto*       hh  = reinterpret_cast<handle*>(h_in);
    hipStream_t s   = hh->stream;
    int         tpb = default_threads_per_block;
    int         gpc = static_cast<int>(ceil_div<rocstatevec_index_t>(count, tpb));

    dev_arena arena(hh, s);

    int32_t* d_bo = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc_and_copy(reinterpret_cast<void**>(&d_bo),
                                              a.bit_ordering.data(),
                                              a.bit_ordering.size() * sizeof(int32_t)));
    void* d_buf_v = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc_and_copy(&d_buf_v, host_buf,
                                              size_t(count) * sizeof(C)));
    auto* d_buf = reinterpret_cast<C*>(d_buf_v);

    hipLaunchKernelGGL(k_accessor_set<C>, dim3(gpc), dim3(tpb), 0, s,
                       reinterpret_cast<C*>(a.sv), d_buf, begin, count,
                       d_bo, static_cast<uint32_t>(a.bit_ordering.size()), a.mask_match);
    ROCSTATEVEC_HIP_CHECK(hipStreamSynchronize(s));
    return ROCSTATEVEC_STATUS_SUCCESS;
}

static rocstatevec_status init_descriptor(
    accessor_descriptor& a, void* sv, rocstatevec_data_type dtype, uint32_t n,
    const int32_t* bit_ordering, uint32_t bo_len,
    const int32_t* mask_bit_string, const int32_t* mask_ordering, uint32_t mask_len)
{
    a.sv           = sv;
    a.dtype        = dtype;
    a.n_index_bits = n;
    a.bit_ordering.assign(bit_ordering, bit_ordering + bo_len);
    a.mask_pat   = 0;
    a.mask_match = 0;
    for(uint32_t i = 0; i < mask_len; ++i)
    {
        rocstatevec_index_t bit = rocstatevec_index_t{1} << mask_ordering[i];
        a.mask_pat |= bit;
        if(mask_bit_string && mask_bit_string[i] != 0) a.mask_match |= bit;
    }
    return ROCSTATEVEC_STATUS_SUCCESS;
}

} // namespace rocstatevec

extern "C" rocstatevec_status rocstatevec_accessor_create(
    rocstatevec_handle h, void* sv, rocstatevec_data_type dtype, uint32_t n,
    rocstatevec_accessor_descriptor* out_acc, const int32_t* bit_ordering, uint32_t bo_len,
    const int32_t* mask_bit_string, const int32_t* mask_ordering, uint32_t mask_len,
    size_t* extra_workspace_size_in_bytes)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(sv); ROCSTATEVEC_CHECK_PTR(out_acc); ROCSTATEVEC_CHECK_PTR(bit_ordering);
    auto* a = new(std::nothrow) accessor_descriptor{};
    if(!a) return ROCSTATEVEC_STATUS_ALLOC_FAILED;
    a->read_only = false;
    init_descriptor(*a, sv, dtype, n, bit_ordering, bo_len,
                    mask_bit_string, mask_ordering, mask_len);
    *out_acc = reinterpret_cast<rocstatevec_accessor_descriptor>(a);
    if(extra_workspace_size_in_bytes) *extra_workspace_size_in_bytes = 0;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_accessor_create_view(
    rocstatevec_handle h, const void* sv, rocstatevec_data_type dtype, uint32_t n,
    rocstatevec_accessor_descriptor* out_acc, const int32_t* bit_ordering, uint32_t bo_len,
    const int32_t* mask_bit_string, const int32_t* mask_ordering, uint32_t mask_len,
    size_t* extra_workspace_size_in_bytes)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(sv); ROCSTATEVEC_CHECK_PTR(out_acc); ROCSTATEVEC_CHECK_PTR(bit_ordering);
    auto* a = new(std::nothrow) accessor_descriptor{};
    if(!a) return ROCSTATEVEC_STATUS_ALLOC_FAILED;
    a->read_only = true;
    init_descriptor(*a, const_cast<void*>(sv), dtype, n, bit_ordering, bo_len,
                    mask_bit_string, mask_ordering, mask_len);
    *out_acc = reinterpret_cast<rocstatevec_accessor_descriptor>(a);
    if(extra_workspace_size_in_bytes) *extra_workspace_size_in_bytes = 0;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_accessor_destroy(rocstatevec_accessor_descriptor a)
{
    using namespace rocstatevec;
    delete reinterpret_cast<accessor_descriptor*>(a);
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_accessor_set_extra_workspace(
    rocstatevec_handle, rocstatevec_accessor_descriptor, void*, size_t)
{
    // Workspace size returned by create is 0; the call is a no-op.
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_accessor_get(
    rocstatevec_handle h, rocstatevec_accessor_descriptor acc, void* buf,
    rocstatevec_index_t begin, rocstatevec_index_t end)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h); ROCSTATEVEC_CHECK_PTR(acc); ROCSTATEVEC_CHECK_PTR(buf);
    auto& a = *reinterpret_cast<accessor_descriptor*>(acc);
    if(a.dtype == ROCSTATEVEC_C_64F) return do_get<c64>(h, a, buf, begin, end);
    if(a.dtype == ROCSTATEVEC_C_32F) return do_get<c32>(h, a, buf, begin, end);
    return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
}

extern "C" rocstatevec_status rocstatevec_accessor_set(
    rocstatevec_handle h, rocstatevec_accessor_descriptor acc, const void* buf,
    rocstatevec_index_t begin, rocstatevec_index_t end)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h); ROCSTATEVEC_CHECK_PTR(acc); ROCSTATEVEC_CHECK_PTR(buf);
    auto& a = *reinterpret_cast<accessor_descriptor*>(acc);
    if(a.dtype == ROCSTATEVEC_C_64F) return do_set<c64>(h, a, buf, begin, end);
    if(a.dtype == ROCSTATEVEC_C_32F) return do_set<c32>(h, a, buf, begin, end);
    return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
}
