/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * `apply_generalized_permutation_matrix` — combination of a permutation
 * on the m-dimensional target sub-space and a diagonal scaling.
 *
 * Forward (adjoint=0): out[base | bits(perm[t])] = diag[perm[t]] * sv[base | bits(t)]
 * Adjoint  (adjoint=1): out[base | bits(t)] = conj(diag[perm[t]]) * sv[base | bits(perm[t])]
 *
 * Either `permutation` or `diagonals` may be NULL: a NULL permutation
 * means identity, a NULL diagonal means all-ones. Operates out-of-place
 * via an internal scratch buffer reserved through the workspace arena.
 *
 * Both `permutation` and `diagonals` are auto-detected as host or device
 * pointers; `is_device_pointer` decides whether to copy or to reuse.
 * ************************************************************************ */

#include "rocstatevec_kernels.hpp"

namespace rocstatevec
{

template <typename C>
__global__ void k_gen_perm(
    const C* in, C* out, rocstatevec_index_t N,
    const rocstatevec_index_t* perm, int has_perm,
    const C* diag, int has_diag, int adjoint,
    uint32_t n_targets,
    const int32_t* targets, rocstatevec_index_t target_mask,
    rocstatevec_index_t control_mask, rocstatevec_index_t control_match)
{
    using T = complex_traits<C>;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    if((i & control_mask) != control_match) { out[i] = in[i]; return; }

    uint32_t t = 0;
    rocstatevec_index_t base = i & ~target_mask;
    for(uint32_t b = 0; b < n_targets; ++b)
        if(((i >> targets[b]) & 1) != 0) t |= (1u << b);

    uint32_t           perm_t   = has_perm ? static_cast<uint32_t>(perm[t]) : t;
    rocstatevec_index_t src_idx = base;
    rocstatevec_index_t dst_idx = base;

    if(adjoint)
    {
        for(uint32_t b = 0; b < n_targets; ++b)
            if(((perm_t >> b) & 1) != 0) src_idx |= (rocstatevec_index_t{1} << targets[b]);
        for(uint32_t b = 0; b < n_targets; ++b)
            if(((t >> b) & 1) != 0)      dst_idx |= (rocstatevec_index_t{1} << targets[b]);
    }
    else
    {
        for(uint32_t b = 0; b < n_targets; ++b)
            if(((t >> b) & 1) != 0)      src_idx |= (rocstatevec_index_t{1} << targets[b]);
        for(uint32_t b = 0; b < n_targets; ++b)
            if(((perm_t >> b) & 1) != 0) dst_idx |= (rocstatevec_index_t{1} << targets[b]);
    }
    if(dst_idx != i) return;

    C amp = in[src_idx];
    if(has_diag)
    {
        C d = diag[perm_t];
        if(adjoint) d = T::conj(d);
        amp = T::mul(d, amp);
    }
    out[i] = amp;
}

template <typename C>
static rocstatevec_status gen_perm_dispatch(
    rocstatevec_handle h_in, void* dsv, uint32_t n,
    const rocstatevec_index_t* permutation,
    const void* diagonals, int has_diag, int adjoint,
    const int32_t* targets, uint32_t n_targets,
    const int32_t* controls, const int32_t* control_bit_values, uint32_t n_controls,
    void* user_ws, size_t user_ws_bytes)
{
    auto*               hh   = reinterpret_cast<handle*>(h_in);
    rocstatevec_index_t N    = state_vector_length(n);
    hipStream_t         s    = hh->stream;
    int                 tpb  = default_threads_per_block;
    int                 gpc  = static_cast<int>(ceil_div<rocstatevec_index_t>(N, tpb));
    uint32_t            m    = 1u << n_targets;

    rocstatevec_index_t target_mask = qubit_mask(targets, n_targets);
    rocstatevec_index_t control_mask = qubit_mask(controls, n_controls);
    rocstatevec_index_t control_match = 0;
    for(uint32_t i = 0; i < n_controls; ++i)
    {
        int v = control_bit_values ? control_bit_values[i] : 1;
        if(v != 0) control_match |= (rocstatevec_index_t{1} << controls[i]);
    }

    dev_arena arena(hh, s, user_ws, user_ws_bytes);

    int32_t* d_targets = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc_and_copy(reinterpret_cast<void**>(&d_targets),
                                              targets, n_targets * sizeof(int32_t)));

    const rocstatevec_index_t* d_perm = nullptr;
    int has_perm = (permutation != nullptr) ? 1 : 0;
    if(has_perm)
    {
        if(is_device_pointer(permutation))
        {
            d_perm = permutation;
        }
        else
        {
            void* tmp = nullptr;
            ROCSTATEVEC_RC_CHECK(arena.alloc_and_copy(&tmp, permutation,
                                                      m * sizeof(rocstatevec_index_t)));
            d_perm = reinterpret_cast<rocstatevec_index_t*>(tmp);
        }
    }

    const C* d_diag = nullptr;
    if(has_diag)
    {
        if(is_device_pointer(diagonals))
        {
            d_diag = reinterpret_cast<const C*>(diagonals);
        }
        else
        {
            void* tmp = nullptr;
            ROCSTATEVEC_RC_CHECK(arena.alloc_and_copy(&tmp, diagonals,
                                                      m * sizeof(C)));
            d_diag = reinterpret_cast<const C*>(tmp);
        }
    }

    void* scratch_v = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc(&scratch_v, size_t(N) * sizeof(C)));
    auto* d_scratch = reinterpret_cast<C*>(scratch_v);
    ROCSTATEVEC_HIP_CHECK(hipMemcpyAsync(d_scratch, dsv, size_t(N) * sizeof(C),
                                         hipMemcpyDeviceToDevice, s));

    hipLaunchKernelGGL(k_gen_perm<C>, dim3(gpc), dim3(tpb), 0, s,
                       d_scratch, reinterpret_cast<C*>(dsv), N,
                       d_perm, has_perm, d_diag, has_diag, adjoint,
                       n_targets, d_targets, target_mask, control_mask, control_match);
    return ROCSTATEVEC_STATUS_SUCCESS;
}

static size_t gen_perm_workspace_bytes(rocstatevec_data_type sv_dtype,
                                       uint32_t              n,
                                       uint32_t              n_targets,
                                       int                   has_diag,
                                       int                   has_perm)
{
    size_t es        = element_size_bytes(sv_dtype);
    size_t targets_b = align_up(size_t(n_targets) * sizeof(int32_t),
                                default_workspace_align);
    size_t m         = size_t{1} << n_targets;
    size_t perm_b    = has_perm
                         ? align_up(m * sizeof(rocstatevec_index_t), default_workspace_align)
                         : 0;
    size_t diag_b    = has_diag
                         ? align_up(m * es, default_workspace_align)
                         : 0;
    size_t sv_n      = size_t{1} << n;
    size_t scratch_b = align_up(sv_n * es, default_workspace_align);
    return targets_b + perm_b + diag_b + scratch_b;
}

} // namespace rocstatevec

extern "C" rocstatevec_status rocstatevec_apply_generalized_permutation_matrix_get_workspace_size(
    rocstatevec_handle, rocstatevec_data_type sv_dtype, uint32_t n_index_bits,
    const rocstatevec_index_t* permutation, const void* diagonals,
    rocstatevec_data_type /*d_dtype*/, const int32_t* /*targets*/,
    uint32_t n_targets, uint32_t /*n_controls*/,
    size_t* extra_workspace_size_in_bytes)
{
    using namespace rocstatevec;
    if(extra_workspace_size_in_bytes == nullptr) return ROCSTATEVEC_STATUS_SUCCESS;
    if(sv_dtype != ROCSTATEVEC_C_64F && sv_dtype != ROCSTATEVEC_C_32F)
    {
        *extra_workspace_size_in_bytes = 0;
        return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
    }
    if(n_targets == 0 || n_targets > 16)
    {
        *extra_workspace_size_in_bytes = 0;
        return ROCSTATEVEC_STATUS_INVALID_VALUE;
    }
    int has_perm = (permutation != nullptr) ? 1 : 0;
    int has_diag = (diagonals != nullptr) ? 1 : 0;
    *extra_workspace_size_in_bytes
        = gen_perm_workspace_bytes(sv_dtype, n_index_bits, n_targets, has_diag, has_perm);
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_apply_generalized_permutation_matrix(
    rocstatevec_handle h, void* state_vector, rocstatevec_data_type sv_dt, uint32_t n,
    const rocstatevec_index_t* permutation, const void* diagonals, rocstatevec_data_type d_dt,
    int32_t adjoint, const int32_t* targets, uint32_t n_targets,
    const int32_t* controls, const int32_t* control_bit_values, uint32_t n_controls,
    void* extra_workspace, size_t extra_workspace_size_in_bytes)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(state_vector);
    if(n_targets == 0 || n_targets > 16) return ROCSTATEVEC_STATUS_INVALID_VALUE;
    if(n_controls > 32)                  return ROCSTATEVEC_STATUS_INVALID_VALUE;
    ROCSTATEVEC_CHECK_PTR(targets);
    int has_diag = (diagonals != nullptr) ? 1 : 0;
    if(has_diag && sv_dt != d_dt) return ROCSTATEVEC_STATUS_NOT_SUPPORTED;

    if(sv_dt == ROCSTATEVEC_C_64F)
        return gen_perm_dispatch<c64>(h, state_vector, n, permutation,
                                      diagonals, has_diag, adjoint,
                                      targets, n_targets, controls, control_bit_values, n_controls,
                                      extra_workspace, extra_workspace_size_in_bytes);
    if(sv_dt == ROCSTATEVEC_C_32F)
        return gen_perm_dispatch<c32>(h, state_vector, n, permutation,
                                      diagonals, has_diag, adjoint,
                                      targets, n_targets, controls, control_bit_values, n_controls,
                                      extra_workspace, extra_workspace_size_in_bytes);
    return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
}
