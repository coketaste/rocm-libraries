/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Apply a dense matrix to a sub-set of qubits, optionally with controls.
 *
 * Algorithm:
 *   1. Stage to an internal scratch buffer (workspace) to avoid in-place
 *      data races between the m_targets gathered loads and the resulting
 *      stores.
 *   2. One thread per output amplitude. If the amplitude's control bits
 *      do not match the requested values, copy through. Otherwise, gather
 *      the m_targets-tuple of source amplitudes by zero/one-substituting
 *      the target bits, multiply by the requested matrix row, and store.
 *   3. Memcpy the scratch buffer back into the user state-vector buffer.
 *
 * Supports up to 10 targets (m = 1024), all 32 controls.
 *
 * Workspace layout (when caller provides one via extra_workspace):
 *   [aligned] device targets[]  : n_targets * sizeof(int32_t)
 *   [aligned] device matrix[]   : 0 if matrix is already device-resident,
 *                                 else m*m * element_size_bytes(dtype)
 *   [aligned] device scratch[]  : N * element_size_bytes(dtype)
 *
 * If the user passes a workspace too small to hold all three sections,
 * the leftover allocations fall back to dev_alloc (which routes through
 * a user-installed device-mem handler when one is bound to the handle).
 * ************************************************************************ */

#include "rocstatevec_kernels.hpp"

namespace rocstatevec
{

template <typename C>
__global__ void k_apply_matrix(
    const C* in, C* out, rocstatevec_index_t N,
    const C* matrix, uint32_t n_targets, int row_major, int adjoint,
    const int32_t* targets, rocstatevec_index_t target_mask,
    rocstatevec_index_t control_mask, rocstatevec_index_t control_match)
{
    using T = complex_traits<C>;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    if((i & control_mask) != control_match) { out[i] = in[i]; return; }

    uint32_t m   = 1u << n_targets;
    uint32_t row = 0;
    rocstatevec_index_t base = i & ~target_mask;
    for(uint32_t b = 0; b < n_targets; ++b)
        if(((i >> targets[b]) & 1) != 0) row |= (1u << b);

    C acc = T::zero();
    for(uint32_t j = 0; j < m; ++j)
    {
        rocstatevec_index_t in_idx = base;
        for(uint32_t b = 0; b < n_targets; ++b)
            if(((j >> b) & 1) != 0) in_idx |= (rocstatevec_index_t{1} << targets[b]);

        uint32_t r = adjoint ? j   : row;
        uint32_t c = adjoint ? row : j;
        C M       = row_major ? matrix[r * m + c] : matrix[c * m + r];
        if(adjoint) M = T::conj(M);

        acc = T::add(acc, T::mul(M, in[in_idx]));
    }
    out[i] = acc;
}

template <typename C>
static rocstatevec_status apply_matrix_dispatch(
    rocstatevec_handle h_in, void* dsv, uint32_t n,
    const void* host_or_device_matrix,
    rocstatevec_matrix_layout layout, int adjoint,
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

    const C* d_matrix = nullptr;
    if(is_device_pointer(host_or_device_matrix))
    {
        d_matrix = reinterpret_cast<const C*>(host_or_device_matrix);
    }
    else
    {
        void* tmp = nullptr;
        ROCSTATEVEC_RC_CHECK(arena.alloc_and_copy(&tmp, host_or_device_matrix,
                                                  size_t(m) * m * sizeof(C)));
        d_matrix = reinterpret_cast<const C*>(tmp);
    }

    void* scratch_v = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc(&scratch_v, size_t(N) * sizeof(C)));
    auto* d_scratch = reinterpret_cast<C*>(scratch_v);

    hipLaunchKernelGGL(k_apply_matrix<C>, dim3(gpc), dim3(tpb), 0, s,
                       reinterpret_cast<const C*>(dsv), d_scratch, N,
                       d_matrix, n_targets,
                       layout == ROCSTATEVEC_MATRIX_LAYOUT_ROW ? 1 : 0,
                       adjoint != 0 ? 1 : 0,
                       d_targets, target_mask, control_mask, control_match);

    ROCSTATEVEC_HIP_CHECK(hipMemcpyAsync(dsv, d_scratch, size_t(N) * sizeof(C),
                                         hipMemcpyDeviceToDevice, s));
    return ROCSTATEVEC_STATUS_SUCCESS;
}

/*! \brief Compute the maximum workspace footprint a call to
 *  `rocstatevec_apply_matrix` could use, assuming the matrix is host
 *  memory. If the matrix turns out to be device-resident at apply time,
 *  the matrix slice goes unused (and remains owned by the caller). */
static size_t apply_matrix_workspace_bytes(rocstatevec_data_type dtype,
                                           uint32_t              n,
                                           uint32_t              n_targets)
{
    size_t es        = element_size_bytes(dtype);
    size_t targets_b = align_up(size_t(n_targets) * sizeof(int32_t),
                                default_workspace_align);
    size_t m         = size_t{1} << n_targets;
    size_t matrix_b  = align_up(m * m * es, default_workspace_align);
    size_t sv_n      = size_t{1} << n;
    size_t scratch_b = align_up(sv_n * es, default_workspace_align);
    return targets_b + matrix_b + scratch_b;
}

} // namespace rocstatevec

extern "C" rocstatevec_status rocstatevec_apply_matrix_get_workspace_size(
    rocstatevec_handle, rocstatevec_data_type sv_dtype, uint32_t n_index_bits,
    const void*, rocstatevec_data_type, rocstatevec_matrix_layout, int32_t,
    uint32_t n_targets, uint32_t /*n_controls*/, rocstatevec_compute_type,
    size_t* extra_workspace_size_in_bytes)
{
    using namespace rocstatevec;
    if(extra_workspace_size_in_bytes == nullptr) return ROCSTATEVEC_STATUS_SUCCESS;
    if(sv_dtype != ROCSTATEVEC_C_64F && sv_dtype != ROCSTATEVEC_C_32F)
    {
        *extra_workspace_size_in_bytes = 0;
        return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
    }
    if(n_targets == 0 || n_targets > 10)
    {
        *extra_workspace_size_in_bytes = 0;
        return ROCSTATEVEC_STATUS_INVALID_VALUE;
    }
    *extra_workspace_size_in_bytes
        = apply_matrix_workspace_bytes(sv_dtype, n_index_bits, n_targets);
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_apply_matrix(
    rocstatevec_handle h, void* state_vector, rocstatevec_data_type sv_dtype, uint32_t n,
    const void* matrix, rocstatevec_data_type m_dtype,
    rocstatevec_matrix_layout layout, int32_t adjoint,
    const int32_t* targets, uint32_t n_targets,
    const int32_t* controls, const int32_t* control_bit_values, uint32_t n_controls,
    rocstatevec_compute_type, void* extra_workspace, size_t extra_workspace_size_in_bytes)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(state_vector); ROCSTATEVEC_CHECK_PTR(matrix);
    if(n_targets == 0 || n_targets > 10) return ROCSTATEVEC_STATUS_INVALID_VALUE;
    if(n_controls > 32)                  return ROCSTATEVEC_STATUS_INVALID_VALUE;
    ROCSTATEVEC_CHECK_PTR(targets);
    if(n_controls > 0)                   ROCSTATEVEC_CHECK_PTR(controls);
    if(sv_dtype != m_dtype)              return ROCSTATEVEC_STATUS_NOT_SUPPORTED;

    if(sv_dtype == ROCSTATEVEC_C_64F)
        return apply_matrix_dispatch<c64>(h, state_vector, n, matrix,
                                          layout, adjoint, targets, n_targets,
                                          controls, control_bit_values, n_controls,
                                          extra_workspace, extra_workspace_size_in_bytes);
    if(sv_dtype == ROCSTATEVEC_C_32F)
        return apply_matrix_dispatch<c32>(h, state_vector, n, matrix,
                                          layout, adjoint, targets, n_targets,
                                          controls, control_bit_values, n_controls,
                                          extra_workspace, extra_workspace_size_in_bytes);
    return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
}
