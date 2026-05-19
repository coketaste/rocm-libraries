/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Phase 9 — `apply_matrix_batched`.
 *
 * Applies one of `n_matrices` `(2^k x 2^k)` matrices to each of the
 * `n_state_vectors` state vectors stored back-to-back in
 * `batched_state_vectors`.
 *
 *   * `BATCH_LAUNCH_INDIVIDUAL` -> per-batch `matrix_indices[]`
 *   * `BATCH_LAUNCH_BROADCAST`  -> single matrix index applied to all
 *
 * The implementation iterates the batch on the host and dispatches the
 * scalar `apply_matrix` kernel per state vector. A more aggressive
 * launch could fuse the batch into a single kernel; we keep the simple,
 * obviously-correct dispatch since cuStateVec users typically use small
 * `k` (1-4 qubits) per gate and the kernel launch overhead is
 * acceptable. The batched matrices array is auto-detected as host or
 * device memory.
 * ************************************************************************ */

#include "rocstatevec_kernels.hpp"

namespace rocstatevec
{

static size_t batched_workspace_bytes(rocstatevec_data_type dtype,
                                      uint32_t              n_index_bits,
                                      uint32_t              n_targets)
{
    // We re-use the per-call apply_matrix workspace footprint.
    size_t es        = element_size_bytes(dtype);
    size_t targets_b = align_up(size_t(n_targets) * sizeof(int32_t),
                                default_workspace_align);
    size_t m         = size_t{1} << n_targets;
    size_t matrix_b  = align_up(m * m * es, default_workspace_align);
    size_t sv_n      = size_t{1} << n_index_bits;
    size_t scratch_b = align_up(sv_n * es, default_workspace_align);
    return targets_b + matrix_b + scratch_b;
}

} // namespace rocstatevec

extern "C" rocstatevec_status rocstatevec_apply_matrix_batched_get_workspace_size(
    rocstatevec_handle, rocstatevec_data_type sv_dtype,
    uint32_t n_index_bits, uint32_t /*n_state_vectors*/,
    rocstatevec_index_t /*state_vector_size*/, rocstatevec_matrix_map_type,
    const int32_t* /*matrix_indices*/, const void* /*matrices*/,
    rocstatevec_data_type /*matrix_dtype*/, rocstatevec_matrix_layout,
    int32_t /*adjoint*/, uint32_t /*n_matrices*/, uint32_t n_targets,
    uint32_t /*n_controls*/, rocstatevec_compute_type,
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
        = batched_workspace_bytes(sv_dtype, n_index_bits, n_targets);
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_apply_matrix_batched(
    rocstatevec_handle h, void* batched_state_vectors, rocstatevec_data_type dtype,
    uint32_t n_index_bits, uint32_t n_state_vectors, rocstatevec_index_t state_vector_size,
    rocstatevec_matrix_map_type map_type, const int32_t* matrix_indices,
    const void* matrices, rocstatevec_data_type matrix_dtype,
    rocstatevec_matrix_layout layout, int32_t adjoint, uint32_t n_matrices,
    const int32_t* targets, uint32_t n_targets, const int32_t* controls,
    const int32_t* control_bit_values, uint32_t n_controls,
    rocstatevec_compute_type compute_type, void* extra_workspace,
    size_t extra_workspace_size_in_bytes)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(batched_state_vectors);
    ROCSTATEVEC_CHECK_PTR(matrices);

    if(matrix_dtype != dtype) return ROCSTATEVEC_STATUS_NOT_SUPPORTED;

    size_t   m_dim         = size_t{1} << n_targets;
    size_t   per_matrix_sz = m_dim * m_dim * element_size_bytes(dtype);
    size_t   per_sv_count  = static_cast<size_t>(state_vector_size);
    if(per_sv_count == 0) per_sv_count = size_t{1} << n_index_bits;
    size_t per_sv_bytes = per_sv_count * element_size_bytes(dtype);

    const uint8_t* matrices_bytes = reinterpret_cast<const uint8_t*>(matrices);
    uint8_t*       sv_bytes       = reinterpret_cast<uint8_t*>(batched_state_vectors);

    for(uint32_t b = 0; b < n_state_vectors; ++b)
    {
        int32_t mi = 0;
        if(map_type == ROCSTATEVEC_MATRIX_MAP_TYPE_BROADCAST)
        {
            mi = matrix_indices ? matrix_indices[0] : 0;
        }
        else
        {
            ROCSTATEVEC_CHECK_PTR(matrix_indices);
            mi = matrix_indices[b];
        }
        if(mi < 0 || uint32_t(mi) >= n_matrices) return ROCSTATEVEC_STATUS_INVALID_VALUE;

        const void* m_ptr = matrices_bytes + size_t(mi) * per_matrix_sz;
        void*       v_ptr = sv_bytes + size_t(b) * per_sv_bytes;

        rocstatevec_status s = rocstatevec_apply_matrix(
            h, v_ptr, dtype, n_index_bits, m_ptr, matrix_dtype, layout, adjoint,
            targets, n_targets, controls, control_bit_values, n_controls,
            compute_type, extra_workspace, extra_workspace_size_in_bytes);
        if(s != ROCSTATEVEC_STATUS_SUCCESS) return s;
    }
    return ROCSTATEVEC_STATUS_SUCCESS;
}
