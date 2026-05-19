/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * `absorb_diagonal_matrix` — multiply each amplitude by the diagonal
 * entry indexed by the bit pattern of its target qubits, gated on the
 * control mask. Equivalent to applying diag(D_{t_0,...,t_{m-1}}) to the
 * sub-block selected by `targets` and `controls`.
 *
 * Auto-detects host vs device `diagonals` via `is_device_pointer` and
 * routes scratch through the user-installed device-mem handler when one
 * is bound to the handle.
 * ************************************************************************ */

#include "rocstatevec_kernels.hpp"

namespace rocstatevec
{

template <typename C>
__global__ void k_absorb_diagonal(
    C* sv, rocstatevec_index_t N, const C* diag, uint32_t n_targets, int adjoint,
    const int32_t* targets, rocstatevec_index_t /*target_mask*/,
    rocstatevec_index_t control_mask, rocstatevec_index_t control_match)
{
    using T = complex_traits<C>;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    if((i & control_mask) != control_match) return;

    uint32_t row = 0;
    for(uint32_t b = 0; b < n_targets; ++b)
        if(((i >> targets[b]) & 1) != 0) row |= (1u << b);

    C d = diag[row];
    if(adjoint) d = T::conj(d);
    sv[i] = T::mul(sv[i], d);
}

template <typename C>
static rocstatevec_status absorb_dispatch(
    rocstatevec_handle h_in, void* dsv, uint32_t n,
    const void* diagonals, int adjoint,
    const int32_t* targets, uint32_t n_targets,
    const int32_t* controls, const int32_t* control_bit_values, uint32_t n_controls)
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

    dev_arena arena(hh, s);

    int32_t* d_targets = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc_and_copy(reinterpret_cast<void**>(&d_targets),
                                              targets, n_targets * sizeof(int32_t)));

    const C* d_diag = nullptr;
    if(is_device_pointer(diagonals))
    {
        d_diag = reinterpret_cast<const C*>(diagonals);
    }
    else
    {
        void* tmp = nullptr;
        ROCSTATEVEC_RC_CHECK(arena.alloc_and_copy(&tmp, diagonals, m * sizeof(C)));
        d_diag = reinterpret_cast<const C*>(tmp);
    }

    hipLaunchKernelGGL(k_absorb_diagonal<C>, dim3(gpc), dim3(tpb), 0, s,
                       reinterpret_cast<C*>(dsv), N, d_diag, n_targets,
                       adjoint != 0 ? 1 : 0,
                       d_targets, target_mask, control_mask, control_match);
    return ROCSTATEVEC_STATUS_SUCCESS;
}

} // namespace rocstatevec

extern "C" rocstatevec_status rocstatevec_absorb_diagonal_matrix(
    rocstatevec_handle h, void* state_vector, rocstatevec_data_type sv_dt, uint32_t n,
    const void* diagonals, rocstatevec_data_type d_dt, int32_t adjoint,
    const int32_t* targets, uint32_t n_targets,
    const int32_t* controls, const int32_t* control_bit_values, uint32_t n_controls)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(state_vector); ROCSTATEVEC_CHECK_PTR(diagonals);
    if(n_targets == 0 || n_targets > 16) return ROCSTATEVEC_STATUS_INVALID_VALUE;
    if(n_controls > 32)                  return ROCSTATEVEC_STATUS_INVALID_VALUE;
    ROCSTATEVEC_CHECK_PTR(targets);
    if(sv_dt != d_dt) return ROCSTATEVEC_STATUS_NOT_SUPPORTED;

    if(sv_dt == ROCSTATEVEC_C_64F)
        return absorb_dispatch<c64>(h, state_vector, n, diagonals, adjoint,
                                    targets, n_targets, controls, control_bit_values, n_controls);
    if(sv_dt == ROCSTATEVEC_C_32F)
        return absorb_dispatch<c32>(h, state_vector, n, diagonals, adjoint,
                                    targets, n_targets, controls, control_bit_values, n_controls);
    return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
}
