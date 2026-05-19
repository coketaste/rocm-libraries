/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * `apply_pauli_rotation` — applies U = exp(-i theta/2 * P_0 ⊗ P_1 ⊗ ...).
 *
 * Decomposition: U = cos(theta/2) * I - i * sin(theta/2) * P. Define
 *
 *     mask_X = bits where P_q = X        (flips, no phase)
 *     mask_Y = bits where P_q = Y        (flips, contributes a phase i)
 *     mask_Z = bits where P_q = Z        (no flip, contributes (-1)^bit)
 *     mask_flip = mask_X | mask_Y
 *     mask_yz   = mask_Y | mask_Z
 *
 * Phase of P|x>: phase(x) = i^n_Y * (-1)^parity(x & mask_yz).
 *
 * For each pair (a, b = a XOR mask_flip) with a < b:
 *
 *     out[a] = cos(t/2) in[a] + (-i sin(t/2)) phase(b) in[b]
 *     out[b] = cos(t/2) in[b] + (-i sin(t/2)) phase(a) in[a]
 *
 * For mask_flip == 0 (pure-Z rotation), each pair degenerates to a single
 * index and the update is:
 *
 *     out[i] = (cos(t/2) - i sin(t/2) phase(i)) in[i]
 *
 * Controls gate the entire 2x2 (or 1x1) block: if controls do not match,
 * leave both amplitudes unchanged.
 * ************************************************************************ */

#include "rocstatevec_kernels.hpp"

#include <cmath>

namespace rocstatevec
{

/*! \brief Multiply a complex value by `i^q` for `q in [0..3]`. */
template <typename C>
__device__ inline C mul_i_pow(C in, int q)
{
    using T = complex_traits<C>;
    using R = typename complex_traits<C>::real_t;
    R re = T::real(in);
    R im = T::imag(in);
    R o_re;
    R o_im;
    switch(q & 3)
    {
    default:
    case 0: o_re =  re; o_im =  im; break;
    case 1: o_re = -im; o_im =  re; break;
    case 2: o_re = -re; o_im = -im; break;
    case 3: o_re =  im; o_im = -re; break;
    }
    C out;
    out.x = o_re;
    out.y = o_im;
    return out;
}

template <typename C>
__device__ inline int phase_quarter(rocstatevec_index_t x, rocstatevec_index_t mask_yz, int n_y)
{
    int q = (parity64(x & mask_yz) != 0) ? 2 : 0;
    return (q + (n_y & 3)) & 3;
}

template <typename C>
__global__ void k_pauli_diag(C* sv, rocstatevec_index_t N, rocstatevec_index_t mask_yz, int n_y,
                             rocstatevec_index_t control_mask, rocstatevec_index_t control_match,
                             typename complex_traits<C>::real_t cos_t, typename complex_traits<C>::real_t sin_t)
{
    using T = complex_traits<C>;
    using R = typename complex_traits<C>::real_t;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    if((i & control_mask) != control_match) return;

    int qi = phase_quarter<C>(i, mask_yz, n_y);
    C   in = sv[i];

    C   neg_i_phase_in = mul_i_pow<C>(in, (qi + 3) & 3);
    C   diag_in        = T::scale(in, cos_t);
    C   off_in         = T::scale(neg_i_phase_in, sin_t);
    sv[i] = T::add(diag_in, off_in);
    (void)R{};
}

template <typename C>
__global__ void k_pauli_pair(C* sv, rocstatevec_index_t N,
                             rocstatevec_index_t mask_flip, rocstatevec_index_t mask_yz, int n_y,
                             rocstatevec_index_t control_mask, rocstatevec_index_t control_match,
                             typename complex_traits<C>::real_t cos_t, typename complex_traits<C>::real_t sin_t)
{
    using T = complex_traits<C>;
    using R = typename complex_traits<C>::real_t;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    rocstatevec_index_t j = i ^ mask_flip;
    if(i >= j) return;
    if((i & control_mask) != control_match) return;

    C in_i = sv[i];
    C in_j = sv[j];

    int qa = phase_quarter<C>(i, mask_yz, n_y);
    int qb = phase_quarter<C>(j, mask_yz, n_y);

    C term_a = mul_i_pow<C>(in_j, (qb + 3) & 3);
    C term_b = mul_i_pow<C>(in_i, (qa + 3) & 3);

    C new_i = T::add(T::scale(in_i, cos_t), T::scale(term_a, sin_t));
    C new_j = T::add(T::scale(in_j, cos_t), T::scale(term_b, sin_t));
    sv[i] = new_i;
    sv[j] = new_j;
    (void)R{};
}

template <typename C>
static rocstatevec_status pauli_dispatch(
    rocstatevec_handle h, void* dsv, uint32_t n,
    double theta, const rocstatevec_pauli* paulis,
    const int32_t* targets, uint32_t n_targets,
    const int32_t* controls, const int32_t* control_bit_values, uint32_t n_controls)
{
    using TR = typename complex_traits<C>::real_t;

    rocstatevec_index_t mask_x = 0, mask_y = 0, mask_z = 0;
    int n_y = 0;
    for(uint32_t i = 0; i < n_targets; ++i)
    {
        rocstatevec_index_t bit = rocstatevec_index_t{1} << targets[i];
        switch(paulis[i])
        {
        case ROCSTATEVEC_PAULI_I: break;
        case ROCSTATEVEC_PAULI_X: mask_x |= bit; break;
        case ROCSTATEVEC_PAULI_Y: mask_y |= bit; ++n_y; break;
        case ROCSTATEVEC_PAULI_Z: mask_z |= bit; break;
        default: return ROCSTATEVEC_STATUS_INVALID_VALUE;
        }
    }
    rocstatevec_index_t mask_flip = mask_x | mask_y;
    rocstatevec_index_t mask_yz   = mask_y | mask_z;

    rocstatevec_index_t control_mask = qubit_mask(controls, n_controls);
    rocstatevec_index_t control_match = 0;
    for(uint32_t i = 0; i < n_controls; ++i)
    {
        int v = control_bit_values ? control_bit_values[i] : 1;
        if(v != 0) control_match |= (rocstatevec_index_t{1} << controls[i]);
    }

    rocstatevec_index_t N    = state_vector_length(n);
    hipStream_t         s    = handle_stream(h);
    int                 tpb  = default_threads_per_block;
    int                 gpc  = static_cast<int>(ceil_div<rocstatevec_index_t>(N, tpb));

    TR cos_t = static_cast<TR>(std::cos(theta * 0.5));
    TR sin_t = static_cast<TR>(std::sin(theta * 0.5));

    if(mask_flip == 0)
    {
        hipLaunchKernelGGL(k_pauli_diag<C>, dim3(gpc), dim3(tpb), 0, s,
                           reinterpret_cast<C*>(dsv), N, mask_yz, n_y,
                           control_mask, control_match, cos_t, sin_t);
    }
    else
    {
        hipLaunchKernelGGL(k_pauli_pair<C>, dim3(gpc), dim3(tpb), 0, s,
                           reinterpret_cast<C*>(dsv), N, mask_flip, mask_yz, n_y,
                           control_mask, control_match, cos_t, sin_t);
    }
    return ROCSTATEVEC_STATUS_SUCCESS;
}

} // namespace rocstatevec

extern "C" rocstatevec_status rocstatevec_apply_pauli_rotation(
    rocstatevec_handle h, void* dsv, rocstatevec_data_type dtype, uint32_t n,
    double theta, const rocstatevec_pauli* paulis,
    const int32_t* targets, uint32_t n_targets,
    const int32_t* controls, const int32_t* control_bit_values, uint32_t n_controls)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(dsv);
    if(n_targets == 0)               return ROCSTATEVEC_STATUS_INVALID_VALUE;
    ROCSTATEVEC_CHECK_PTR(paulis);
    ROCSTATEVEC_CHECK_PTR(targets);
    if(n_controls > 32)              return ROCSTATEVEC_STATUS_INVALID_VALUE;
    if(n_controls > 0)               ROCSTATEVEC_CHECK_PTR(controls);

    if(dtype == ROCSTATEVEC_C_64F)
        return pauli_dispatch<c64>(h, dsv, n, theta, paulis, targets, n_targets,
                                   controls, control_bit_values, n_controls);
    if(dtype == ROCSTATEVEC_C_32F)
        return pauli_dispatch<c32>(h, dsv, n, theta, paulis, targets, n_targets,
                                   controls, control_bit_values, n_controls);
    return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
}
