/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * `compute_expectations_on_pauli_basis` — for each Pauli string `P_k`,
 * compute <psi|P_k|psi>.
 *
 * For a Pauli string acting as P|x> = phase(x) |x XOR mask_flip>:
 *
 *     <psi|P|psi> = sum_x conj(amp[x]) * phase(x XOR mask_flip)
 *                                      * amp[x XOR mask_flip]
 *
 * The result is provably real (Pauli operators are Hermitian); the kernel
 * accumulates only the real part into a single double per Pauli string.
 *
 * Phase factor: phase(x) = i^n_Y * (-1)^parity(x & mask_yz). The kernel
 * combines the i^n_Y prefactor at the end on the host (it is constant
 * per Pauli string).
 * ************************************************************************ */

#include "rocstatevec_kernels.hpp"

namespace rocstatevec
{

template <typename C>
__global__ void k_pauli_expectation_diag(
    const C* sv, rocstatevec_index_t N, rocstatevec_index_t mask_yz, double* d_acc)
{
    using T = complex_traits<C>;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int    s   = (parity64(i & mask_yz) != 0) ? -1 : 1;
    double v   = static_cast<double>(T::abs2(sv[i]));
    atomicAdd(d_acc, double(s) * v);
}

/*! \brief Pair-wise contribution to <psi|P|psi> when `mask_flip != 0`.
 *
 *  Let z_a = conj(amp[a]) * amp[b] for a = i, b = a XOR mask_flip, a < b.
 *  Let s_a = (-1)^parity(a & mask_yz). With S = i^n_y * s_a, the pair
 *  contribution is real and given by:
 *
 *      n_y even -> 2 Re(z_a) * Re(S)   [Re(S) = (-1)^(n_y/2) * s_a]
 *      n_y odd  -> 2 Im(z_a) * Im(S)   [Im(S) = (-1)^((n_y-1)/2) * s_a]
 */
template <typename C>
__global__ void k_pauli_expectation_pair(
    const C* sv, rocstatevec_index_t N,
    rocstatevec_index_t mask_flip, rocstatevec_index_t mask_yz,
    int n_y, double* d_acc)
{
    using T = complex_traits<C>;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    rocstatevec_index_t j = i ^ mask_flip;
    if(i >= j) return;

    C in_i = sv[i];
    C in_j = sv[j];

    double re_i = static_cast<double>(T::real(in_i));
    double im_i = static_cast<double>(T::imag(in_i));
    double re_j = static_cast<double>(T::real(in_j));
    double im_j = static_cast<double>(T::imag(in_j));

    double z_re = re_i * re_j + im_i * im_j;
    double z_im = re_i * im_j - im_i * re_j;

    double s_a = (parity64(i & mask_yz) != 0) ? -1.0 : 1.0;
    double contrib;
    if((n_y & 1) == 0)
    {
        double sign = (((n_y >> 1) & 1) == 0) ? 1.0 : -1.0;
        contrib = 2.0 * z_re * (sign * s_a);
    }
    else
    {
        double sign = ((((n_y - 1) >> 1) & 1) == 0) ? 1.0 : -1.0;
        contrib = 2.0 * z_im * (sign * s_a);
    }
    atomicAdd(d_acc, contrib);
}

template <typename C>
static rocstatevec_status pauli_basis_dispatch(
    rocstatevec_handle h_in, const void* dsv, uint32_t n,
    double* expectations, const rocstatevec_pauli* const* paulis_arr,
    uint32_t n_arrays, const int32_t* const* basis_bits_arr, const uint32_t* basis_lens)
{
    auto*               hh   = reinterpret_cast<handle*>(h_in);
    rocstatevec_index_t N    = state_vector_length(n);
    hipStream_t         s    = hh->stream;
    int                 tpb  = default_threads_per_block;
    int                 gpc  = static_cast<int>(ceil_div<rocstatevec_index_t>(N, tpb));

    dev_arena arena(hh, s);
    void*     d_acc_v = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc(&d_acc_v, sizeof(double)));
    auto* d_acc = reinterpret_cast<double*>(d_acc_v);

    for(uint32_t k = 0; k < n_arrays; ++k)
    {
        rocstatevec_index_t mask_x = 0, mask_y = 0, mask_z = 0;
        int n_y = 0;
        const rocstatevec_pauli* p   = paulis_arr[k];
        const int32_t*           tg  = basis_bits_arr[k];
        uint32_t                 nb  = basis_lens[k];
        for(uint32_t i = 0; i < nb; ++i)
        {
            rocstatevec_index_t bit = rocstatevec_index_t{1} << tg[i];
            switch(p[i])
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

        ROCSTATEVEC_HIP_CHECK(hipMemsetAsync(d_acc, 0, sizeof(double), s));

        if(mask_flip == 0)
        {
            hipLaunchKernelGGL(k_pauli_expectation_diag<C>, dim3(gpc), dim3(tpb), 0, s,
                               reinterpret_cast<const C*>(dsv), N, mask_yz, d_acc);
        }
        else
        {
            hipLaunchKernelGGL(k_pauli_expectation_pair<C>, dim3(gpc), dim3(tpb), 0, s,
                               reinterpret_cast<const C*>(dsv), N,
                               mask_flip, mask_yz, n_y, d_acc);
        }
        double host_v = 0.0;
        ROCSTATEVEC_HIP_CHECK(hipMemcpyAsync(&host_v, d_acc, sizeof(double),
                                             hipMemcpyDeviceToHost, s));
        ROCSTATEVEC_HIP_CHECK(hipStreamSynchronize(s));
        expectations[k] = host_v;
    }
    return ROCSTATEVEC_STATUS_SUCCESS;
}

} // namespace rocstatevec

extern "C" rocstatevec_status rocstatevec_compute_expectations_on_pauli_basis(
    rocstatevec_handle h, const void* dsv, rocstatevec_data_type dtype, uint32_t n,
    double* expectations, const rocstatevec_pauli* const* paulis_arr,
    uint32_t n_arrays, const int32_t* const* basis_bits_arr, const uint32_t* basis_lens)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(dsv); ROCSTATEVEC_CHECK_PTR(expectations);
    if(n_arrays == 0) return ROCSTATEVEC_STATUS_INVALID_VALUE;
    ROCSTATEVEC_CHECK_PTR(paulis_arr); ROCSTATEVEC_CHECK_PTR(basis_bits_arr);
    ROCSTATEVEC_CHECK_PTR(basis_lens);

    if(dtype == ROCSTATEVEC_C_64F)
        return pauli_basis_dispatch<c64>(h, dsv, n, expectations, paulis_arr, n_arrays,
                                         basis_bits_arr, basis_lens);
    if(dtype == ROCSTATEVEC_C_32F)
        return pauli_basis_dispatch<c32>(h, dsv, n, expectations, paulis_arr, n_arrays,
                                         basis_bits_arr, basis_lens);
    return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
}
