/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * `rocstatevec_initialize_state_vector` writes one of four named presets
 * into a freshly allocated state vector buffer:
 *   * ZERO    — |0...0>
 *   * UNIFORM — uniform superposition of all 2^n basis states
 *   * GHZ     — (|0...0> + |1...1>) / sqrt(2)
 *   * W       — equal superposition of all single-bit-set basis states
 * ************************************************************************ */

#include "rocstatevec_kernels.hpp"

#include <cmath>

namespace rocstatevec
{

template <typename C>
__global__ void k_init_zero(C* sv, rocstatevec_index_t N)
{
    using T = complex_traits<C>;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    sv[i] = (i == 0) ? T::one() : T::zero();
}

template <typename C>
__global__ void k_init_uniform(C* sv, rocstatevec_index_t N, typename complex_traits<C>::real_t amp)
{
    using T = complex_traits<C>;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    sv[i] = T::from_real(amp);
}

template <typename C>
__global__ void k_init_ghz(C* sv, rocstatevec_index_t N, typename complex_traits<C>::real_t amp)
{
    using T = complex_traits<C>;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    if(i == 0 || i == N - 1) sv[i] = T::from_real(amp);
    else                     sv[i] = T::zero();
}

template <typename C>
__global__ void k_init_w(C* sv, rocstatevec_index_t N, uint32_t n_qubits,
                         typename complex_traits<C>::real_t amp)
{
    using T = complex_traits<C>;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int p = 0;
    for(uint32_t b = 0; b < n_qubits; ++b)
        if(((i >> b) & 1) != 0) ++p;
    sv[i] = (p == 1) ? T::from_real(amp) : T::zero();
}

template <typename C>
static rocstatevec_status init_dispatch(rocstatevec_handle h, void* dsv, uint32_t n,
                                        rocstatevec_state_vector_type kind)
{
    using TR                 = typename complex_traits<C>::real_t;
    rocstatevec_index_t N    = state_vector_length(n);
    auto*               buf  = reinterpret_cast<C*>(dsv);
    hipStream_t         s    = handle_stream(h);
    int                 tpb  = default_threads_per_block;
    int                 gpc  = static_cast<int>(ceil_div<rocstatevec_index_t>(N, tpb));

    switch(kind)
    {
    case ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO:
        hipLaunchKernelGGL(k_init_zero<C>, dim3(gpc), dim3(tpb), 0, s, buf, N);
        break;
    case ROCSTATEVEC_STATE_VECTOR_TYPE_UNIFORM:
    {
        TR amp = static_cast<TR>(1.0 / std::sqrt(static_cast<double>(N)));
        hipLaunchKernelGGL(k_init_uniform<C>, dim3(gpc), dim3(tpb), 0, s, buf, N, amp);
        break;
    }
    case ROCSTATEVEC_STATE_VECTOR_TYPE_GHZ:
    {
        TR amp = static_cast<TR>(1.0 / std::sqrt(2.0));
        hipLaunchKernelGGL(k_init_ghz<C>, dim3(gpc), dim3(tpb), 0, s, buf, N, amp);
        break;
    }
    case ROCSTATEVEC_STATE_VECTOR_TYPE_W:
    {
        if(n == 0) return ROCSTATEVEC_STATUS_INVALID_VALUE;
        TR amp = static_cast<TR>(1.0 / std::sqrt(static_cast<double>(n)));
        hipLaunchKernelGGL(k_init_w<C>, dim3(gpc), dim3(tpb), 0, s, buf, N, n, amp);
        break;
    }
    default:
        return ROCSTATEVEC_STATUS_INVALID_VALUE;
    }
    return ROCSTATEVEC_STATUS_SUCCESS;
}

} // namespace rocstatevec

extern "C" rocstatevec_status rocstatevec_initialize_state_vector(
    rocstatevec_handle             h,
    void*                          state_vector,
    rocstatevec_data_type          dtype,
    uint32_t                       n_index_bits,
    rocstatevec_state_vector_type  kind)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(state_vector);
    if(n_index_bits > 50) return ROCSTATEVEC_STATUS_INVALID_VALUE;

    rocstatevec_status rc;
    if(dtype == ROCSTATEVEC_C_64F)
        rc = init_dispatch<c64>(h, state_vector, n_index_bits, kind);
    else if(dtype == ROCSTATEVEC_C_32F)
        rc = init_dispatch<c32>(h, state_vector, n_index_bits, kind);
    else
        return ROCSTATEVEC_STATUS_NOT_SUPPORTED;

    if(rc != ROCSTATEVEC_STATUS_SUCCESS) return rc;
    ROCSTATEVEC_HIP_CHECK(hipGetLastError());
    return ROCSTATEVEC_STATUS_SUCCESS;
}
