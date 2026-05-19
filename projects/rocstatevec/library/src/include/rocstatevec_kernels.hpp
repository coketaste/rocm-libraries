/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Internal kernel helpers and complex-arithmetic utilities used by every
 * rocSTATEVEC source file that runs HIP kernels.
 * ************************************************************************ */

#ifndef ROCSTATEVEC_KERNELS_HPP
#define ROCSTATEVEC_KERNELS_HPP

#include "rocstatevec_internal.hpp"

#include <hip/hip_runtime.h>

namespace rocstatevec
{

using c64 = hipDoubleComplex;
using c32 = hipFloatComplex;

/*! \brief Type-tag used to dispatch templates between FP32 and FP64 complex. */
template <typename T>
struct complex_traits
{
};

template <>
struct complex_traits<c64>
{
    using real_t = double;
    __host__ __device__ static c64 zero()                   { return make_hipDoubleComplex(0.0, 0.0); }
    __host__ __device__ static c64 one()                    { return make_hipDoubleComplex(1.0, 0.0); }
    __host__ __device__ static c64 from_real(real_t r)      { return make_hipDoubleComplex(r, 0.0); }
    __host__ __device__ static real_t real(const c64& z)    { return z.x; }
    __host__ __device__ static real_t imag(const c64& z)    { return z.y; }
    __host__ __device__ static c64 add(const c64& a, const c64& b) { return make_hipDoubleComplex(a.x + b.x, a.y + b.y); }
    __host__ __device__ static c64 sub(const c64& a, const c64& b) { return make_hipDoubleComplex(a.x - b.x, a.y - b.y); }
    __host__ __device__ static c64 mul(const c64& a, const c64& b)
    {
        return make_hipDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
    __host__ __device__ static c64 conj(const c64& a) { return make_hipDoubleComplex(a.x, -a.y); }
    __host__ __device__ static real_t abs2(const c64& a) { return a.x * a.x + a.y * a.y; }
    __host__ __device__ static c64 scale(const c64& a, real_t s) { return make_hipDoubleComplex(a.x * s, a.y * s); }
};

template <>
struct complex_traits<c32>
{
    using real_t = float;
    __host__ __device__ static c32 zero()                   { return make_hipFloatComplex(0.0f, 0.0f); }
    __host__ __device__ static c32 one()                    { return make_hipFloatComplex(1.0f, 0.0f); }
    __host__ __device__ static c32 from_real(real_t r)      { return make_hipFloatComplex(r, 0.0f); }
    __host__ __device__ static real_t real(const c32& z)    { return z.x; }
    __host__ __device__ static real_t imag(const c32& z)    { return z.y; }
    __host__ __device__ static c32 add(const c32& a, const c32& b) { return make_hipFloatComplex(a.x + b.x, a.y + b.y); }
    __host__ __device__ static c32 sub(const c32& a, const c32& b) { return make_hipFloatComplex(a.x - b.x, a.y - b.y); }
    __host__ __device__ static c32 mul(const c32& a, const c32& b)
    {
        return make_hipFloatComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
    __host__ __device__ static c32 conj(const c32& a) { return make_hipFloatComplex(a.x, -a.y); }
    __host__ __device__ static real_t abs2(const c32& a) { return a.x * a.x + a.y * a.y; }
    __host__ __device__ static c32 scale(const c32& a, real_t s) { return make_hipFloatComplex(a.x * s, a.y * s); }
};

/*! \brief Build a bitmask from an array of qubit indices.
 *  Same semantics on host and device. */
__host__ __device__ inline rocstatevec_index_t
qubit_mask(const int32_t* bits, uint32_t n_bits)
{
    rocstatevec_index_t m = 0;
    for(uint32_t i = 0; i < n_bits; ++i) m |= (rocstatevec_index_t{1} << bits[i]);
    return m;
}

/*! \brief Population-count-and-parity (returns 0 or 1). */
__host__ __device__ inline int parity64(rocstatevec_index_t v)
{
#if defined(__HIP_DEVICE_COMPILE__)
    return __popcll(v) & 1;
#else
    int p = 0;
    while(v) { v &= v - 1; ++p; }
    return p & 1;
#endif
}

/*! \brief Stream extraction helper for the bound stream on a handle. */
inline hipStream_t handle_stream(rocstatevec_handle h)
{
    return reinterpret_cast<rocstatevec::handle*>(h)->stream;
}

inline rocstatevec_status dispatch_complex(rocstatevec_data_type t)
{
    if(t != ROCSTATEVEC_C_32F && t != ROCSTATEVEC_C_64F) return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

} // namespace rocstatevec

#endif /* ROCSTATEVEC_KERNELS_HPP */
