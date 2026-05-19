/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#pragma once

#include <rocstatevec.h>

#include <hip/hip_runtime.h>

#include <complex>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace rocstatevec::test
{

inline constexpr double tol_fp64 = 1e-12;
inline constexpr double tol_fp32 = 5e-6;

/*! \brief Skip a GoogleTest case if no AMD GPU is available; \return true if skipped. */
inline bool skip_if_no_gpu()
{
    int n = 0;
    if(hipGetDeviceCount(&n) != hipSuccess || n == 0)
    {
        std::printf("[ SKIPPED ] no HIP-capable device found.\n");
        return true;
    }
    return false;
}

struct handle_guard
{
    rocstatevec_handle h = nullptr;
    handle_guard()
    {
        if(rocstatevec_create_handle(&h) != ROCSTATEVEC_STATUS_SUCCESS) h = nullptr;
    }
    ~handle_guard() { if(h) rocstatevec_destroy_handle(h); }
    handle_guard(const handle_guard&)            = delete;
    handle_guard& operator=(const handle_guard&) = delete;
};

template <typename T>
struct device_buffer
{
    T*     ptr   = nullptr;
    size_t count = 0;
    explicit device_buffer(size_t n) : count(n)
    {
        hipMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T));
    }
    ~device_buffer() { if(ptr) hipFree(ptr); }
    device_buffer(const device_buffer&)            = delete;
    device_buffer& operator=(const device_buffer&) = delete;
};

template <typename T>
inline void copy_to_device(T* dst, const T* src, size_t n)
{
    hipMemcpy(dst, src, n * sizeof(T), hipMemcpyHostToDevice);
}
template <typename T>
inline void copy_to_host(T* dst, const T* src, size_t n)
{
    hipMemcpy(dst, src, n * sizeof(T), hipMemcpyDeviceToHost);
}

/*! \brief Sum |amp|^2 across a host buffer. */
template <typename T>
inline double l2_norm_squared(const std::vector<std::complex<T>>& v)
{
    double s = 0.0;
    for(const auto& z : v) s += static_cast<double>(z.real()) * static_cast<double>(z.real())
                              + static_cast<double>(z.imag()) * static_cast<double>(z.imag());
    return s;
}

} // namespace rocstatevec::test
