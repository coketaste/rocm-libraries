/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef ROCDENSITYMAT_TEST_HELPERS_HPP
#define ROCDENSITYMAT_TEST_HELPERS_HPP

#include <rocdensitymat.h>
#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

#define ROC_DM_OK(call) ASSERT_EQ((call), ROCDENSITYMAT_STATUS_SUCCESS)

#define ROC_DM_EXPECT_NOT_SUPPORTED(call) \
    ASSERT_EQ((call), ROCDENSITYMAT_STATUS_NOT_SUPPORTED)

inline void hip_check(hipError_t e, const char* file, int line)
{
    if(e != hipSuccess)
        FAIL() << "HIP error at " << file << ":" << line << " - " << hipGetErrorString(e);
}
#define HIP_OK(call) hip_check((call), __FILE__, __LINE__)

/*! \brief Allocate a complex device buffer of `n` elements, copy
 *  `host_data` into it, and return the device pointer. The caller is
 *  responsible for hipFree-ing the returned pointer. */
inline hipDoubleComplex*
upload_c64(const std::vector<std::complex<double>>& host_data)
{
    hipDoubleComplex* d = nullptr;
    if(hipMalloc(&d, host_data.size() * sizeof(hipDoubleComplex)) != hipSuccess)
        return nullptr;
    std::vector<hipDoubleComplex> tmp(host_data.size());
    for(size_t i = 0; i < host_data.size(); ++i)
    {
        tmp[i].x = host_data[i].real();
        tmp[i].y = host_data[i].imag();
    }
    if(hipMemcpy(d, tmp.data(), tmp.size() * sizeof(hipDoubleComplex),
                 hipMemcpyHostToDevice) != hipSuccess)
    {
        hipFree(d);
        return nullptr;
    }
    return d;
}

/*! \brief Read back a complex device buffer to a host vector. */
inline std::vector<std::complex<double>>
download_c64(const hipDoubleComplex* d, size_t n)
{
    std::vector<hipDoubleComplex> tmp(n);
    hipMemcpy(tmp.data(), d, n * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost);
    std::vector<std::complex<double>> out(n);
    for(size_t i = 0; i < n; ++i) out[i] = std::complex<double>(tmp[i].x, tmp[i].y);
    return out;
}

#endif
