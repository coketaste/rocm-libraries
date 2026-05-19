/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Tensor QR factorization.
 *
 * v0.1.0 algorithm:
 *   1. Matricize the input into A (m x n) with rows = Q-modes, cols
 *      = R-modes. Share mode is the column index of A and row index
 *      of R.
 *   2. Run modified Gram-Schmidt to produce Q (m x n) with
 *      orthonormal columns and R (n x n) upper-triangular.
 *   3. Upload Q, R to device.
 *
 * Supports real fp32 / fp64. Complex left for follow-up.
 * ************************************************************************ */

#include "roctensornet_descriptors.hpp"

#include <cmath>
#include <vector>

using namespace roctensornet;

namespace
{

template <typename T>
void mgs_qr_host(T* A, T* R, size_t m, size_t n)
{
    for(size_t j = 0; j < n; ++j)
    {
        for(size_t i = 0; i < n; ++i) R[i + j * n] = T(0);
    }
    for(size_t j = 0; j < n; ++j)
    {
        for(size_t i = 0; i < j; ++i)
        {
            double dot = 0;
            for(size_t k = 0; k < m; ++k)
                dot += static_cast<double>(A[k + i * m]) * static_cast<double>(A[k + j * m]);
            R[i + j * n] = static_cast<T>(dot);
            for(size_t k = 0; k < m; ++k)
                A[k + j * m] = static_cast<T>(static_cast<double>(A[k + j * m]) - dot * static_cast<double>(A[k + i * m]));
        }
        double nrm = 0;
        for(size_t k = 0; k < m; ++k)
            nrm += static_cast<double>(A[k + j * m]) * static_cast<double>(A[k + j * m]);
        nrm = std::sqrt(nrm);
        R[j + j * n] = static_cast<T>(nrm);
        if(nrm > 0)
            for(size_t k = 0; k < m; ++k)
                A[k + j * m] = static_cast<T>(static_cast<double>(A[k + j * m]) / nrm);
    }
}

template <typename T>
roctensornet_status qr_real_dispatch(const tensor_descriptor_st& tin,  const void* raw_in,
                                     const tensor_descriptor_st& tq,   void*       raw_q,
                                     const tensor_descriptor_st& tr,   void*       raw_r,
                                     hipStream_t                 stream)
{
    size_t n_in = tin.num_elements();
    std::vector<T> host_in(n_in);
    if(hipMemcpyAsync(host_in.data(), raw_in, n_in * sizeof(T), hipMemcpyDeviceToHost, stream) != hipSuccess
       || hipStreamSynchronize(stream) != hipSuccess)
        return ROCTENSORNET_STATUS_EXECUTION_FAILED;

    /* row_modes = modes appearing in tin AND tq, col_modes = appearing in tin AND tr. */
    auto in_set = [&](const std::vector<int32_t>& v, int32_t x) {
        for(auto y : v) if(y == x) return true;
        return false;
    };
    std::vector<int32_t> row_modes, col_modes;
    for(auto m : tq.modes) if(in_set(tin.modes, m)) row_modes.push_back(m);
    for(auto m : tr.modes) if(in_set(tin.modes, m)) col_modes.push_back(m);

    /* Layout to dense matrix. */
    auto extent_of = [&](int32_t m) -> roctensornet_index_t {
        for(size_t i = 0; i < tin.modes.size(); ++i) if(tin.modes[i] == m) return tin.extents[i];
        return 1;
    };
    auto stride_of = [&](int32_t m) -> roctensornet_index_t {
        for(size_t i = 0; i < tin.modes.size(); ++i) if(tin.modes[i] == m) return tin.strides[i];
        return 0;
    };

    size_t m = 1; for(auto mm : row_modes) m *= static_cast<size_t>(extent_of(mm));
    size_t n = 1; for(auto mm : col_modes) n *= static_cast<size_t>(extent_of(mm));
    if(m < n) return ROCTENSORNET_STATUS_NOT_SUPPORTED;

    std::vector<T> A(m * n);
    for(size_t j = 0; j < n; ++j)
    {
        size_t rj = j;
        std::vector<roctensornet_index_t> coord_c(col_modes.size());
        for(size_t k = 0; k < col_modes.size(); ++k)
        {
            roctensornet_index_t e = extent_of(col_modes[k]);
            coord_c[k] = static_cast<roctensornet_index_t>(rj % static_cast<size_t>(e));
            rj /= static_cast<size_t>(e);
        }
        for(size_t i = 0; i < m; ++i)
        {
            size_t ri = i;
            roctensornet_index_t off = 0;
            for(size_t k = 0; k < row_modes.size(); ++k)
            {
                roctensornet_index_t e = extent_of(row_modes[k]);
                roctensornet_index_t v = static_cast<roctensornet_index_t>(ri % static_cast<size_t>(e));
                ri /= static_cast<size_t>(e);
                off += v * stride_of(row_modes[k]);
            }
            for(size_t k = 0; k < col_modes.size(); ++k)
                off += coord_c[k] * stride_of(col_modes[k]);
            A[i + j * m] = host_in[off];
        }
    }
    std::vector<T> R(n * n);
    mgs_qr_host<T>(A.data(), R.data(), m, n);

    if(hipMemcpyAsync(raw_q, A.data(), m * n * sizeof(T), hipMemcpyHostToDevice, stream) != hipSuccess
       || hipMemcpyAsync(raw_r, R.data(), n * n * sizeof(T), hipMemcpyHostToDevice, stream) != hipSuccess
       || hipStreamSynchronize(stream) != hipSuccess)
        return ROCTENSORNET_STATUS_EXECUTION_FAILED;
    return ROCTENSORNET_STATUS_SUCCESS;
}

} // anon

extern "C" {

roctensornet_status
roctensornet_tensor_qr(roctensornet_handle              h,
                       roctensornet_tensor_descriptor   desc_in,
                       const void*                      raw_in,
                       roctensornet_tensor_descriptor   desc_q,
                       void*                            raw_q,
                       roctensornet_tensor_descriptor   desc_r,
                       void*                            raw_r,
                       roctensornet_workspace_descriptor workspace,
                       hipStream_t                      stream)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* hh  = reinterpret_cast<handle*>(h);
    auto* tin = cast<tensor_descriptor_st>(desc_in);
    auto* tq  = cast<tensor_descriptor_st>(desc_q);
    auto* tr  = cast<tensor_descriptor_st>(desc_r);
    if(tin == nullptr || tq == nullptr || tr == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(raw_in);
    ROCTENSORNET_CHECK_PTR(raw_q);
    ROCTENSORNET_CHECK_PTR(raw_r);
    (void)workspace;
    hipStream_t s = stream ? stream : hh->stream;
    switch(tin->data_type)
    {
    case ROCTENSORNET_R_32F: return qr_real_dispatch<float>(*tin, raw_in, *tq, raw_q, *tr, raw_r, s);
    case ROCTENSORNET_R_64F: return qr_real_dispatch<double>(*tin, raw_in, *tq, raw_q, *tr, raw_r, s);
    default: return ROCTENSORNET_STATUS_NOT_SUPPORTED;
    }
}

} // extern "C"
