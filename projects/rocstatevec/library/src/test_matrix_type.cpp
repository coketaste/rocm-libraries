/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * `test_matrix_type` — return a residual norm that is zero when the
 * matrix exhibits the requested property and grows monotonically as the
 * deviation increases.
 *
 *   * GENERAL   -> always returns 0.0
 *   * UNITARY   -> ||M^H M - I||_F
 *   * HERMITIAN -> ||M - M^H||_F
 *
 * Matrix dimension m = 2^n_targets is small (we cap at 16 targets) so
 * the computation runs on the host. The matrix is auto-detected as host
 * or device memory; device-resident matrices are first downloaded into
 * a temporary host buffer using the bound stream.
 *
 * The user-provided workspace is unused (no GPU work is done here);
 * `*_get_workspace_size` returns 0 to reflect that.
 * ************************************************************************ */

#include "rocstatevec_kernels.hpp"

#include <cmath>
#include <vector>

namespace rocstatevec
{

template <typename C>
static rocstatevec_status test_dispatch(
    rocstatevec_matrix_type kind, const void* h_matrix, rocstatevec_matrix_layout layout,
    uint32_t n_targets, int adjoint, double* out)
{
    using T  = complex_traits<C>;
    uint32_t m  = 1u << n_targets;
    auto idx = [&](uint32_t r, uint32_t c) -> size_t {
        return (layout == ROCSTATEVEC_MATRIX_LAYOUT_ROW) ? size_t(r) * m + c
                                                          : size_t(c) * m + r;
    };
    const auto* mat = reinterpret_cast<const C*>(h_matrix);

    auto get = [&](uint32_t r, uint32_t c) -> C {
        return adjoint ? T::conj(mat[idx(c, r)]) : mat[idx(r, c)];
    };

    double sum_sq = 0.0;
    if(kind == ROCSTATEVEC_MATRIX_TYPE_HERMITIAN)
    {
        for(uint32_t r = 0; r < m; ++r)
            for(uint32_t c = 0; c < m; ++c)
            {
                C a   = get(r, c);
                C b   = T::conj(get(c, r));
                C dif = T::sub(a, b);
                sum_sq += static_cast<double>(T::abs2(dif));
            }
    }
    else if(kind == ROCSTATEVEC_MATRIX_TYPE_UNITARY)
    {
        for(uint32_t r = 0; r < m; ++r)
        {
            for(uint32_t c = 0; c < m; ++c)
            {
                C acc = T::zero();
                for(uint32_t k = 0; k < m; ++k)
                {
                    C a = T::conj(get(k, r));
                    C b = get(k, c);
                    acc = T::add(acc, T::mul(a, b));
                }
                if(r == c)
                {
                    C dif = T::sub(acc, T::one());
                    sum_sq += static_cast<double>(T::abs2(dif));
                }
                else
                {
                    sum_sq += static_cast<double>(T::abs2(acc));
                }
            }
        }
    }
    *out = std::sqrt(sum_sq);
    return ROCSTATEVEC_STATUS_SUCCESS;
}

} // namespace rocstatevec

extern "C" rocstatevec_status rocstatevec_test_matrix_type_get_workspace_size(
    rocstatevec_handle, rocstatevec_matrix_type, const void*, rocstatevec_data_type,
    rocstatevec_matrix_layout, uint32_t, int32_t, rocstatevec_compute_type,
    size_t* extra_workspace_size_in_bytes)
{
    if(extra_workspace_size_in_bytes) *extra_workspace_size_in_bytes = 0;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_test_matrix_type(
    rocstatevec_handle h, double* residual_norm, rocstatevec_matrix_type kind,
    const void* matrix, rocstatevec_data_type dtype, rocstatevec_matrix_layout layout,
    uint32_t n_targets, int32_t adjoint, rocstatevec_compute_type,
    void* /*ws*/, size_t /*ws_bytes*/)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(residual_norm);
    if(n_targets == 0 || n_targets > 16) return ROCSTATEVEC_STATUS_INVALID_VALUE;

    if(kind == ROCSTATEVEC_MATRIX_TYPE_GENERAL)
    {
        *residual_norm = 0.0;
        return ROCSTATEVEC_STATUS_SUCCESS;
    }
    ROCSTATEVEC_CHECK_PTR(matrix);
    if(dtype != ROCSTATEVEC_C_64F && dtype != ROCSTATEVEC_C_32F)
        return ROCSTATEVEC_STATUS_NOT_SUPPORTED;

    // If the matrix is device-resident, download once into a host buffer
    // so the host-side norm computation can iterate it directly.
    std::vector<uint8_t> host_buf;
    const void*          host_matrix = matrix;
    if(is_device_pointer(matrix))
    {
        size_t      m  = size_t{1} << n_targets;
        size_t      es = element_size_bytes(dtype);
        size_t      sz = m * m * es;
        host_buf.resize(sz);
        auto* hh = reinterpret_cast<handle*>(h);
        if(hipMemcpyAsync(host_buf.data(), matrix, sz, hipMemcpyDeviceToHost, hh->stream)
           != hipSuccess)
        {
            return ROCSTATEVEC_STATUS_EXECUTION_FAILED;
        }
        if(hipStreamSynchronize(hh->stream) != hipSuccess)
        {
            return ROCSTATEVEC_STATUS_EXECUTION_FAILED;
        }
        host_matrix = host_buf.data();
    }

    if(dtype == ROCSTATEVEC_C_64F)
        return test_dispatch<c64>(kind, host_matrix, layout, n_targets, adjoint, residual_norm);
    if(dtype == ROCSTATEVEC_C_32F)
        return test_dispatch<c32>(kind, host_matrix, layout, n_targets, adjoint, residual_norm);
    return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
}
