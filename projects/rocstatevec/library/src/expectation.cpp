/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * `compute_expectation` — <psi|M|psi> for a dense matrix on chosen qubits.
 *
 * Algorithm: one thread per amplitude `i` in the state vector. The
 * thread:
 *   1. Builds row_idx = bits of `i` at the basis-bit positions.
 *   2. For each column `j in [0, m)`, gathers amp_j by replacing the
 *      basis bits of `i` with bits of j.
 *   3. Computes conj(sv[i]) * sum_j M[row, col] * amp_j (or its
 *      adjoint variant) and atomic-adds into a 2-double accumulator.
 *
 * Output is read back to host and written into `expectation_value` in
 * the requested precision.
 *
 * `residual_norm` reports the Frobenius distance between `M` and its
 * Hermitian projection 0.5 * (M + M^H).
 *
 * Workspace allocations route through `dev_arena` so they honor the
 * caller-provided workspace and the handle's device-mem handler. The
 * matrix may be host or device memory; `is_device_pointer` selects the
 * code path automatically.
 * ************************************************************************ */

#include "rocstatevec_kernels.hpp"

#include <cmath>
#include <vector>

namespace rocstatevec
{

template <typename C>
__global__ void k_compute_expectation(
    const C* sv, rocstatevec_index_t N,
    const C* matrix, uint32_t n_basis_bits, int row_major,
    const int32_t* basis_bits, rocstatevec_index_t basis_mask,
    double* d_acc /* size 2: real, imag */)
{
    using T = complex_traits<C>;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    uint32_t m   = 1u << n_basis_bits;
    uint32_t row = 0;
    rocstatevec_index_t base = i & ~basis_mask;
    for(uint32_t b = 0; b < n_basis_bits; ++b)
        if(((i >> basis_bits[b]) & 1) != 0) row |= (1u << b);

    C accum = T::zero();
    for(uint32_t j = 0; j < m; ++j)
    {
        rocstatevec_index_t in_idx = base;
        for(uint32_t b = 0; b < n_basis_bits; ++b)
            if(((j >> b) & 1) != 0) in_idx |= (rocstatevec_index_t{1} << basis_bits[b]);

        C M = row_major ? matrix[row * m + j] : matrix[j * m + row];
        accum = T::add(accum, T::mul(M, sv[in_idx]));
    }
    C term = T::mul(T::conj(sv[i]), accum);
    atomicAdd(d_acc + 0, static_cast<double>(T::real(term)));
    atomicAdd(d_acc + 1, static_cast<double>(T::imag(term)));
}

/*! \brief Compute || M - 0.5*(M + M^H) ||_F over a host-resident matrix.
 *  Matrix dimension m = 2^n_basis_bits is small so the host loop is
 *  cheap and avoids extra kernel infrastructure. */
template <typename C>
static double host_hermitian_residual(const void* host_matrix,
                                      rocstatevec_matrix_layout layout,
                                      uint32_t n_basis_bits)
{
    using T   = complex_traits<C>;
    uint32_t m  = 1u << n_basis_bits;
    auto idx = [&](uint32_t r, uint32_t c) -> size_t {
        return (layout == ROCSTATEVEC_MATRIX_LAYOUT_ROW) ? size_t(r) * m + c
                                                          : size_t(c) * m + r;
    };
    const auto* mat = reinterpret_cast<const C*>(host_matrix);
    double sum_sq = 0.0;
    for(uint32_t r = 0; r < m; ++r)
    {
        for(uint32_t c = 0; c < m; ++c)
        {
            // anti-Hermitian part = 0.5 * (M - M^H)
            C a = mat[idx(r, c)];
            C b = T::conj(mat[idx(c, r)]);
            C diff = T::sub(a, b);
            double re = 0.5 * static_cast<double>(T::real(diff));
            double im = 0.5 * static_cast<double>(T::imag(diff));
            sum_sq += re * re + im * im;
        }
    }
    return std::sqrt(sum_sq);
}

template <typename C>
static rocstatevec_status compute_expectation_dispatch(
    rocstatevec_handle h_in, const void* dsv, uint32_t n,
    void* expectation_value, rocstatevec_data_type e_dt,
    const void* matrix, rocstatevec_matrix_layout layout,
    const int32_t* basis_bits, uint32_t n_basis_bits,
    void* user_ws, size_t user_ws_bytes)
{
    auto*               hh   = reinterpret_cast<handle*>(h_in);
    rocstatevec_index_t N    = state_vector_length(n);
    hipStream_t         s    = hh->stream;
    int                 tpb  = default_threads_per_block;
    int                 gpc  = static_cast<int>(ceil_div<rocstatevec_index_t>(N, tpb));
    uint32_t            m    = 1u << n_basis_bits;

    rocstatevec_index_t basis_mask = qubit_mask(basis_bits, n_basis_bits);

    dev_arena arena(hh, s, user_ws, user_ws_bytes);

    int32_t* d_basis = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc_and_copy(reinterpret_cast<void**>(&d_basis),
                                              basis_bits,
                                              n_basis_bits * sizeof(int32_t)));

    const C* d_matrix = nullptr;
    if(is_device_pointer(matrix))
    {
        d_matrix = reinterpret_cast<const C*>(matrix);
    }
    else
    {
        void* tmp = nullptr;
        ROCSTATEVEC_RC_CHECK(arena.alloc_and_copy(&tmp, matrix,
                                                  size_t(m) * m * sizeof(C)));
        d_matrix = reinterpret_cast<const C*>(tmp);
    }

    void* d_acc_v = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc(&d_acc_v, 2 * sizeof(double)));
    auto* d_acc = reinterpret_cast<double*>(d_acc_v);
    ROCSTATEVEC_HIP_CHECK(hipMemsetAsync(d_acc, 0, 2 * sizeof(double), s));

    hipLaunchKernelGGL(k_compute_expectation<C>, dim3(gpc), dim3(tpb), 0, s,
                       reinterpret_cast<const C*>(dsv), N,
                       d_matrix, n_basis_bits,
                       layout == ROCSTATEVEC_MATRIX_LAYOUT_ROW ? 1 : 0,
                       d_basis, basis_mask, d_acc);

    double host_acc[2] = {0, 0};
    ROCSTATEVEC_HIP_CHECK(hipMemcpyAsync(host_acc, d_acc, 2 * sizeof(double),
                                         hipMemcpyDeviceToHost, s));
    ROCSTATEVEC_HIP_CHECK(hipStreamSynchronize(s));

    switch(e_dt)
    {
    case ROCSTATEVEC_C_64F: {
        auto* o = reinterpret_cast<double*>(expectation_value);
        o[0] = host_acc[0]; o[1] = host_acc[1]; break;
    }
    case ROCSTATEVEC_C_32F: {
        auto* o = reinterpret_cast<float*>(expectation_value);
        o[0] = static_cast<float>(host_acc[0]); o[1] = static_cast<float>(host_acc[1]); break;
    }
    case ROCSTATEVEC_R_64F:
        *reinterpret_cast<double*>(expectation_value) = host_acc[0]; break;
    case ROCSTATEVEC_R_32F:
        *reinterpret_cast<float*>(expectation_value) = static_cast<float>(host_acc[0]); break;
    default:
        return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
    }
    return ROCSTATEVEC_STATUS_SUCCESS;
}

static size_t expectation_workspace_bytes(rocstatevec_data_type sv_dt,
                                          uint32_t              n_basis_bits)
{
    size_t es        = element_size_bytes(sv_dt);
    size_t basis_b   = align_up(size_t(n_basis_bits) * sizeof(int32_t),
                                default_workspace_align);
    size_t m         = size_t{1} << n_basis_bits;
    size_t matrix_b  = align_up(m * m * es, default_workspace_align);
    size_t acc_b     = align_up(2 * sizeof(double), default_workspace_align);
    return basis_b + matrix_b + acc_b;
}

} // namespace rocstatevec

extern "C" rocstatevec_status rocstatevec_compute_expectation_get_workspace_size(
    rocstatevec_handle, rocstatevec_data_type sv_dtype, uint32_t /*n*/,
    const void*, rocstatevec_data_type, rocstatevec_matrix_layout,
    uint32_t n_basis_bits,
    rocstatevec_compute_type, size_t* extra_workspace_size_in_bytes)
{
    using namespace rocstatevec;
    if(extra_workspace_size_in_bytes == nullptr) return ROCSTATEVEC_STATUS_SUCCESS;
    if(sv_dtype != ROCSTATEVEC_C_64F && sv_dtype != ROCSTATEVEC_C_32F)
    {
        *extra_workspace_size_in_bytes = 0;
        return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
    }
    if(n_basis_bits == 0 || n_basis_bits > 10)
    {
        *extra_workspace_size_in_bytes = 0;
        return ROCSTATEVEC_STATUS_INVALID_VALUE;
    }
    *extra_workspace_size_in_bytes
        = expectation_workspace_bytes(sv_dtype, n_basis_bits);
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_compute_expectation(
    rocstatevec_handle h, const void* state_vector,
    rocstatevec_data_type sv_dt, uint32_t n,
    void* expectation_value, rocstatevec_data_type e_dt, double* residual_norm,
    const void* matrix, rocstatevec_data_type m_dt, rocstatevec_matrix_layout layout,
    const int32_t* basis_bits, uint32_t n_basis_bits,
    rocstatevec_compute_type, void* extra_workspace, size_t extra_workspace_size_in_bytes)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(state_vector); ROCSTATEVEC_CHECK_PTR(expectation_value);
    ROCSTATEVEC_CHECK_PTR(matrix);
    if(n_basis_bits == 0 || n_basis_bits > 10) return ROCSTATEVEC_STATUS_INVALID_VALUE;
    ROCSTATEVEC_CHECK_PTR(basis_bits);
    if(sv_dt != m_dt) return ROCSTATEVEC_STATUS_NOT_SUPPORTED;

    if(residual_norm)
    {
        // Compute on a host-resident copy of the matrix; download if
        // necessary, using the handle-bound stream so we do not bypass
        // the user's stream ordering.
        if(is_device_pointer(matrix))
        {
            auto*                hh = reinterpret_cast<handle*>(h);
            size_t               m  = size_t{1} << n_basis_bits;
            size_t               es = element_size_bytes(sv_dt);
            std::vector<uint8_t> host_buf(m * m * es);
            if(hipMemcpyAsync(host_buf.data(), matrix, m * m * es,
                              hipMemcpyDeviceToHost, hh->stream) != hipSuccess
               || hipStreamSynchronize(hh->stream) != hipSuccess)
            {
                return ROCSTATEVEC_STATUS_EXECUTION_FAILED;
            }
            *residual_norm = (sv_dt == ROCSTATEVEC_C_64F)
                                 ? host_hermitian_residual<c64>(host_buf.data(), layout, n_basis_bits)
                                 : host_hermitian_residual<c32>(host_buf.data(), layout, n_basis_bits);
        }
        else
        {
            *residual_norm = (sv_dt == ROCSTATEVEC_C_64F)
                                 ? host_hermitian_residual<c64>(matrix, layout, n_basis_bits)
                                 : host_hermitian_residual<c32>(matrix, layout, n_basis_bits);
        }
    }

    if(sv_dt == ROCSTATEVEC_C_64F)
        return compute_expectation_dispatch<c64>(h, state_vector, n, expectation_value, e_dt,
                                                 matrix, layout, basis_bits, n_basis_bits,
                                                 extra_workspace, extra_workspace_size_in_bytes);
    if(sv_dt == ROCSTATEVEC_C_32F)
        return compute_expectation_dispatch<c32>(h, state_vector, n, expectation_value, e_dt,
                                                 matrix, layout, basis_bits, n_basis_bits,
                                                 extra_workspace, extra_workspace_size_in_bytes);
    return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
}
