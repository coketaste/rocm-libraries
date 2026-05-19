/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Sampler descriptor + preprocess + sample.
 *
 * Algorithm:
 *   1. Preprocess: compute |amp[i]|^2 into a device array of doubles, run
 *      rocPRIM inclusive-scan to produce a cumulative-probability table;
 *      record the total norm as `cum[N-1]`.
 *   2. Sample: for each shot s, scale randnum[s] by the total norm to get
 *      a target, binary-search the cumulative table for the smallest k
 *      with cum[k] >= target, then project k onto the bit_ordering subset
 *      to produce the output bit-string.
 *
 *   The sampler stores the cumulative table on the heap; it is freed by
 *   `rocstatevec_sampler_destroy`. The workspace size returned by
 *   `sampler_create` is 0 — the cumulative table is allocated internally
 *   by `sampler_preprocess`. A future revision can switch to user-managed
 *   workspace once the sampler grows hot.
 * ************************************************************************ */

#include "rocstatevec_kernels.hpp"

#include <rocprim/rocprim.hpp>

namespace rocstatevec
{

struct sampler_descriptor
{
    const void*           sv           = nullptr;
    rocstatevec_data_type dtype        = ROCSTATEVEC_C_64F;
    uint32_t              n_index_bits = 0;
    uint32_t              n_max_shots  = 0;

    double*               d_cum        = nullptr;
    size_t                d_cum_bytes  = 0;
    handle*               h_cum        = nullptr; // owning handle for d_cum cleanup
    double                total_norm   = 0.0;
    double                offset       = 0.0;
    bool                  preprocessed = false;
};

template <typename C>
__global__ void k_abs2_into_double(const C* sv, rocstatevec_index_t N, double* out)
{
    using T = complex_traits<C>;
    rocstatevec_index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) out[i] = static_cast<double>(T::abs2(sv[i]));
}

__global__ void k_search_and_project(
    const double* cum, rocstatevec_index_t N, double total,
    const double* randnums, uint32_t n_shots,
    const int32_t* bit_ordering, uint32_t bs_len,
    rocstatevec_index_t* out_bits, double offset)
{
    uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= n_shots) return;

    double target = (randnums[s] - offset) * total;
    if(target < 0.0)   target = 0.0;
    if(target > total) target = total;

    // Lower-bound binary search for smallest k where cum[k] >= target.
    rocstatevec_index_t lo = 0, hi = N - 1;
    while(lo < hi)
    {
        rocstatevec_index_t mid = lo + (hi - lo) / 2;
        if(cum[mid] < target) lo = mid + 1;
        else                  hi = mid;
    }

    rocstatevec_index_t k   = lo;
    rocstatevec_index_t out = 0;
    for(uint32_t b = 0; b < bs_len; ++b)
        out |= (((k >> bit_ordering[b]) & 1) << b);
    out_bits[s] = out;
}

template <typename C>
static rocstatevec_status preprocess_dispatch(rocstatevec_handle h_in, sampler_descriptor& sd)
{
    auto*               hh   = reinterpret_cast<handle*>(h_in);
    rocstatevec_index_t N    = state_vector_length(sd.n_index_bits);
    hipStream_t         s    = hh->stream;
    int                 tpb  = default_threads_per_block;
    int                 gpc  = static_cast<int>(ceil_div<rocstatevec_index_t>(N, tpb));

    size_t cum_bytes = size_t(N) * sizeof(double);
    if(sd.d_cum == nullptr)
    {
        void* p = nullptr;
        ROCSTATEVEC_RC_CHECK(dev_alloc(hh, &p, cum_bytes, s));
        sd.d_cum       = reinterpret_cast<double*>(p);
        sd.d_cum_bytes = cum_bytes;
        sd.h_cum       = hh;
    }

    hipLaunchKernelGGL(k_abs2_into_double<C>, dim3(gpc), dim3(tpb), 0, s,
                       reinterpret_cast<const C*>(sd.sv), N, sd.d_cum);

    dev_arena arena(hh, s);
    size_t    scan_temp_bytes = 0;
    rocprim::inclusive_scan(nullptr, scan_temp_bytes,
                            sd.d_cum, sd.d_cum, size_t(N), rocprim::plus<double>(), s);
    void* scan_temp = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc(&scan_temp, scan_temp_bytes));
    rocprim::inclusive_scan(scan_temp, scan_temp_bytes,
                            sd.d_cum, sd.d_cum, size_t(N), rocprim::plus<double>(), s);

    double host_total = 0.0;
    ROCSTATEVEC_HIP_CHECK(hipMemcpyAsync(&host_total, sd.d_cum + (N - 1), sizeof(double),
                                         hipMemcpyDeviceToHost, s));
    ROCSTATEVEC_HIP_CHECK(hipStreamSynchronize(s));
    sd.total_norm   = host_total;
    sd.preprocessed = true;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

} // namespace rocstatevec

extern "C" rocstatevec_status rocstatevec_sampler_create(
    rocstatevec_handle h, const void* state_vector, rocstatevec_data_type dtype,
    uint32_t n_index_bits, rocstatevec_sampler_descriptor* out_sampler,
    uint32_t n_max_shots, size_t* extra_workspace_size_in_bytes)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(state_vector); ROCSTATEVEC_CHECK_PTR(out_sampler);
    if(dtype != ROCSTATEVEC_C_64F && dtype != ROCSTATEVEC_C_32F)
        return ROCSTATEVEC_STATUS_NOT_SUPPORTED;

    auto* sd = new(std::nothrow) sampler_descriptor{};
    if(sd == nullptr) return ROCSTATEVEC_STATUS_ALLOC_FAILED;
    sd->sv           = state_vector;
    sd->dtype        = dtype;
    sd->n_index_bits = n_index_bits;
    sd->n_max_shots  = n_max_shots;
    *out_sampler     = reinterpret_cast<rocstatevec_sampler_descriptor>(sd);
    if(extra_workspace_size_in_bytes) *extra_workspace_size_in_bytes = 0;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_sampler_destroy(rocstatevec_sampler_descriptor sampler)
{
    using namespace rocstatevec;
    if(sampler == nullptr) return ROCSTATEVEC_STATUS_SUCCESS;
    auto* sd = reinterpret_cast<sampler_descriptor*>(sampler);
    if(sd->d_cum != nullptr)
    {
        hipStream_t s = sd->h_cum ? sd->h_cum->stream : hipStream_t{nullptr};
        (void)dev_free(sd->h_cum, sd->d_cum, sd->d_cum_bytes, s);
    }
    delete sd;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_sampler_preprocess(
    rocstatevec_handle h, rocstatevec_sampler_descriptor sampler,
    void* /*ws*/, size_t /*ws_bytes*/)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(sampler);
    auto& sd = *reinterpret_cast<sampler_descriptor*>(sampler);

    if(sd.dtype == ROCSTATEVEC_C_64F) return preprocess_dispatch<c64>(h, sd);
    if(sd.dtype == ROCSTATEVEC_C_32F) return preprocess_dispatch<c32>(h, sd);
    return ROCSTATEVEC_STATUS_NOT_SUPPORTED;
}

extern "C" rocstatevec_status rocstatevec_sampler_get_squared_norm(
    rocstatevec_handle h, rocstatevec_sampler_descriptor sampler, double* norm)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(sampler); ROCSTATEVEC_CHECK_PTR(norm);
    auto& sd = *reinterpret_cast<sampler_descriptor*>(sampler);
    if(!sd.preprocessed) return ROCSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED;
    *norm = sd.total_norm;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_sampler_apply_sub_sv_offset(
    rocstatevec_handle h, rocstatevec_sampler_descriptor sampler,
    int32_t /*sub_sv_index*/, uint32_t /*n_sub_svs*/, double offset, double norm)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(sampler);
    auto& sd = *reinterpret_cast<sampler_descriptor*>(sampler);
    sd.offset     = offset;
    sd.total_norm = norm;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_sampler_sample(
    rocstatevec_handle h, rocstatevec_sampler_descriptor sampler,
    rocstatevec_index_t* bit_strings, const int32_t* bit_ordering,
    uint32_t bs_len, const double* randnums, uint32_t n_shots,
    rocstatevec_sampler_output /*output*/)
{
    using namespace rocstatevec;
    ROCSTATEVEC_CHECK_HANDLE(h);
    ROCSTATEVEC_CHECK_PTR(sampler);
    ROCSTATEVEC_CHECK_PTR(bit_strings); ROCSTATEVEC_CHECK_PTR(bit_ordering);
    ROCSTATEVEC_CHECK_PTR(randnums);
    if(bs_len == 0 || n_shots == 0) return ROCSTATEVEC_STATUS_INVALID_VALUE;

    auto& sd = *reinterpret_cast<sampler_descriptor*>(sampler);
    if(!sd.preprocessed) return ROCSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED;

    auto*               hh   = reinterpret_cast<handle*>(h);
    rocstatevec_index_t N    = state_vector_length(sd.n_index_bits);
    hipStream_t         s    = hh->stream;
    int                 tpb  = default_threads_per_block;
    int                 gpc  = static_cast<int>(ceil_div<uint32_t>(n_shots, tpb));

    dev_arena arena(hh, s);

    int32_t* d_bo = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc_and_copy(reinterpret_cast<void**>(&d_bo),
                                              bit_ordering, bs_len * sizeof(int32_t)));

    double* d_rand = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc_and_copy(reinterpret_cast<void**>(&d_rand),
                                              randnums, n_shots * sizeof(double)));

    void* d_out_v = nullptr;
    ROCSTATEVEC_RC_CHECK(arena.alloc(&d_out_v, n_shots * sizeof(rocstatevec_index_t)));
    auto* d_out = reinterpret_cast<rocstatevec_index_t*>(d_out_v);

    hipLaunchKernelGGL(k_search_and_project, dim3(gpc), dim3(tpb), 0, s,
                       sd.d_cum, N, sd.total_norm, d_rand, n_shots,
                       d_bo, bs_len, d_out, sd.offset);

    ROCSTATEVEC_HIP_CHECK(hipMemcpyAsync(bit_strings, d_out,
                                         n_shots * sizeof(rocstatevec_index_t),
                                         hipMemcpyDeviceToHost, s));
    ROCSTATEVEC_HIP_CHECK(hipStreamSynchronize(s));
    return ROCSTATEVEC_STATUS_SUCCESS;
}
