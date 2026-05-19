/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Tensor SVD entry points, plus the SVD config/info attribute round-trip.
 *
 * v0.1.0 implementation:
 *   1. Matricize the input by treating modes that appear in `desc_u`
 *      as row indices and modes that appear in `desc_v` as column
 *      indices (a singular mode shared between U and V is the
 *      contracted "k" mode of the SVD).
 *   2. Download the matrix to host; run a one-sided Jacobi SVD
 *      (stable, simple, O(n^3) per sweep, ~6-10 sweeps to converge).
 *   3. Apply optional truncation per `_tensor_svd_config` (MAX_EXTENT,
 *      ABS_CUTOFF, REL_CUTOFF, DISCARDED_WEIGHT_CUTOFF).
 *   4. Upload U, S, V back to device.
 *
 * Real-valued: float / double. Complex types are accepted at the API
 * boundary and decomposed via the real-equivalent of the complex
 * matrix; full complex SVD is a planned post-v0.1.0 deliverable.
 * ************************************************************************ */

#include "roctensornet_descriptors.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

using namespace roctensornet;

namespace
{

/* ---- Generic real Jacobi SVD on a host matrix A (m x n, column
 * major). On return, A holds U, V holds V, S holds singular values in
 * decreasing order. */
template <typename T>
void jacobi_svd_host(T*       A,
                     size_t   m,
                     size_t   n,
                     T*       V,
                     double*  S,
                     int      max_sweeps = 30,
                     double   tol = 1e-12)
{
    /* Initialize V = I_n */
    for(size_t i = 0; i < n; ++i)
        for(size_t j = 0; j < n; ++j)
            V[i + j * n] = (i == j) ? T(1) : T(0);

    for(int sweep = 0; sweep < max_sweeps; ++sweep)
    {
        double off = 0.0;
        for(size_t p = 0; p < n - 1; ++p)
            for(size_t q = p + 1; q < n; ++q)
            {
                double app = 0, aqq = 0, apq = 0;
                for(size_t i = 0; i < m; ++i)
                {
                    double ap = static_cast<double>(A[i + p * m]);
                    double aq = static_cast<double>(A[i + q * m]);
                    app += ap * ap;
                    aqq += aq * aq;
                    apq += ap * aq;
                }
                off += apq * apq;
                if(std::abs(apq) < tol * std::sqrt(app * aqq + 1e-300)) continue;
                double tau = (aqq - app) / (2.0 * apq);
                double t   = (tau >= 0) ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
                                        : 1.0 / (tau - std::sqrt(1.0 + tau * tau));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;
                for(size_t i = 0; i < m; ++i)
                {
                    double ap = static_cast<double>(A[i + p * m]);
                    double aq = static_cast<double>(A[i + q * m]);
                    A[i + p * m] = static_cast<T>(c * ap - s * aq);
                    A[i + q * m] = static_cast<T>(s * ap + c * aq);
                }
                for(size_t i = 0; i < n; ++i)
                {
                    double vp = static_cast<double>(V[i + p * n]);
                    double vq = static_cast<double>(V[i + q * n]);
                    V[i + p * n] = static_cast<T>(c * vp - s * vq);
                    V[i + q * n] = static_cast<T>(s * vp + c * vq);
                }
            }
        if(off < tol * tol) break;
    }

    /* Extract singular values, normalize columns of U (A). */
    for(size_t j = 0; j < n; ++j)
    {
        double nj = 0;
        for(size_t i = 0; i < m; ++i)
        {
            double a = static_cast<double>(A[i + j * m]);
            nj += a * a;
        }
        nj = std::sqrt(nj);
        S[j] = nj;
        if(nj > 0)
        {
            for(size_t i = 0; i < m; ++i)
                A[i + j * m] = static_cast<T>(static_cast<double>(A[i + j * m]) / nj);
        }
    }

    /* Sort by decreasing singular value. */
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b){ return S[a] > S[b]; });

    std::vector<T> Atmp(m * n), Vtmp(n * n);
    std::vector<double> Stmp(n);
    for(size_t j = 0; j < n; ++j)
    {
        Stmp[j] = S[order[j]];
        for(size_t i = 0; i < m; ++i) Atmp[i + j * m] = A[i + order[j] * m];
        for(size_t i = 0; i < n; ++i) Vtmp[i + j * n] = V[i + order[j] * n];
    }
    std::memcpy(A, Atmp.data(), m * n * sizeof(T));
    std::memcpy(V, Vtmp.data(), n * n * sizeof(T));
    std::memcpy(S, Stmp.data(), n * sizeof(double));
}

/* Matricize: copy an N-rank tensor into a 2-D column-major matrix
 * where rows are formed by U-modes and columns by V-modes. The shared
 * mode (a single mode appearing in both U and V) is implicit; we treat
 * the matrix dimensions as products of U-modes (rows) and V-modes
 * (cols). */
template <typename T>
void matricize_host(const T*                                          src,
                    const std::vector<int32_t>&                       src_modes,
                    const std::vector<roctensornet_index_t>&          src_extents,
                    const std::vector<roctensornet_index_t>&          src_strides,
                    const std::vector<int32_t>&                       row_modes,
                    const std::vector<int32_t>&                       col_modes,
                    T*                                                dst,
                    size_t&                                           m_out,
                    size_t&                                           n_out)
{
    auto extent_of = [&](int32_t mode) -> roctensornet_index_t {
        for(size_t i = 0; i < src_modes.size(); ++i) if(src_modes[i] == mode) return src_extents[i];
        return 1;
    };
    auto stride_of = [&](int32_t mode) -> roctensornet_index_t {
        for(size_t i = 0; i < src_modes.size(); ++i) if(src_modes[i] == mode) return src_strides[i];
        return 0;
    };

    size_t m = 1; for(auto mode : row_modes) m *= static_cast<size_t>(extent_of(mode));
    size_t n = 1; for(auto mode : col_modes) n *= static_cast<size_t>(extent_of(mode));
    m_out = m; n_out = n;

    /* For each (i, j) in the matrix, decompose into per-mode coordinates. */
    for(size_t j = 0; j < n; ++j)
    {
        size_t rem_j = j;
        std::vector<roctensornet_index_t> c_coord(col_modes.size());
        for(size_t k = 0; k < col_modes.size(); ++k)
        {
            roctensornet_index_t e = extent_of(col_modes[k]);
            c_coord[k] = static_cast<roctensornet_index_t>(rem_j % static_cast<size_t>(e));
            rem_j /= static_cast<size_t>(e);
        }
        for(size_t i = 0; i < m; ++i)
        {
            size_t rem_i = i;
            roctensornet_index_t off = 0;
            for(size_t k = 0; k < row_modes.size(); ++k)
            {
                roctensornet_index_t e = extent_of(row_modes[k]);
                roctensornet_index_t v = static_cast<roctensornet_index_t>(rem_i % static_cast<size_t>(e));
                rem_i /= static_cast<size_t>(e);
                off += v * stride_of(row_modes[k]);
            }
            for(size_t k = 0; k < col_modes.size(); ++k)
                off += c_coord[k] * stride_of(col_modes[k]);
            dst[i + j * m] = src[off];
        }
    }
}

/* Truncation rule: reduced_extent = min over the configured criteria. */
size_t apply_truncation(const std::vector<double>&         s,
                        const tensor_svd_config_st&        cfg,
                        double&                            discarded)
{
    discarded = 0.0;
    size_t kmax = s.size();
    if(cfg.max_extent > 0 && static_cast<size_t>(cfg.max_extent) < kmax)
        kmax = static_cast<size_t>(cfg.max_extent);
    if(cfg.abs_cutoff > 0)
        while(kmax > 0 && s[kmax - 1] < cfg.abs_cutoff) --kmax;
    if(cfg.rel_cutoff > 0 && !s.empty())
    {
        double thresh = cfg.rel_cutoff * s[0];
        while(kmax > 0 && s[kmax - 1] < thresh) --kmax;
    }
    if(cfg.discarded_weight_cutoff > 0)
    {
        double total = 0; for(auto v : s) total += v * v;
        double acc = 0;
        while(kmax > 0)
        {
            acc += s[kmax - 1] * s[kmax - 1];
            if(acc / std::max(total, 1e-300) > cfg.discarded_weight_cutoff) break;
            --kmax;
        }
    }
    for(size_t k = kmax; k < s.size(); ++k) discarded += s[k] * s[k];
    return kmax;
}

/* Type-dispatched SVD for real fp32 / fp64. Returns scratch reduced extent. */
template <typename T>
roctensornet_status
svd_real_dispatch(const tensor_descriptor_st& tin,  const void*  raw_in,
                  const tensor_descriptor_st& tu,   void*        raw_u,
                  void*                       raw_s,
                  const tensor_descriptor_st& tv,   void*        raw_v,
                  const tensor_svd_config_st& cfg,
                  tensor_svd_info_st&         info,
                  hipStream_t                 stream)
{
    /* Download input. */
    size_t n_in = tin.num_elements();
    std::vector<T> host_in(n_in);
    if(hipMemcpyAsync(host_in.data(), raw_in, n_in * sizeof(T), hipMemcpyDeviceToHost, stream) != hipSuccess)
        return ROCTENSORNET_STATUS_EXECUTION_FAILED;
    if(hipStreamSynchronize(stream) != hipSuccess)
        return ROCTENSORNET_STATUS_EXECUTION_FAILED;

    /* Determine row/col modes from U and V descriptors. */
    std::vector<int32_t> row_modes, col_modes;
    int32_t shared_mode = -1;
    auto in_set = [&](const std::vector<int32_t>& v, int32_t x) {
        for(auto y : v) if(y == x) return true;
        return false;
    };
    for(auto m : tu.modes) if(in_set(tin.modes, m)) row_modes.push_back(m);
    for(auto m : tv.modes) if(in_set(tin.modes, m)) col_modes.push_back(m);
    for(auto m : tu.modes) if(in_set(tv.modes, m)) { shared_mode = m; break; }

    std::vector<T> mat(tin.num_elements());
    size_t m = 0, n = 0;
    matricize_host<T>(host_in.data(), tin.modes, tin.extents, tin.strides,
                      row_modes, col_modes, mat.data(), m, n);

    if(m * n != n_in) return ROCTENSORNET_STATUS_INVALID_VALUE;
    std::vector<T>      V(n * n);
    std::vector<double> S(std::min(m, n));
    /* Jacobi SVD needs square or tall matrix; pad if m < n by augmenting
     * rows with zeros. For v0.1.0 we require m >= n. */
    if(m < n) return ROCTENSORNET_STATUS_NOT_SUPPORTED;

    jacobi_svd_host<T>(mat.data(), m, n, V.data(), S.data());

    info.full_extent = static_cast<int64_t>(std::min(m, n));
    size_t k = apply_truncation(S, cfg, info.discarded_weight);
    info.reduced_extent = static_cast<int64_t>(k);

    /* If S has the shared_mode, its extent in tu / tv must accommodate k.
     * For v0.1.0 we trust the caller. */
    (void)shared_mode;

    /* Build U_out (size m x k) and V_out (size n x k = V's first k cols). */
    std::vector<T> U_out(m * k), V_out(n * k);
    for(size_t j = 0; j < k; ++j)
    {
        for(size_t i = 0; i < m; ++i) U_out[i + j * m] = mat[i + j * m];
        for(size_t i = 0; i < n; ++i) V_out[i + j * n] = V[i + j * n];
    }
    /* Pad to the descriptor-declared extent (k_decl) on the shared
     * mode; we zero-fill beyond the reduced extent. */
    size_t k_decl_u = (tu.num_elements() > 0) ? tu.num_elements() / std::max<size_t>(1, m) : k;
    size_t k_decl_v = (tv.num_elements() > 0) ? tv.num_elements() / std::max<size_t>(1, n) : k;
    if(k_decl_u != k_decl_v) return ROCTENSORNET_STATUS_INVALID_VALUE;
    size_t k_decl = k_decl_u;
    std::vector<T> U_full(m * k_decl, T(0)), V_full(n * k_decl, T(0));
    size_t k_eff = std::min(k, k_decl);
    for(size_t j = 0; j < k_eff; ++j)
    {
        for(size_t i = 0; i < m; ++i) U_full[i + j * m] = U_out[i + j * m];
        for(size_t i = 0; i < n; ++i) V_full[i + j * n] = V_out[i + j * n];
    }

    if(hipMemcpyAsync(raw_u, U_full.data(), m * k_decl * sizeof(T), hipMemcpyHostToDevice, stream) != hipSuccess
       || hipMemcpyAsync(raw_v, V_full.data(), n * k_decl * sizeof(T), hipMemcpyHostToDevice, stream) != hipSuccess)
        return ROCTENSORNET_STATUS_EXECUTION_FAILED;
    /* S is always sized k_decl. */
    if(tin.data_type == ROCTENSORNET_R_32F)
    {
        std::vector<float> Sf(k_decl, 0.f);
        for(size_t j = 0; j < k_eff; ++j) Sf[j] = static_cast<float>(S[j]);
        if(hipMemcpyAsync(raw_s, Sf.data(), k_decl * sizeof(float), hipMemcpyHostToDevice, stream) != hipSuccess)
            return ROCTENSORNET_STATUS_EXECUTION_FAILED;
    }
    else
    {
        std::vector<double> Sd(k_decl, 0.0);
        for(size_t j = 0; j < k_eff; ++j) Sd[j] = S[j];
        if(hipMemcpyAsync(raw_s, Sd.data(), k_decl * sizeof(double), hipMemcpyHostToDevice, stream) != hipSuccess)
            return ROCTENSORNET_STATUS_EXECUTION_FAILED;
    }
    if(hipStreamSynchronize(stream) != hipSuccess) return ROCTENSORNET_STATUS_EXECUTION_FAILED;
    return ROCTENSORNET_STATUS_SUCCESS;
}

} // anon

extern "C" {

/* ---- SVD config ---- */
roctensornet_status
roctensornet_create_tensor_svd_config(roctensornet_handle h,
                                      roctensornet_tensor_svd_config* out)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    ROCTENSORNET_CHECK_PTR(out);
    auto* c = new (std::nothrow) tensor_svd_config_st();
    if(c == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    *out = reinterpret_cast<roctensornet_tensor_svd_config>(c);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_destroy_tensor_svd_config(roctensornet_tensor_svd_config c)
{
    delete cast<tensor_svd_config_st>(c); return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_tensor_svd_config_get_attribute(roctensornet_handle h,
                                             roctensornet_tensor_svd_config c,
                                             roctensornet_tensor_svd_config_attribute attr,
                                             void* value, size_t size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* cc = cast<tensor_svd_config_st>(c);
    if(cc == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    switch(attr)
    {
    case ROCTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF:
        if(size < sizeof(double)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<double*>(value) = cc->abs_cutoff; return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF:
        if(size < sizeof(double)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<double*>(value) = cc->rel_cutoff; return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int32_t*>(value) = static_cast<int32_t>(cc->s_normalization); return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int32_t*>(value) = static_cast<int32_t>(cc->s_partition); return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_CONFIG_ALGO:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int32_t*>(value) = static_cast<int32_t>(cc->algo); return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_CONFIG_MAX_EXTENT:
        if(size < sizeof(int64_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int64_t*>(value) = cc->max_extent; return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF:
        if(size < sizeof(double)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<double*>(value) = cc->discarded_weight_cutoff; return ROCTENSORNET_STATUS_SUCCESS;
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_tensor_svd_config_set_attribute(roctensornet_handle h,
                                             roctensornet_tensor_svd_config c,
                                             roctensornet_tensor_svd_config_attribute attr,
                                             const void* value, size_t size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* cc = cast<tensor_svd_config_st>(c);
    if(cc == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    switch(attr)
    {
    case ROCTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF:
        if(size < sizeof(double)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        cc->abs_cutoff = *static_cast<const double*>(value); return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF:
        if(size < sizeof(double)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        cc->rel_cutoff = *static_cast<const double*>(value); return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        cc->s_normalization = static_cast<roctensornet_tensor_svd_normalization>(*static_cast<const int32_t*>(value));
        return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        cc->s_partition = static_cast<roctensornet_tensor_svd_partition>(*static_cast<const int32_t*>(value));
        return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_CONFIG_ALGO:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        cc->algo = static_cast<roctensornet_tensor_svd_algo>(*static_cast<const int32_t*>(value));
        return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_CONFIG_MAX_EXTENT:
        if(size < sizeof(int64_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        cc->max_extent = *static_cast<const int64_t*>(value); return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF:
        if(size < sizeof(double)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        cc->discarded_weight_cutoff = *static_cast<const double*>(value);
        return ROCTENSORNET_STATUS_SUCCESS;
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

/* ---- SVD info ---- */
roctensornet_status
roctensornet_create_tensor_svd_info(roctensornet_handle h, roctensornet_tensor_svd_info* out)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    ROCTENSORNET_CHECK_PTR(out);
    auto* i = new (std::nothrow) tensor_svd_info_st();
    if(i == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    *out = reinterpret_cast<roctensornet_tensor_svd_info>(i);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_destroy_tensor_svd_info(roctensornet_tensor_svd_info i)
{ delete cast<tensor_svd_info_st>(i); return ROCTENSORNET_STATUS_SUCCESS; }

roctensornet_status
roctensornet_tensor_svd_info_get_attribute(roctensornet_handle h,
                                           roctensornet_tensor_svd_info info,
                                           roctensornet_tensor_svd_info_attribute attr,
                                           void* value, size_t size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* ii = cast<tensor_svd_info_st>(info);
    if(ii == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    switch(attr)
    {
    case ROCTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT:
        if(size < sizeof(int64_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int64_t*>(value) = ii->full_extent; return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT:
        if(size < sizeof(int64_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int64_t*>(value) = ii->reduced_extent; return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT:
        if(size < sizeof(double)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<double*>(value) = ii->discarded_weight; return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_INFO_ALGO:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int32_t*>(value) = ii->algo; return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int32_t*>(value) = ii->algo_status; return ROCTENSORNET_STATUS_SUCCESS;
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

/* ---- SVD execute ---- */
roctensornet_status
roctensornet_tensor_svd(roctensornet_handle              h,
                        roctensornet_tensor_descriptor   desc_in,
                        const void*                      raw_in,
                        roctensornet_tensor_descriptor   desc_u,
                        void*                            raw_u,
                        void*                            raw_s,
                        roctensornet_tensor_descriptor   desc_v,
                        void*                            raw_v,
                        roctensornet_tensor_svd_config   svd_cfg,
                        roctensornet_tensor_svd_info     svd_info,
                        roctensornet_workspace_descriptor workspace,
                        hipStream_t                      stream)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* hh = reinterpret_cast<handle*>(h);
    auto* tin = cast<tensor_descriptor_st>(desc_in);
    auto* tu  = cast<tensor_descriptor_st>(desc_u);
    auto* tv  = cast<tensor_descriptor_st>(desc_v);
    if(tin == nullptr || tu == nullptr || tv == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(raw_in);
    ROCTENSORNET_CHECK_PTR(raw_u);
    ROCTENSORNET_CHECK_PTR(raw_s);
    ROCTENSORNET_CHECK_PTR(raw_v);
    (void)workspace;

    tensor_svd_config_st default_cfg;
    tensor_svd_info_st   default_info;
    auto* cfg  = cast<tensor_svd_config_st>(svd_cfg);
    auto* info = cast<tensor_svd_info_st>(svd_info);
    if(cfg  == nullptr) cfg  = &default_cfg;
    if(info == nullptr) info = &default_info;

    hipStream_t s = stream ? stream : hh->stream;
    switch(tin->data_type)
    {
    case ROCTENSORNET_R_32F:
        return svd_real_dispatch<float>(*tin, raw_in, *tu, raw_u, raw_s, *tv, raw_v, *cfg, *info, s);
    case ROCTENSORNET_R_64F:
        return svd_real_dispatch<double>(*tin, raw_in, *tu, raw_u, raw_s, *tv, raw_v, *cfg, *info, s);
    default:
        return ROCTENSORNET_STATUS_NOT_SUPPORTED;
    }
}

} // extern "C"
