/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Workspace descriptor.
 *
 * The descriptor stores per-(memspace, kind) slots. `compute_*_sizes`
 * walks the call's operands to derive an honest byte count; the
 * executor (`contraction_execute.cpp` and `tensor_svd.cpp`) consumes
 * the slot via a `dev_arena` for the SCRATCH kind, and a long-lived
 * persistent allocator for the CACHE kind.
 * ************************************************************************ */

#include "roctensornet_descriptors.hpp"

#include <algorithm>

using namespace roctensornet;

namespace
{

/* Compute the scratch + cache byte requirement for a contraction plan
 * derived from `(network, info)`. Scratch ~= 2 * largest pair's
 * intermediate (two ping-pong buffers); cache ~= sum of intermediate
 * bytes across the path. */
void compute_contraction_sizes(const network_descriptor_st&         nd,
                               const contraction_optimizer_info_st& info,
                               int64_t&                             out_scratch,
                               int64_t&                             out_cache)
{
    size_t scratch = 0, cache = 0;
    for(const auto& n : info.path)
    {
        if(n.intermediate_bytes > scratch) scratch = n.intermediate_bytes;
        cache += n.intermediate_bytes;
    }
    /* Plus the largest input as staging buffer for permutation. */
    size_t largest_input = 0;
    for(const auto& td : nd.inputs)
        largest_input = std::max(largest_input, td.num_bytes());
    scratch = std::max(scratch, largest_input);
    /* Ping-pong factor. */
    scratch *= 2;
    /* Honor the workspace alignment. */
    scratch = align_up(scratch, default_workspace_align);
    cache   = align_up(cache,   default_workspace_align);
    out_scratch = static_cast<int64_t>(scratch);
    out_cache   = static_cast<int64_t>(cache);
}

void compute_svd_sizes(const tensor_descriptor_st& tin,
                       const tensor_descriptor_st& tu,
                       const tensor_descriptor_st& tv,
                       int64_t&                    out_scratch)
{
    /* Need scratch to hold a matricized copy + singular-value vector +
     * O(m*n) workspace for gesvd in single precision. We size for the
     * complex case; that bounds the real case. */
    size_t m = 1; for(auto e : tu.extents) m *= static_cast<size_t>(e);
    size_t n = 1; for(auto e : tv.extents) n *= static_cast<size_t>(e);
    size_t es = element_size_bytes(tin.data_type);
    size_t mat = m * n * es;
    size_t sng = std::min(m, n) * sizeof(double);
    size_t lwork = 5 * std::min(m, n) * es;
    out_scratch = static_cast<int64_t>(align_up(mat * 3 + sng + lwork,
                                                default_workspace_align));
}

void compute_qr_sizes(const tensor_descriptor_st& tin,
                      const tensor_descriptor_st& tq,
                      const tensor_descriptor_st& tr,
                      int64_t&                    out_scratch)
{
    size_t m = 1; for(auto e : tq.extents) m *= static_cast<size_t>(e);
    size_t n = 1; for(auto e : tr.extents) n *= static_cast<size_t>(e);
    size_t es = element_size_bytes(tin.data_type);
    size_t mat = m * n * es;
    size_t tau = std::min(m, n) * es;
    size_t lwork = 2 * std::min(m, n) * es;
    out_scratch = static_cast<int64_t>(align_up(mat * 2 + tau + lwork,
                                                default_workspace_align));
}

} // anon

extern "C" {

roctensornet_status
roctensornet_create_workspace_descriptor(roctensornet_handle handle,
                                         roctensornet_workspace_descriptor* out)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    ROCTENSORNET_CHECK_PTR(out);
    auto* w = new (std::nothrow) workspace_descriptor_st();
    if(w == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    *out = reinterpret_cast<roctensornet_workspace_descriptor>(w);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_destroy_workspace_descriptor(roctensornet_workspace_descriptor d)
{
    delete cast<workspace_descriptor_st>(d);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_workspace_compute_contraction_sizes(
    roctensornet_handle                     handle,
    roctensornet_network_descriptor         network_desc,
    roctensornet_contraction_optimizer_info info,
    roctensornet_workspace_descriptor       workspace_desc)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    auto* nd = cast<network_descriptor_st>(network_desc);
    auto* ii = cast<contraction_optimizer_info_st>(info);
    auto* w  = cast<workspace_descriptor_st>(workspace_desc);
    if(nd == nullptr || ii == nullptr || w == nullptr)
        return ROCTENSORNET_STATUS_NOT_INITIALIZED;

    int64_t scratch = 0, cache = 0;
    compute_contraction_sizes(*nd, *ii, scratch, cache);
    w->device_scratch.required = scratch;
    w->device_cache.required   = cache;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_workspace_compute_svd_sizes(
    roctensornet_handle              handle,
    roctensornet_tensor_descriptor   desc_in,
    roctensornet_tensor_descriptor   desc_u,
    roctensornet_tensor_descriptor   desc_v,
    roctensornet_tensor_svd_config   svd_config,
    roctensornet_workspace_descriptor workspace_desc)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    (void)svd_config;
    auto* tin = cast<tensor_descriptor_st>(desc_in);
    auto* tu  = cast<tensor_descriptor_st>(desc_u);
    auto* tv  = cast<tensor_descriptor_st>(desc_v);
    auto* w   = cast<workspace_descriptor_st>(workspace_desc);
    if(tin == nullptr || tu == nullptr || tv == nullptr || w == nullptr)
        return ROCTENSORNET_STATUS_NOT_INITIALIZED;

    int64_t scratch = 0;
    compute_svd_sizes(*tin, *tu, *tv, scratch);
    w->device_scratch.required = scratch;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_workspace_compute_qr_sizes(
    roctensornet_handle              handle,
    roctensornet_tensor_descriptor   desc_in,
    roctensornet_tensor_descriptor   desc_q,
    roctensornet_tensor_descriptor   desc_r,
    roctensornet_workspace_descriptor workspace_desc)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    auto* tin = cast<tensor_descriptor_st>(desc_in);
    auto* tq  = cast<tensor_descriptor_st>(desc_q);
    auto* tr  = cast<tensor_descriptor_st>(desc_r);
    auto* w   = cast<workspace_descriptor_st>(workspace_desc);
    if(tin == nullptr || tq == nullptr || tr == nullptr || w == nullptr)
        return ROCTENSORNET_STATUS_NOT_INITIALIZED;

    int64_t scratch = 0;
    compute_qr_sizes(*tin, *tq, *tr, scratch);
    w->device_scratch.required = scratch;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_workspace_get_memory_size(
    roctensornet_handle               handle,
    roctensornet_workspace_descriptor desc,
    roctensornet_worksize_pref        pref,
    roctensornet_memspace             mem_space,
    roctensornet_workspace_kind       kind,
    int64_t*                          size_out)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    ROCTENSORNET_CHECK_PTR(size_out);
    auto* w = cast<workspace_descriptor_st>(desc);
    if(w == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    auto* slot = w->select(mem_space, kind);
    if(slot == nullptr) return ROCTENSORNET_STATUS_INVALID_VALUE;
    /* MIN, RECOMMENDED, MAX collapse to a single value in v0.1.0. */
    (void)pref;
    *size_out = slot->required;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_workspace_set_memory(roctensornet_handle               handle,
                                  roctensornet_workspace_descriptor desc,
                                  roctensornet_memspace             mem_space,
                                  roctensornet_workspace_kind       kind,
                                  void*                             memory_buffer,
                                  int64_t                           memory_size)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    auto* w = cast<workspace_descriptor_st>(desc);
    if(w == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    auto* slot = w->select(mem_space, kind);
    if(slot == nullptr) return ROCTENSORNET_STATUS_INVALID_VALUE;
    if(memory_buffer == nullptr && memory_size != 0)
        return ROCTENSORNET_STATUS_INVALID_VALUE;
    slot->buffer       = memory_buffer;
    slot->buffer_bytes = memory_size;
    slot->user_owned   = true;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_workspace_get_memory(roctensornet_handle               handle,
                                  roctensornet_workspace_descriptor desc,
                                  roctensornet_memspace             mem_space,
                                  roctensornet_workspace_kind       kind,
                                  void**                            buffer_out,
                                  int64_t*                          size_out)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    auto* w = cast<workspace_descriptor_st>(desc);
    if(w == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    auto* slot = w->select(mem_space, kind);
    if(slot == nullptr) return ROCTENSORNET_STATUS_INVALID_VALUE;
    if(buffer_out) *buffer_out = slot->buffer;
    if(size_out)   *size_out   = slot->buffer_bytes;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_workspace_purge_cache(roctensornet_handle               handle,
                                   roctensornet_workspace_descriptor desc,
                                   roctensornet_memspace             mem_space)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    auto* w = cast<workspace_descriptor_st>(desc);
    if(w == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    auto* slot = w->select(mem_space, ROCTENSORNET_WORKSPACE_CACHE);
    if(slot == nullptr) return ROCTENSORNET_STATUS_INVALID_VALUE;
    /* The cache is otherwise long-lived and managed by the executor;
     * marking it dirty causes the next call to repopulate it. v0.1.0
     * does not retain device-resident state across `contraction()`
     * calls, so the only action here is to drop user-supplied state. */
    if(!slot->user_owned)
    {
        slot->buffer       = nullptr;
        slot->buffer_bytes = 0;
    }
    return ROCTENSORNET_STATUS_SUCCESS;
}

} // extern "C"
