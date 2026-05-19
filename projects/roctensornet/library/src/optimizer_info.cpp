/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Contraction optimizer info: lifecycle, attribute round-trip, and
 * pack/unpack (binary serialization owned by rocTENSORNET).
 *
 * The serialization format is a versioned little-endian blob:
 *   [u32 magic = 'RTNI']
 *   [u32 version = 1]
 *   [i32 num_inputs]
 *   [i32 num_output_modes]
 *   [i32 path_len]
 *   path_len * { i32 left, i32 right, i32 out_modes_count,
 *                out_modes_count * { i32 mode, i64 extent },
 *                u64 flops, u64 intermediate_bytes }
 *   [i32 num_sliced]
 *   num_sliced * { i32 mode, i64 extent }
 *   [i64 num_slices, u64 flop_count, u64 largest_tensor]
 *   [i32 data_type, i32 compute_type]
 *   [i32 num_output_modes]
 *   num_output_modes * { i32 mode, i64 extent }
 *
 * Pack/unpack is internal-only; the user treats the blob as opaque.
 * ************************************************************************ */

#include "roctensornet_descriptors.hpp"

#include <cstring>

using namespace roctensornet;

namespace
{
constexpr uint32_t magic_value   = 0x494E5452U; /* 'RTNI' little-endian */
constexpr uint32_t blob_version  = 1U;

template <class T> void put(uint8_t*& p, const T& v) { std::memcpy(p, &v, sizeof(T)); p += sizeof(T); }
template <class T> bool get(const uint8_t*& p, const uint8_t* end, T& v)
{
    if(static_cast<size_t>(end - p) < sizeof(T)) return false;
    std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return true;
}
} // anon

extern "C" {

roctensornet_status
roctensornet_create_contraction_optimizer_info(
    roctensornet_handle handle,
    roctensornet_network_descriptor desc,
    roctensornet_contraction_optimizer_info* out)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    ROCTENSORNET_CHECK_PTR(out);
    auto* nd = cast<network_descriptor_st>(desc);
    if(nd == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    auto* info = new (std::nothrow) contraction_optimizer_info_st();
    if(info == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    info->num_inputs       = nd->num_inputs;
    info->num_output_modes = nd->output.num_modes();
    info->output_modes     = nd->output.modes;
    info->output_extents   = nd->output.extents;
    info->data_type        = nd->data_type;
    info->compute_type     = nd->compute_type;
    info->num_slices       = 1;
    *out = reinterpret_cast<roctensornet_contraction_optimizer_info>(info);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_destroy_contraction_optimizer_info(
    roctensornet_contraction_optimizer_info i)
{
    delete cast<contraction_optimizer_info_st>(i);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_contraction_optimizer_info_get_attribute(
    roctensornet_handle                               handle,
    roctensornet_contraction_optimizer_info           info,
    roctensornet_contraction_optimizer_info_attribute attr,
    void*                                             value,
    size_t                                            size)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    auto* ii = cast<contraction_optimizer_info_st>(info);
    if(ii == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    switch(attr)
    {
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES:
        if(size < sizeof(int64_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int64_t*>(value) = ii->num_slices;
        return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int32_t*>(value) = (int32_t)ii->sliced_modes.size();
        return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT:
        if(size < sizeof(size_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<size_t*>(value) = ii->flop_count;
        return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR:
        if(size < sizeof(size_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<size_t*>(value) = ii->largest_tensor;
        return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_EFFECTIVE_FLOPS:
        if(size < sizeof(double)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<double*>(value) = static_cast<double>(ii->flop_count);
        return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE:
    {
        size_t need = sizeof(int32_t) * ii->sliced_modes.size();
        if(size < need) return ROCTENSORNET_STATUS_INVALID_VALUE;
        std::memcpy(value, ii->sliced_modes.data(), need);
        return ROCTENSORNET_STATUS_SUCCESS;
    }
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT:
    {
        size_t need = sizeof(roctensornet_index_t) * ii->sliced_extents.size();
        if(size < need) return ROCTENSORNET_STATUS_INVALID_VALUE;
        std::memcpy(value, ii->sliced_extents.data(), need);
        return ROCTENSORNET_STATUS_SUCCESS;
    }
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH:
    {
        /* Returns pairs of i32 (left, right). */
        size_t need = sizeof(int32_t) * 2 * ii->path.size();
        if(size < need) return ROCTENSORNET_STATUS_INVALID_VALUE;
        auto* p = static_cast<int32_t*>(value);
        for(const auto& n : ii->path) { *p++ = n.left; *p++ = n.right; }
        return ROCTENSORNET_STATUS_SUCCESS;
    }
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_INTERMEDIATE_MODES:
    {
        /* For each pair node we return (num_modes, modes...). */
        size_t need = sizeof(int32_t);
        for(const auto& n : ii->path) need += sizeof(int32_t) * (1 + n.out_modes.size());
        if(size < need) return ROCTENSORNET_STATUS_INVALID_VALUE;
        auto* p = static_cast<int32_t*>(value);
        *p++ = static_cast<int32_t>(ii->path.size());
        for(const auto& n : ii->path)
        {
            *p++ = static_cast<int32_t>(n.out_modes.size());
            for(auto m : n.out_modes) *p++ = m;
        }
        return ROCTENSORNET_STATUS_SUCCESS;
    }
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_contraction_optimizer_info_set_attribute(
    roctensornet_handle                               handle,
    roctensornet_contraction_optimizer_info           info,
    roctensornet_contraction_optimizer_info_attribute attr,
    const void*                                       value,
    size_t                                            size)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    auto* ii = cast<contraction_optimizer_info_st>(info);
    if(ii == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    (void)attr; (void)size;
    /* The info object is normally produced by `contraction_optimize`;
     * user-side writes are rare and only documented for advanced
     * scenarios. v0.1.0 supports none. */
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_contraction_optimizer_info_get_packed_size(
    roctensornet_handle                     handle,
    roctensornet_contraction_optimizer_info info,
    size_t*                                 out_size)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    ROCTENSORNET_CHECK_PTR(out_size);
    auto* ii = cast<contraction_optimizer_info_st>(info);
    if(ii == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    size_t s = 0;
    s += 8;                                        /* magic + version       */
    s += sizeof(int32_t) * 3;                      /* num_inputs, num_out_modes, path_len */
    for(const auto& n : ii->path)
    {
        s += sizeof(int32_t) * 3;                  /* left, right, out_modes_count */
        s += (sizeof(int32_t) + sizeof(int64_t)) * n.out_modes.size();
        s += sizeof(uint64_t) * 2;                 /* flops, intermediate_bytes */
    }
    s += sizeof(int32_t);                          /* num_sliced            */
    s += (sizeof(int32_t) + sizeof(int64_t)) * ii->sliced_modes.size();
    s += sizeof(int64_t) + sizeof(uint64_t) * 2;   /* num_slices, flop_count, largest */
    s += sizeof(int32_t) * 2;                      /* data_type, compute_type */
    s += sizeof(int32_t);                          /* duplicate num_output_modes */
    s += (sizeof(int32_t) + sizeof(int64_t)) * ii->output_modes.size();
    *out_size = s;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_contraction_optimizer_info_pack_data(
    roctensornet_handle                     handle,
    roctensornet_contraction_optimizer_info info,
    void*                                   buffer,
    size_t                                  size)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    ROCTENSORNET_CHECK_PTR(buffer);
    auto* ii = cast<contraction_optimizer_info_st>(info);
    if(ii == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    size_t need = 0;
    auto rc = roctensornet_contraction_optimizer_info_get_packed_size(handle, info, &need);
    if(rc != ROCTENSORNET_STATUS_SUCCESS) return rc;
    if(size < need) return ROCTENSORNET_STATUS_INSUFFICIENT_WORKSPACE;

    auto*  p = static_cast<uint8_t*>(buffer);
    put(p, magic_value);
    put(p, blob_version);
    put(p, ii->num_inputs);
    put(p, ii->num_output_modes);
    int32_t path_len = static_cast<int32_t>(ii->path.size());
    put(p, path_len);
    for(const auto& n : ii->path)
    {
        put(p, n.left); put(p, n.right);
        int32_t m = static_cast<int32_t>(n.out_modes.size());
        put(p, m);
        for(size_t k = 0; k < n.out_modes.size(); ++k)
        {
            put(p, n.out_modes[k]);
            put(p, n.out_extents[k]);
        }
        put(p, static_cast<uint64_t>(n.flops));
        put(p, static_cast<uint64_t>(n.intermediate_bytes));
    }
    int32_t ns = static_cast<int32_t>(ii->sliced_modes.size());
    put(p, ns);
    for(int32_t k = 0; k < ns; ++k)
    {
        put(p, ii->sliced_modes[k]);
        put(p, ii->sliced_extents[k]);
    }
    put(p, ii->num_slices);
    put(p, static_cast<uint64_t>(ii->flop_count));
    put(p, static_cast<uint64_t>(ii->largest_tensor));
    int32_t dt = static_cast<int32_t>(ii->data_type);
    int32_t ct = static_cast<int32_t>(ii->compute_type);
    put(p, dt); put(p, ct);
    int32_t nout = static_cast<int32_t>(ii->output_modes.size());
    put(p, nout);
    for(int32_t k = 0; k < nout; ++k)
    {
        put(p, ii->output_modes[k]);
        put(p, ii->output_extents[k]);
    }
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_create_contraction_optimizer_info_from_packed_data(
    roctensornet_handle             handle,
    roctensornet_network_descriptor desc,
    const void*                     buffer,
    size_t                          size,
    roctensornet_contraction_optimizer_info* out)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    ROCTENSORNET_CHECK_PTR(buffer);
    ROCTENSORNET_CHECK_PTR(out);
    (void)desc;

    const uint8_t* p   = static_cast<const uint8_t*>(buffer);
    const uint8_t* end = p + size;
    uint32_t magic = 0, ver = 0;
    if(!get(p, end, magic) || magic != magic_value) return ROCTENSORNET_STATUS_INVALID_VALUE;
    if(!get(p, end, ver)   || ver   != blob_version) return ROCTENSORNET_STATUS_NOT_SUPPORTED;

    auto* ii = new (std::nothrow) contraction_optimizer_info_st();
    if(ii == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;

    int32_t path_len = 0;
    if(!get(p, end, ii->num_inputs)
       || !get(p, end, ii->num_output_modes)
       || !get(p, end, path_len)) { delete ii; return ROCTENSORNET_STATUS_INVALID_VALUE; }
    ii->path.resize(path_len);
    for(int32_t i = 0; i < path_len; ++i)
    {
        auto& n = ii->path[i];
        int32_t mm = 0;
        if(!get(p, end, n.left) || !get(p, end, n.right) || !get(p, end, mm))
        { delete ii; return ROCTENSORNET_STATUS_INVALID_VALUE; }
        n.out_modes.resize(mm);
        n.out_extents.resize(mm);
        for(int32_t k = 0; k < mm; ++k)
        {
            if(!get(p, end, n.out_modes[k]) || !get(p, end, n.out_extents[k]))
            { delete ii; return ROCTENSORNET_STATUS_INVALID_VALUE; }
        }
        uint64_t f = 0, ib = 0;
        if(!get(p, end, f) || !get(p, end, ib))
        { delete ii; return ROCTENSORNET_STATUS_INVALID_VALUE; }
        n.flops              = static_cast<size_t>(f);
        n.intermediate_bytes = static_cast<size_t>(ib);
    }
    int32_t ns = 0;
    if(!get(p, end, ns)) { delete ii; return ROCTENSORNET_STATUS_INVALID_VALUE; }
    ii->sliced_modes.resize(ns);
    ii->sliced_extents.resize(ns);
    for(int32_t k = 0; k < ns; ++k)
        if(!get(p, end, ii->sliced_modes[k]) || !get(p, end, ii->sliced_extents[k]))
        { delete ii; return ROCTENSORNET_STATUS_INVALID_VALUE; }
    uint64_t fc = 0, lt = 0;
    if(!get(p, end, ii->num_slices) || !get(p, end, fc) || !get(p, end, lt))
    { delete ii; return ROCTENSORNET_STATUS_INVALID_VALUE; }
    ii->flop_count     = static_cast<size_t>(fc);
    ii->largest_tensor = static_cast<size_t>(lt);
    int32_t dt = 0, ct = 0;
    if(!get(p, end, dt) || !get(p, end, ct))
    { delete ii; return ROCTENSORNET_STATUS_INVALID_VALUE; }
    ii->data_type    = static_cast<roctensornet_data_type>(dt);
    ii->compute_type = static_cast<roctensornet_compute_type>(ct);
    int32_t nout = 0;
    if(!get(p, end, nout)) { delete ii; return ROCTENSORNET_STATUS_INVALID_VALUE; }
    ii->output_modes.resize(nout);
    ii->output_extents.resize(nout);
    for(int32_t k = 0; k < nout; ++k)
        if(!get(p, end, ii->output_modes[k]) || !get(p, end, ii->output_extents[k]))
        { delete ii; return ROCTENSORNET_STATUS_INVALID_VALUE; }

    *out = reinterpret_cast<roctensornet_contraction_optimizer_info>(ii);
    return ROCTENSORNET_STATUS_SUCCESS;
}

} // extern "C"
