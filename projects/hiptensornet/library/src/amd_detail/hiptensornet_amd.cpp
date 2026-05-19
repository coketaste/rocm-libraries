/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * AMD backend for hipTENSORNET: a 1:1 pass-through to rocTENSORNET.
 *
 * The hipTENSORNET handle types are layout-compatible (opaque void*)
 * with the rocTENSORNET handle types; we cast between them with
 * reinterpret_cast at the API boundary. Enum tokens are
 * value-compatible by construction (the values are identical and
 * documented in both headers).
 * ************************************************************************ */

#include "hiptensornet.h"
#include "roctensornet.h"

#include <cstring>

namespace
{
/* ---- Trivial cast helpers (no ABI translation needed for v0.1.0). */
#define X(rt, ht) \
    inline rt to_roc(ht v) { return reinterpret_cast<rt>(v); } \
    inline ht to_hip(rt v) { return reinterpret_cast<ht>(v); } \
    inline rt* to_roc_pp(ht* v) { return reinterpret_cast<rt*>(v); }
X(roctensornet_handle,                              hiptensornet_handle)
X(roctensornet_network_descriptor,                  hiptensornet_network_descriptor)
X(roctensornet_tensor_descriptor,                   hiptensornet_tensor_descriptor)
X(roctensornet_contraction_optimizer_config,        hiptensornet_contraction_optimizer_config)
X(roctensornet_contraction_optimizer_info,          hiptensornet_contraction_optimizer_info)
X(roctensornet_workspace_descriptor,                hiptensornet_workspace_descriptor)
X(roctensornet_contraction_plan,                    hiptensornet_contraction_plan)
X(roctensornet_contraction_autotune_preference,     hiptensornet_contraction_autotune_preference)
X(roctensornet_slice_group,                         hiptensornet_slice_group)
X(roctensornet_tensor_svd_config,                   hiptensornet_tensor_svd_config)
X(roctensornet_tensor_svd_info,                     hiptensornet_tensor_svd_info)
X(roctensornet_network_operator,                    hiptensornet_network_operator)
X(roctensornet_state,                               hiptensornet_state)
X(roctensornet_state_marginal,                      hiptensornet_state_marginal)
X(roctensornet_state_sampler,                       hiptensornet_state_sampler)
X(roctensornet_state_expectation,                   hiptensornet_state_expectation)
X(roctensornet_state_accessor,                      hiptensornet_state_accessor)
#undef X

inline hiptensornet_status to_hip_status(roctensornet_status s)
{ return static_cast<hiptensornet_status>(static_cast<int>(s)); }
inline roctensornet_data_type to_roc_dtype(hiptensornet_data_type d)
{ return static_cast<roctensornet_data_type>(static_cast<int>(d)); }
inline roctensornet_compute_type to_roc_ctype(hiptensornet_compute_type d)
{ return static_cast<roctensornet_compute_type>(static_cast<int>(d)); }
} // anon

extern "C" {

#define FORWARD(roc_fn, hip_fn, ...) return to_hip_status(roc_fn(__VA_ARGS__))

/* ---- Handle / library ---- */
hiptensornet_status hiptensornet_create(hiptensornet_handle* h)
{
    roctensornet_handle r = nullptr;
    auto s = roctensornet_create(&r);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *h = to_hip(r);
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_destroy(hiptensornet_handle h)
{ return to_hip_status(roctensornet_destroy(to_roc(h))); }
hiptensornet_status hiptensornet_get_version(int* v) { return to_hip_status(roctensornet_get_version(v)); }
hiptensornet_status hiptensornet_get_hip_runtime_version(int* v) { return to_hip_status(roctensornet_get_hip_runtime_version(v)); }
const char* hiptensornet_get_error_string(hiptensornet_status s)
{ return roctensornet_get_error_string(static_cast<roctensornet_status>(s)); }
hiptensornet_status hiptensornet_set_stream(hiptensornet_handle h, hipStream_t s)
{ return to_hip_status(roctensornet_set_stream(to_roc(h), s)); }
hiptensornet_status hiptensornet_get_stream(hiptensornet_handle h, hipStream_t* s)
{ return to_hip_status(roctensornet_get_stream(to_roc(h), s)); }
hiptensornet_status hiptensornet_set_device_mem_handler(hiptensornet_handle h,
    const hiptensornet_device_mem_handler_t* m)
{
    if(m == nullptr) return to_hip_status(roctensornet_set_device_mem_handler(to_roc(h), nullptr));
    roctensornet_device_mem_handler_t rm{};
    rm.ctx          = m->ctx;
    rm.device_alloc = m->device_alloc;
    rm.device_free  = m->device_free;
    std::memcpy(rm.name, m->name, sizeof(rm.name));
    return to_hip_status(roctensornet_set_device_mem_handler(to_roc(h), &rm));
}
hiptensornet_status hiptensornet_get_device_mem_handler(hiptensornet_handle h,
    hiptensornet_device_mem_handler_t* out)
{
    roctensornet_device_mem_handler_t rm{};
    auto s = roctensornet_get_device_mem_handler(to_roc(h), &rm);
    if(s == ROCTENSORNET_STATUS_SUCCESS && out)
    {
        out->ctx          = rm.ctx;
        out->device_alloc = rm.device_alloc;
        out->device_free  = rm.device_free;
        std::memcpy(out->name, rm.name, sizeof(out->name));
    }
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_logger_set_callback(hiptensornet_logger_callback_t cb)
{ return to_hip_status(roctensornet_logger_set_callback(reinterpret_cast<roctensornet_logger_callback_t>(cb))); }
hiptensornet_status hiptensornet_logger_set_callback_data(hiptensornet_logger_callback_data_t cb, void* u)
{ return to_hip_status(roctensornet_logger_set_callback_data(reinterpret_cast<roctensornet_logger_callback_data_t>(cb), u)); }
hiptensornet_status hiptensornet_logger_set_file(void* file)
{ return to_hip_status(roctensornet_logger_set_file(file)); }
hiptensornet_status hiptensornet_logger_open_file(const char* file_name)
{ return to_hip_status(roctensornet_logger_open_file(file_name)); }
hiptensornet_status hiptensornet_logger_set_level(int32_t level)
{ return to_hip_status(roctensornet_logger_set_level(level)); }
hiptensornet_status hiptensornet_logger_set_mask(int32_t mask)
{ return to_hip_status(roctensornet_logger_set_mask(mask)); }
hiptensornet_status hiptensornet_logger_force_disable(void)
{ return to_hip_status(roctensornet_logger_force_disable()); }

/* ---- Network / tensor descriptor ---- */
hiptensornet_status
hiptensornet_create_network_descriptor(hiptensornet_handle h, int32_t ni,
                                       const int32_t* nm,
                                       const hiptensornet_index_t* const* ex,
                                       const hiptensornet_index_t* const* st,
                                       const int32_t* const* mi,
                                       const uint32_t* al,
                                       int32_t no,
                                       const hiptensornet_index_t* eo,
                                       const hiptensornet_index_t* so,
                                       const int32_t* mo,
                                       uint32_t ao,
                                       hiptensornet_data_type dt,
                                       hiptensornet_compute_type ct,
                                       hiptensornet_network_descriptor* out)
{
    roctensornet_network_descriptor rd = nullptr;
    auto s = roctensornet_create_network_descriptor(
        to_roc(h), ni, nm,
        reinterpret_cast<const roctensornet_index_t* const*>(ex),
        reinterpret_cast<const roctensornet_index_t* const*>(st),
        mi, al, no,
        reinterpret_cast<const roctensornet_index_t*>(eo),
        reinterpret_cast<const roctensornet_index_t*>(so),
        mo, ao,
        to_roc_dtype(dt), to_roc_ctype(ct), &rd);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *out = to_hip(rd);
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_destroy_network_descriptor(hiptensornet_network_descriptor d)
{ return to_hip_status(roctensornet_destroy_network_descriptor(to_roc(d))); }
hiptensornet_status hiptensornet_get_output_tensor_descriptor(hiptensornet_handle h,
    hiptensornet_network_descriptor d, hiptensornet_tensor_descriptor* out)
{
    roctensornet_tensor_descriptor rt = nullptr;
    auto s = roctensornet_get_output_tensor_descriptor(to_roc(h), to_roc(d), &rt);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *out = to_hip(rt);
    return to_hip_status(s);
}
hiptensornet_status
hiptensornet_create_tensor_descriptor(hiptensornet_handle h, int32_t nm,
                                      const hiptensornet_index_t* ext,
                                      const hiptensornet_index_t* str,
                                      const int32_t* modes,
                                      hiptensornet_data_type dt,
                                      hiptensornet_tensor_descriptor* out)
{
    roctensornet_tensor_descriptor rt = nullptr;
    auto s = roctensornet_create_tensor_descriptor(
        to_roc(h), nm, reinterpret_cast<const roctensornet_index_t*>(ext),
        reinterpret_cast<const roctensornet_index_t*>(str),
        modes, to_roc_dtype(dt), &rt);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *out = to_hip(rt);
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_destroy_tensor_descriptor(hiptensornet_tensor_descriptor d)
{ return to_hip_status(roctensornet_destroy_tensor_descriptor(to_roc(d))); }
hiptensornet_status
hiptensornet_get_tensor_details(hiptensornet_handle h, hiptensornet_tensor_descriptor d,
                                int32_t* nm, size_t* db, hiptensornet_data_type* dt,
                                int32_t* modes, hiptensornet_index_t* ext,
                                hiptensornet_index_t* str)
{
    roctensornet_data_type rdt;
    auto s = roctensornet_get_tensor_details(
        to_roc(h), to_roc(d), nm, db,
        dt ? &rdt : nullptr, modes,
        reinterpret_cast<roctensornet_index_t*>(ext),
        reinterpret_cast<roctensornet_index_t*>(str));
    if(dt && s == ROCTENSORNET_STATUS_SUCCESS) *dt = static_cast<hiptensornet_data_type>(static_cast<int>(rdt));
    return to_hip_status(s);
}
hiptensornet_status
hiptensornet_network_get_attribute(hiptensornet_handle h, hiptensornet_network_descriptor d,
                                   hiptensornet_network_attribute a, void* v, size_t sz)
{
    return to_hip_status(roctensornet_network_get_attribute(
        to_roc(h), to_roc(d),
        static_cast<roctensornet_network_attribute>(static_cast<int>(a)), v, sz));
}
hiptensornet_status
hiptensornet_network_set_attribute(hiptensornet_handle h, hiptensornet_network_descriptor d,
                                   hiptensornet_network_attribute a, const void* v, size_t sz)
{
    return to_hip_status(roctensornet_network_set_attribute(
        to_roc(h), to_roc(d),
        static_cast<roctensornet_network_attribute>(static_cast<int>(a)), v, sz));
}

/* ---- Optimizer ---- */
hiptensornet_status hiptensornet_create_contraction_optimizer_config(hiptensornet_handle h, hiptensornet_contraction_optimizer_config* o)
{
    roctensornet_contraction_optimizer_config r = nullptr;
    auto s = roctensornet_create_contraction_optimizer_config(to_roc(h), &r);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *o = to_hip(r);
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_destroy_contraction_optimizer_config(hiptensornet_contraction_optimizer_config c)
{ return to_hip_status(roctensornet_destroy_contraction_optimizer_config(to_roc(c))); }
hiptensornet_status hiptensornet_contraction_optimizer_config_get_attribute(
    hiptensornet_handle h, hiptensornet_contraction_optimizer_config c,
    hiptensornet_contraction_optimizer_config_attribute a, void* v, size_t s)
{
    return to_hip_status(roctensornet_contraction_optimizer_config_get_attribute(
        to_roc(h), to_roc(c),
        static_cast<roctensornet_contraction_optimizer_config_attribute>(static_cast<int>(a)),
        v, s));
}
hiptensornet_status hiptensornet_contraction_optimizer_config_set_attribute(
    hiptensornet_handle h, hiptensornet_contraction_optimizer_config c,
    hiptensornet_contraction_optimizer_config_attribute a, const void* v, size_t s)
{
    return to_hip_status(roctensornet_contraction_optimizer_config_set_attribute(
        to_roc(h), to_roc(c),
        static_cast<roctensornet_contraction_optimizer_config_attribute>(static_cast<int>(a)),
        v, s));
}
hiptensornet_status hiptensornet_create_contraction_optimizer_info(hiptensornet_handle h, hiptensornet_network_descriptor d, hiptensornet_contraction_optimizer_info* o)
{
    roctensornet_contraction_optimizer_info r = nullptr;
    auto s = roctensornet_create_contraction_optimizer_info(to_roc(h), to_roc(d), &r);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *o = to_hip(r);
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_destroy_contraction_optimizer_info(hiptensornet_contraction_optimizer_info i)
{ return to_hip_status(roctensornet_destroy_contraction_optimizer_info(to_roc(i))); }
hiptensornet_status hiptensornet_contraction_optimizer_info_get_attribute(
    hiptensornet_handle h, hiptensornet_contraction_optimizer_info i,
    hiptensornet_contraction_optimizer_info_attribute a, void* v, size_t s)
{
    return to_hip_status(roctensornet_contraction_optimizer_info_get_attribute(
        to_roc(h), to_roc(i),
        static_cast<roctensornet_contraction_optimizer_info_attribute>(static_cast<int>(a)),
        v, s));
}
hiptensornet_status hiptensornet_contraction_optimizer_info_set_attribute(
    hiptensornet_handle h, hiptensornet_contraction_optimizer_info i,
    hiptensornet_contraction_optimizer_info_attribute a, const void* v, size_t s)
{
    return to_hip_status(roctensornet_contraction_optimizer_info_set_attribute(
        to_roc(h), to_roc(i),
        static_cast<roctensornet_contraction_optimizer_info_attribute>(static_cast<int>(a)),
        v, s));
}
hiptensornet_status hiptensornet_contraction_optimizer_info_get_packed_size(
    hiptensornet_handle h, hiptensornet_contraction_optimizer_info i, size_t* out_size)
{ return to_hip_status(roctensornet_contraction_optimizer_info_get_packed_size(to_roc(h), to_roc(i), out_size)); }
hiptensornet_status hiptensornet_contraction_optimizer_info_pack_data(
    hiptensornet_handle h, hiptensornet_contraction_optimizer_info i, void* buf, size_t size)
{ return to_hip_status(roctensornet_contraction_optimizer_info_pack_data(to_roc(h), to_roc(i), buf, size)); }
hiptensornet_status hiptensornet_create_contraction_optimizer_info_from_packed_data(
    hiptensornet_handle h, hiptensornet_network_descriptor d,
    const void* buf, size_t size, hiptensornet_contraction_optimizer_info* out)
{
    roctensornet_contraction_optimizer_info r = nullptr;
    auto s = roctensornet_create_contraction_optimizer_info_from_packed_data(
        to_roc(h), to_roc(d), buf, size, &r);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *out = to_hip(r);
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_contraction_optimize(hiptensornet_handle h, hiptensornet_network_descriptor d, hiptensornet_contraction_optimizer_config c, uint64_t ws, hiptensornet_contraction_optimizer_info i)
{ return to_hip_status(roctensornet_contraction_optimize(to_roc(h), to_roc(d), to_roc(c), ws, to_roc(i))); }

/* ---- Workspace ---- */
hiptensornet_status hiptensornet_create_workspace_descriptor(hiptensornet_handle h, hiptensornet_workspace_descriptor* o)
{
    roctensornet_workspace_descriptor r = nullptr;
    auto s = roctensornet_create_workspace_descriptor(to_roc(h), &r);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *o = to_hip(r);
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_destroy_workspace_descriptor(hiptensornet_workspace_descriptor d)
{ return to_hip_status(roctensornet_destroy_workspace_descriptor(to_roc(d))); }
hiptensornet_status hiptensornet_workspace_compute_contraction_sizes(hiptensornet_handle h, hiptensornet_network_descriptor d, hiptensornet_contraction_optimizer_info i, hiptensornet_workspace_descriptor w)
{ return to_hip_status(roctensornet_workspace_compute_contraction_sizes(to_roc(h), to_roc(d), to_roc(i), to_roc(w))); }
hiptensornet_status hiptensornet_workspace_compute_svd_sizes(
    hiptensornet_handle h, hiptensornet_tensor_descriptor din,
    hiptensornet_tensor_descriptor du, hiptensornet_tensor_descriptor dv,
    hiptensornet_tensor_svd_config c, hiptensornet_workspace_descriptor w)
{ return to_hip_status(roctensornet_workspace_compute_svd_sizes(to_roc(h), to_roc(din), to_roc(du), to_roc(dv), to_roc(c), to_roc(w))); }
hiptensornet_status hiptensornet_workspace_compute_qr_sizes(
    hiptensornet_handle h, hiptensornet_tensor_descriptor din,
    hiptensornet_tensor_descriptor dq, hiptensornet_tensor_descriptor dr,
    hiptensornet_workspace_descriptor w)
{ return to_hip_status(roctensornet_workspace_compute_qr_sizes(to_roc(h), to_roc(din), to_roc(dq), to_roc(dr), to_roc(w))); }
hiptensornet_status hiptensornet_workspace_get_memory_size(hiptensornet_handle h, hiptensornet_workspace_descriptor d, hiptensornet_worksize_pref p, hiptensornet_memspace m, hiptensornet_workspace_kind k, int64_t* sz)
{
    return to_hip_status(roctensornet_workspace_get_memory_size(
        to_roc(h), to_roc(d),
        static_cast<roctensornet_worksize_pref>(static_cast<int>(p)),
        static_cast<roctensornet_memspace>(static_cast<int>(m)),
        static_cast<roctensornet_workspace_kind>(static_cast<int>(k)), sz));
}
hiptensornet_status hiptensornet_workspace_set_memory(hiptensornet_handle h, hiptensornet_workspace_descriptor d, hiptensornet_memspace m, hiptensornet_workspace_kind k, void* buf, int64_t sz)
{
    return to_hip_status(roctensornet_workspace_set_memory(
        to_roc(h), to_roc(d),
        static_cast<roctensornet_memspace>(static_cast<int>(m)),
        static_cast<roctensornet_workspace_kind>(static_cast<int>(k)), buf, sz));
}
hiptensornet_status hiptensornet_workspace_get_memory(hiptensornet_handle h, hiptensornet_workspace_descriptor d, hiptensornet_memspace m, hiptensornet_workspace_kind k, void** buf, int64_t* sz)
{
    return to_hip_status(roctensornet_workspace_get_memory(
        to_roc(h), to_roc(d),
        static_cast<roctensornet_memspace>(static_cast<int>(m)),
        static_cast<roctensornet_workspace_kind>(static_cast<int>(k)), buf, sz));
}
hiptensornet_status hiptensornet_workspace_purge_cache(hiptensornet_handle h, hiptensornet_workspace_descriptor d, hiptensornet_memspace m)
{
    return to_hip_status(roctensornet_workspace_purge_cache(
        to_roc(h), to_roc(d),
        static_cast<roctensornet_memspace>(static_cast<int>(m))));
}

/* ---- Contraction ---- */
hiptensornet_status hiptensornet_create_contraction_plan(hiptensornet_handle h, hiptensornet_network_descriptor d, hiptensornet_contraction_optimizer_info i, hiptensornet_workspace_descriptor w, hiptensornet_contraction_plan* p)
{
    roctensornet_contraction_plan r = nullptr;
    auto s = roctensornet_create_contraction_plan(to_roc(h), to_roc(d), to_roc(i), to_roc(w), &r);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *p = to_hip(r);
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_destroy_contraction_plan(hiptensornet_contraction_plan p)
{ return to_hip_status(roctensornet_destroy_contraction_plan(to_roc(p))); }
hiptensornet_status hiptensornet_contraction(hiptensornet_handle h, hiptensornet_contraction_plan p, const void* const* in, void* out, hiptensornet_workspace_descriptor w, int64_t slice, hipStream_t s)
{ return to_hip_status(roctensornet_contraction(to_roc(h), to_roc(p), in, out, to_roc(w), slice, s)); }
hiptensornet_status hiptensornet_contract_slices(hiptensornet_handle h, hiptensornet_contraction_plan p, const void* const* in, void* out, int32_t acc, hiptensornet_workspace_descriptor w, hiptensornet_slice_group g, hipStream_t s)
{ return to_hip_status(roctensornet_contract_slices(to_roc(h), to_roc(p), in, out, acc, to_roc(w), to_roc(g), s)); }
hiptensornet_status hiptensornet_create_contraction_autotune_preference(hiptensornet_handle h, hiptensornet_contraction_autotune_preference* o)
{
    roctensornet_contraction_autotune_preference r = nullptr;
    auto s = roctensornet_create_contraction_autotune_preference(to_roc(h), &r);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *o = to_hip(r);
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_destroy_contraction_autotune_preference(hiptensornet_contraction_autotune_preference o)
{ return to_hip_status(roctensornet_destroy_contraction_autotune_preference(to_roc(o))); }
hiptensornet_status hiptensornet_contraction_autotune_preference_get_attribute(
    hiptensornet_handle h, hiptensornet_contraction_autotune_preference p,
    hiptensornet_contraction_autotune_preference_attribute a, void* v, size_t s)
{
    return to_hip_status(roctensornet_contraction_autotune_preference_get_attribute(
        to_roc(h), to_roc(p),
        static_cast<roctensornet_contraction_autotune_preference_attribute>(static_cast<int>(a)),
        v, s));
}
hiptensornet_status hiptensornet_contraction_autotune_preference_set_attribute(
    hiptensornet_handle h, hiptensornet_contraction_autotune_preference p,
    hiptensornet_contraction_autotune_preference_attribute a, const void* v, size_t s)
{
    return to_hip_status(roctensornet_contraction_autotune_preference_set_attribute(
        to_roc(h), to_roc(p),
        static_cast<roctensornet_contraction_autotune_preference_attribute>(static_cast<int>(a)),
        v, s));
}
hiptensornet_status hiptensornet_contraction_autotune(hiptensornet_handle h, hiptensornet_contraction_plan p, const void* const* in, void* out, hiptensornet_workspace_descriptor w, hiptensornet_contraction_autotune_preference pref, hipStream_t s)
{ return to_hip_status(roctensornet_contraction_autotune(to_roc(h), to_roc(p), in, out, to_roc(w), to_roc(pref), s)); }

/* ---- Slice group ---- */
hiptensornet_status hiptensornet_create_slice_group_from_id_range(hiptensornet_handle h, int64_t a, int64_t b, int64_t c, hiptensornet_slice_group* o)
{
    roctensornet_slice_group r = nullptr;
    auto s = roctensornet_create_slice_group_from_id_range(to_roc(h), a, b, c, &r);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *o = to_hip(r);
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_create_slice_group_from_ids(hiptensornet_handle h, const int64_t* ids, int32_t n, hiptensornet_slice_group* o)
{
    roctensornet_slice_group r = nullptr;
    auto s = roctensornet_create_slice_group_from_ids(to_roc(h), ids, n, &r);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *o = to_hip(r);
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_destroy_slice_group(hiptensornet_slice_group g)
{ return to_hip_status(roctensornet_destroy_slice_group(to_roc(g))); }

/* ---- Tensor SVD / QR ---- */
hiptensornet_status hiptensornet_create_tensor_svd_config(hiptensornet_handle h, hiptensornet_tensor_svd_config* o)
{
    roctensornet_tensor_svd_config r = nullptr;
    auto s = roctensornet_create_tensor_svd_config(to_roc(h), &r);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *o = to_hip(r);
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_destroy_tensor_svd_config(hiptensornet_tensor_svd_config c)
{ return to_hip_status(roctensornet_destroy_tensor_svd_config(to_roc(c))); }
hiptensornet_status hiptensornet_tensor_svd_config_get_attribute(
    hiptensornet_handle h, hiptensornet_tensor_svd_config c,
    hiptensornet_tensor_svd_config_attribute a, void* v, size_t s)
{
    return to_hip_status(roctensornet_tensor_svd_config_get_attribute(
        to_roc(h), to_roc(c),
        static_cast<roctensornet_tensor_svd_config_attribute>(static_cast<int>(a)),
        v, s));
}
hiptensornet_status hiptensornet_tensor_svd_config_set_attribute(
    hiptensornet_handle h, hiptensornet_tensor_svd_config c,
    hiptensornet_tensor_svd_config_attribute a, const void* v, size_t s)
{
    return to_hip_status(roctensornet_tensor_svd_config_set_attribute(
        to_roc(h), to_roc(c),
        static_cast<roctensornet_tensor_svd_config_attribute>(static_cast<int>(a)),
        v, s));
}
hiptensornet_status hiptensornet_create_tensor_svd_info(hiptensornet_handle h, hiptensornet_tensor_svd_info* o)
{
    roctensornet_tensor_svd_info r = nullptr;
    auto s = roctensornet_create_tensor_svd_info(to_roc(h), &r);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *o = to_hip(r);
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_destroy_tensor_svd_info(hiptensornet_tensor_svd_info i)
{ return to_hip_status(roctensornet_destroy_tensor_svd_info(to_roc(i))); }
hiptensornet_status hiptensornet_tensor_svd_info_get_attribute(
    hiptensornet_handle h, hiptensornet_tensor_svd_info i,
    hiptensornet_tensor_svd_info_attribute a, void* v, size_t s)
{
    return to_hip_status(roctensornet_tensor_svd_info_get_attribute(
        to_roc(h), to_roc(i),
        static_cast<roctensornet_tensor_svd_info_attribute>(static_cast<int>(a)),
        v, s));
}
hiptensornet_status hiptensornet_tensor_svd(hiptensornet_handle h, hiptensornet_tensor_descriptor din, const void* in, hiptensornet_tensor_descriptor du, void* u, void* sv, hiptensornet_tensor_descriptor dv, void* v, hiptensornet_tensor_svd_config c, hiptensornet_tensor_svd_info i, hiptensornet_workspace_descriptor w, hipStream_t st)
{ return to_hip_status(roctensornet_tensor_svd(to_roc(h), to_roc(din), in, to_roc(du), u, sv, to_roc(dv), v, to_roc(c), to_roc(i), to_roc(w), st)); }
hiptensornet_status hiptensornet_tensor_qr(hiptensornet_handle h, hiptensornet_tensor_descriptor din, const void* in, hiptensornet_tensor_descriptor dq, void* q, hiptensornet_tensor_descriptor dr, void* r, hiptensornet_workspace_descriptor w, hipStream_t st)
{ return to_hip_status(roctensornet_tensor_qr(to_roc(h), to_roc(din), in, to_roc(dq), q, to_roc(dr), r, to_roc(w), st)); }

/* ---- Gradient ---- */
hiptensornet_status hiptensornet_compute_gradients_backward(hiptensornet_handle h, hiptensornet_contraction_plan p, const void* const* in, const void* dout, void* const* dins, int32_t acc, hiptensornet_workspace_descriptor w, hipStream_t s)
{ return to_hip_status(roctensornet_compute_gradients_backward(to_roc(h), to_roc(p), in, dout, dins, acc, to_roc(w), s)); }

/* ---- Network operator ---- */
hiptensornet_status hiptensornet_create_network_operator(hiptensornet_handle h, int32_t nsm, const hiptensornet_index_t* ext, hiptensornet_data_type dt, hiptensornet_network_operator* o)
{
    roctensornet_network_operator r = nullptr;
    auto s = roctensornet_create_network_operator(to_roc(h), nsm,
        reinterpret_cast<const roctensornet_index_t*>(ext), to_roc_dtype(dt), &r);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *o = to_hip(r);
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_destroy_network_operator(hiptensornet_network_operator op)
{ return to_hip_status(roctensornet_destroy_network_operator(to_roc(op))); }
hiptensornet_status hiptensornet_network_operator_append_product(hiptensornet_handle h, hiptensornet_network_operator op, const void* coeff, int32_t nt, const int32_t* nm, const int32_t* const* modes, const hiptensornet_index_t* const* strides, const void* const* data, int64_t* cid)
{
    return to_hip_status(roctensornet_network_operator_append_product(
        to_roc(h), to_roc(op), coeff, nt, nm, modes,
        reinterpret_cast<const roctensornet_index_t* const*>(strides),
        data, cid));
}
hiptensornet_status hiptensornet_network_operator_append_mpo(
    hiptensornet_handle h, hiptensornet_network_operator op, const void* coefficient,
    int32_t num_state_modes, const int32_t* state_modes,
    const int32_t* tensor_mode_extents, const int64_t* tensor_mode_strides,
    const void* const* tensor_data, int32_t boundary_condition, int64_t* component_id)
{
    return to_hip_status(roctensornet_network_operator_append_mpo(
        to_roc(h), to_roc(op), coefficient, num_state_modes, state_modes,
        tensor_mode_extents, tensor_mode_strides, tensor_data,
        boundary_condition, component_id));
}

/* ---- Network state ---- */
hiptensornet_status hiptensornet_create_state(hiptensornet_handle h, hiptensornet_state_purity p, int32_t nsm, const hiptensornet_index_t* ext, hiptensornet_data_type dt, hiptensornet_state* o)
{
    roctensornet_state r = nullptr;
    auto s = roctensornet_create_state(to_roc(h),
        static_cast<roctensornet_state_purity>(static_cast<int>(p)),
        nsm, reinterpret_cast<const roctensornet_index_t*>(ext),
        to_roc_dtype(dt), &r);
    if(s == ROCTENSORNET_STATUS_SUCCESS) *o = to_hip(r);
    return to_hip_status(s);
}
hiptensornet_status hiptensornet_destroy_state(hiptensornet_state s)
{ return to_hip_status(roctensornet_destroy_state(to_roc(s))); }
hiptensornet_status hiptensornet_state_apply_tensor_operator(hiptensornet_handle h, hiptensornet_state s, int32_t nsm, const int32_t* sm, void* td, const int64_t* ts, int32_t im, int32_t adj, int32_t u, int64_t* tid)
{ return to_hip_status(roctensornet_state_apply_tensor_operator(to_roc(h), to_roc(s), nsm, sm, td, ts, im, adj, u, tid)); }
hiptensornet_status hiptensornet_state_apply_controlled_tensor_operator(
    hiptensornet_handle h, hiptensornet_state s,
    int32_t num_control_modes, const int32_t* control_modes, const int64_t* control_values,
    int32_t num_target_modes, const int32_t* target_modes,
    void* tensor_data, const int64_t* tensor_mode_strides,
    int32_t immutable, int32_t adjoint, int32_t unitary, int64_t* tensor_id)
{
    return to_hip_status(roctensornet_state_apply_controlled_tensor_operator(
        to_roc(h), to_roc(s), num_control_modes, control_modes, control_values,
        num_target_modes, target_modes, tensor_data, tensor_mode_strides,
        immutable, adjoint, unitary, tensor_id));
}
hiptensornet_status hiptensornet_state_apply_unitary_channel(
    hiptensornet_handle h, hiptensornet_state s,
    int32_t num_state_modes, const int32_t* state_modes,
    int32_t num_tensors, void* const* tensor_data,
    const int64_t* const* tensor_mode_strides, const double* probabilities,
    int64_t* channel_id)
{
    return to_hip_status(roctensornet_state_apply_unitary_channel(
        to_roc(h), to_roc(s), num_state_modes, state_modes, num_tensors,
        tensor_data, tensor_mode_strides, probabilities, channel_id));
}
hiptensornet_status hiptensornet_state_apply_general_channel(
    hiptensornet_handle h, hiptensornet_state s,
    int32_t num_state_modes, const int32_t* state_modes,
    int32_t num_tensors, void* const* tensor_data,
    const int64_t* const* tensor_mode_strides, int64_t* channel_id)
{
    return to_hip_status(roctensornet_state_apply_general_channel(
        to_roc(h), to_roc(s), num_state_modes, state_modes, num_tensors,
        tensor_data, tensor_mode_strides, channel_id));
}
hiptensornet_status hiptensornet_state_update_tensor_operator(
    hiptensornet_handle h, hiptensornet_state s, int64_t tensor_id, void* tensor_data, int32_t unitary)
{
    return to_hip_status(roctensornet_state_update_tensor_operator(
        to_roc(h), to_roc(s), tensor_id, tensor_data, unitary));
}
hiptensornet_status hiptensornet_state_configure(
    hiptensornet_handle h, hiptensornet_state s, hiptensornet_state_attribute a, const void* v, size_t sz)
{
    return to_hip_status(roctensornet_state_configure(
        to_roc(h), to_roc(s),
        static_cast<roctensornet_state_attribute>(static_cast<int>(a)), v, sz));
}
hiptensornet_status hiptensornet_state_get_info(
    hiptensornet_handle h, hiptensornet_state s, hiptensornet_state_attribute a, void* v, size_t sz)
{
    return to_hip_status(roctensornet_state_get_info(
        to_roc(h), to_roc(s),
        static_cast<roctensornet_state_attribute>(static_cast<int>(a)), v, sz));
}
hiptensornet_status hiptensornet_state_prepare(
    hiptensornet_handle h, hiptensornet_state s, size_t max_workspace_size_device,
    hiptensornet_workspace_descriptor workspace, hipStream_t stream)
{
    return to_hip_status(roctensornet_state_prepare(
        to_roc(h), to_roc(s), max_workspace_size_device, to_roc(workspace), stream));
}
hiptensornet_status hiptensornet_state_compute(
    hiptensornet_handle h, hiptensornet_state s,
    hiptensornet_workspace_descriptor workspace, void* const* state_tensors_out, hipStream_t stream)
{
    return to_hip_status(roctensornet_state_compute(
        to_roc(h), to_roc(s), to_roc(workspace), state_tensors_out, stream));
}

/* ---- Derived ---- */
hiptensornet_status hiptensornet_create_marginal(hiptensornet_handle h, hiptensornet_state s, int32_t nm, const int32_t* mm, int32_t npm, const int32_t* pm, const int64_t* mts, hiptensornet_state_marginal* o)
{
    roctensornet_state_marginal r = nullptr;
    auto rs = roctensornet_create_marginal(to_roc(h), to_roc(s), nm, mm, npm, pm, mts, &r);
    if(rs == ROCTENSORNET_STATUS_SUCCESS) *o = to_hip(r);
    return to_hip_status(rs);
}
hiptensornet_status hiptensornet_destroy_marginal(hiptensornet_state_marginal m)
{ return to_hip_status(roctensornet_destroy_marginal(to_roc(m))); }
hiptensornet_status hiptensornet_marginal_configure(
    hiptensornet_handle h, hiptensornet_state_marginal m,
    hiptensornet_marginal_attribute a, const void* v, size_t sz)
{
    return to_hip_status(roctensornet_marginal_configure(
        to_roc(h), to_roc(m),
        static_cast<roctensornet_marginal_attribute>(static_cast<int>(a)), v, sz));
}
hiptensornet_status hiptensornet_marginal_get_info(
    hiptensornet_handle h, hiptensornet_state_marginal m,
    hiptensornet_marginal_attribute a, void* v, size_t sz)
{
    return to_hip_status(roctensornet_marginal_get_info(
        to_roc(h), to_roc(m),
        static_cast<roctensornet_marginal_attribute>(static_cast<int>(a)), v, sz));
}
hiptensornet_status hiptensornet_marginal_prepare(
    hiptensornet_handle h, hiptensornet_state_marginal m,
    size_t max_workspace_size_device, hiptensornet_workspace_descriptor workspace, hipStream_t stream)
{
    return to_hip_status(roctensornet_marginal_prepare(
        to_roc(h), to_roc(m), max_workspace_size_device, to_roc(workspace), stream));
}
hiptensornet_status hiptensornet_marginal_compute(
    hiptensornet_handle h, hiptensornet_state_marginal m,
    const int64_t* projected_mode_values, hiptensornet_workspace_descriptor workspace,
    void* marginal_tensor, hipStream_t stream)
{
    return to_hip_status(roctensornet_marginal_compute(
        to_roc(h), to_roc(m), projected_mode_values, to_roc(workspace),
        marginal_tensor, stream));
}
hiptensornet_status hiptensornet_create_sampler(hiptensornet_handle h, hiptensornet_state s, int32_t n, const int32_t* mts, hiptensornet_state_sampler* o)
{
    roctensornet_state_sampler r = nullptr;
    auto rs = roctensornet_create_sampler(to_roc(h), to_roc(s), n, mts, &r);
    if(rs == ROCTENSORNET_STATUS_SUCCESS) *o = to_hip(r);
    return to_hip_status(rs);
}
hiptensornet_status hiptensornet_destroy_sampler(hiptensornet_state_sampler s)
{ return to_hip_status(roctensornet_destroy_sampler(to_roc(s))); }
hiptensornet_status hiptensornet_sampler_configure(
    hiptensornet_handle h, hiptensornet_state_sampler s,
    hiptensornet_sampler_attribute a, const void* v, size_t sz)
{
    return to_hip_status(roctensornet_sampler_configure(
        to_roc(h), to_roc(s),
        static_cast<roctensornet_sampler_attribute>(static_cast<int>(a)), v, sz));
}
hiptensornet_status hiptensornet_sampler_get_info(
    hiptensornet_handle h, hiptensornet_state_sampler s,
    hiptensornet_sampler_attribute a, void* v, size_t sz)
{
    return to_hip_status(roctensornet_sampler_get_info(
        to_roc(h), to_roc(s),
        static_cast<roctensornet_sampler_attribute>(static_cast<int>(a)), v, sz));
}
hiptensornet_status hiptensornet_sampler_prepare(
    hiptensornet_handle h, hiptensornet_state_sampler s,
    size_t max_workspace_size_device, hiptensornet_workspace_descriptor workspace, hipStream_t stream)
{
    return to_hip_status(roctensornet_sampler_prepare(
        to_roc(h), to_roc(s), max_workspace_size_device, to_roc(workspace), stream));
}
hiptensornet_status hiptensornet_sampler_sample(
    hiptensornet_handle h, hiptensornet_state_sampler s,
    int64_t num_shots, hiptensornet_workspace_descriptor workspace, int64_t* samples, hipStream_t stream)
{
    return to_hip_status(roctensornet_sampler_sample(
        to_roc(h), to_roc(s), num_shots, to_roc(workspace), samples, stream));
}
hiptensornet_status hiptensornet_create_expectation(hiptensornet_handle h, hiptensornet_state s, hiptensornet_network_operator op, hiptensornet_state_expectation* o)
{
    roctensornet_state_expectation r = nullptr;
    auto rs = roctensornet_create_expectation(to_roc(h), to_roc(s), to_roc(op), &r);
    if(rs == ROCTENSORNET_STATUS_SUCCESS) *o = to_hip(r);
    return to_hip_status(rs);
}
hiptensornet_status hiptensornet_destroy_expectation(hiptensornet_state_expectation e)
{ return to_hip_status(roctensornet_destroy_expectation(to_roc(e))); }
hiptensornet_status hiptensornet_expectation_configure(
    hiptensornet_handle h, hiptensornet_state_expectation e,
    hiptensornet_expectation_attribute a, const void* v, size_t sz)
{
    return to_hip_status(roctensornet_expectation_configure(
        to_roc(h), to_roc(e),
        static_cast<roctensornet_expectation_attribute>(static_cast<int>(a)), v, sz));
}
hiptensornet_status hiptensornet_expectation_get_info(
    hiptensornet_handle h, hiptensornet_state_expectation e,
    hiptensornet_expectation_attribute a, void* v, size_t sz)
{
    return to_hip_status(roctensornet_expectation_get_info(
        to_roc(h), to_roc(e),
        static_cast<roctensornet_expectation_attribute>(static_cast<int>(a)), v, sz));
}
hiptensornet_status hiptensornet_expectation_prepare(
    hiptensornet_handle h, hiptensornet_state_expectation e,
    size_t max_workspace_size_device, hiptensornet_workspace_descriptor workspace, hipStream_t stream)
{
    return to_hip_status(roctensornet_expectation_prepare(
        to_roc(h), to_roc(e), max_workspace_size_device, to_roc(workspace), stream));
}
hiptensornet_status hiptensornet_expectation_compute(
    hiptensornet_handle h, hiptensornet_state_expectation e,
    hiptensornet_workspace_descriptor workspace, void* expectation_value, void* state_norm, hipStream_t stream)
{
    return to_hip_status(roctensornet_expectation_compute(
        to_roc(h), to_roc(e), to_roc(workspace), expectation_value, state_norm, stream));
}
hiptensornet_status hiptensornet_create_accessor(hiptensornet_handle h, hiptensornet_state s, int32_t npm, const int32_t* pm, const int64_t* ats, hiptensornet_state_accessor* o)
{
    roctensornet_state_accessor r = nullptr;
    auto rs = roctensornet_create_accessor(to_roc(h), to_roc(s), npm, pm, ats, &r);
    if(rs == ROCTENSORNET_STATUS_SUCCESS) *o = to_hip(r);
    return to_hip_status(rs);
}
hiptensornet_status hiptensornet_destroy_accessor(hiptensornet_state_accessor a)
{ return to_hip_status(roctensornet_destroy_accessor(to_roc(a))); }
hiptensornet_status hiptensornet_accessor_configure(
    hiptensornet_handle h, hiptensornet_state_accessor a,
    hiptensornet_accessor_attribute attr, const void* v, size_t sz)
{
    return to_hip_status(roctensornet_accessor_configure(
        to_roc(h), to_roc(a),
        static_cast<roctensornet_accessor_attribute>(static_cast<int>(attr)), v, sz));
}
hiptensornet_status hiptensornet_accessor_get_info(
    hiptensornet_handle h, hiptensornet_state_accessor a,
    hiptensornet_accessor_attribute attr, void* v, size_t sz)
{
    return to_hip_status(roctensornet_accessor_get_info(
        to_roc(h), to_roc(a),
        static_cast<roctensornet_accessor_attribute>(static_cast<int>(attr)), v, sz));
}
hiptensornet_status hiptensornet_accessor_prepare(
    hiptensornet_handle h, hiptensornet_state_accessor a,
    size_t max_workspace_size_device, hiptensornet_workspace_descriptor workspace, hipStream_t stream)
{
    return to_hip_status(roctensornet_accessor_prepare(
        to_roc(h), to_roc(a), max_workspace_size_device, to_roc(workspace), stream));
}
hiptensornet_status hiptensornet_accessor_compute(
    hiptensornet_handle h, hiptensornet_state_accessor a,
    const int64_t* projected_mode_values, hiptensornet_workspace_descriptor workspace,
    void* amplitudes_tensor, void* state_norm, hipStream_t stream)
{
    return to_hip_status(roctensornet_accessor_compute(
        to_roc(h), to_roc(a), projected_mode_values, to_roc(workspace),
        amplitudes_tensor, state_norm, stream));
}

} // extern "C"
