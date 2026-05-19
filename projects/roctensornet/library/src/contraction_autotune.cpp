/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Contraction autotune preference + execution.
 *
 * v0.1.0 supports a single contraction algorithm (the generic loop-nest
 * kernel in `roctensornet_kernels.hpp`), so autotune is a no-op that
 * exists for API completeness. The function call does invoke a single
 * contraction pass to warm caches and to verify the plan is runnable.
 * ************************************************************************ */

#include "roctensornet_descriptors.hpp"

using namespace roctensornet;

extern "C" {

roctensornet_status
roctensornet_create_contraction_autotune_preference(
    roctensornet_handle h,
    roctensornet_contraction_autotune_preference* out)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    ROCTENSORNET_CHECK_PTR(out);
    auto* p = new (std::nothrow) contraction_autotune_preference_st();
    if(p == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    *out = reinterpret_cast<roctensornet_contraction_autotune_preference>(p);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_destroy_contraction_autotune_preference(
    roctensornet_contraction_autotune_preference p)
{
    delete cast<contraction_autotune_preference_st>(p);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_contraction_autotune_preference_get_attribute(
    roctensornet_handle                                     h,
    roctensornet_contraction_autotune_preference            p,
    roctensornet_contraction_autotune_preference_attribute  attr,
    void*                                                   value,
    size_t                                                  size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* pp = cast<contraction_autotune_preference_st>(p);
    if(pp == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
    switch(attr)
    {
    case ROCTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS:
        *static_cast<int32_t*>(value) = pp->max_iterations;     return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_CONTRACTION_AUTOTUNE_INTERMEDIATE_MODES:
        *static_cast<int32_t*>(value) = pp->intermediate_modes; return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_CONTRACTION_AUTOTUNE_GEMM_ALGORITHM:
        *static_cast<int32_t*>(value) = pp->gemm_algorithm;     return ROCTENSORNET_STATUS_SUCCESS;
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_contraction_autotune_preference_set_attribute(
    roctensornet_handle                                     h,
    roctensornet_contraction_autotune_preference            p,
    roctensornet_contraction_autotune_preference_attribute  attr,
    const void*                                             value,
    size_t                                                  size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* pp = cast<contraction_autotune_preference_st>(p);
    if(pp == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
    int32_t v = *static_cast<const int32_t*>(value);
    switch(attr)
    {
    case ROCTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS:    pp->max_iterations    = v; return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_CONTRACTION_AUTOTUNE_INTERMEDIATE_MODES: pp->intermediate_modes = v; return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_CONTRACTION_AUTOTUNE_GEMM_ALGORITHM:    pp->gemm_algorithm    = v; return ROCTENSORNET_STATUS_SUCCESS;
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

/* Defined in contraction_execute.cpp; we forward through it. */
roctensornet_status
roctensornet_contraction(roctensornet_handle               handle,
                         roctensornet_contraction_plan     plan,
                         const void* const*                raw_data_in,
                         void*                             raw_data_out,
                         roctensornet_workspace_descriptor workspace,
                         int64_t                           slice_id,
                         hipStream_t                       stream);

roctensornet_status
roctensornet_contraction_autotune(roctensornet_handle               handle,
                                  roctensornet_contraction_plan     plan,
                                  const void* const*                raw_data_in,
                                  void*                             raw_data_out,
                                  roctensornet_workspace_descriptor workspace,
                                  roctensornet_contraction_autotune_preference prefs,
                                  hipStream_t                       stream)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    (void)prefs;
    /* Single dry run; result is correct but we discard it (the caller
     * is expected to invoke `contraction()` for real work). For an
     * idempotent plan this is harmless. */
    return roctensornet_contraction(handle, plan, raw_data_in, raw_data_out,
                                    workspace, /*slice=*/0, stream);
}

} // extern "C"
