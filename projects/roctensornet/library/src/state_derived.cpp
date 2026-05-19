/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Network-state-derived computations:
 *   - Marginal:   tr_{not marginal modes}(|psi><psi|)
 *   - Sampler:    sample marginal probabilities
 *   - Expectation: <psi| H |psi>
 *   - Accessor:    <projection|psi>
 *
 * In v0.1.0 the lifecycle, configure/get-info, and prepare entries are
 * all wired and round-trip faithfully. The compute/sample entries
 * return NOT_SUPPORTED with the same rationale as state_compute: the
 * networking-descriptor synthesizer that turns the gate list +
 * marginal/sampler/expectation pattern into a contraction network is
 * a follow-up. The configure/get-info path is exercised by the test
 * suite to verify API conformance.
 * ************************************************************************ */

#include "roctensornet_descriptors.hpp"

using namespace roctensornet;

extern "C" {

/* ============================== MARGINAL ============================== */
roctensornet_status
roctensornet_create_marginal(roctensornet_handle h, roctensornet_state state,
                             int32_t num_marginal_modes, const int32_t* marginal_modes,
                             int32_t num_projected_modes, const int32_t* projected_modes,
                             const int64_t* marginal_tensor_strides,
                             roctensornet_state_marginal* out)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    ROCTENSORNET_CHECK_PTR(out);
    if(state == nullptr) return ROCTENSORNET_STATUS_INVALID_VALUE;
    auto* m = new (std::nothrow) state_marginal_st();
    if(m == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    m->state = state;
    if(num_marginal_modes > 0)  m->marginal_modes.assign(marginal_modes,  marginal_modes  + num_marginal_modes);
    if(num_projected_modes > 0) m->projected_modes.assign(projected_modes, projected_modes + num_projected_modes);
    if(marginal_tensor_strides != nullptr)
        m->marginal_tensor_strides.assign(marginal_tensor_strides,
                                          marginal_tensor_strides + 2 * num_marginal_modes);
    *out = reinterpret_cast<roctensornet_state_marginal>(m);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_destroy_marginal(roctensornet_state_marginal m)
{ delete cast<state_marginal_st>(m); return ROCTENSORNET_STATUS_SUCCESS; }

roctensornet_status
roctensornet_marginal_configure(roctensornet_handle h, roctensornet_state_marginal m,
                                roctensornet_marginal_attribute attr, const void* value, size_t size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* mm = cast<state_marginal_st>(m);
    if(mm == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    if(attr == ROCTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES)
    {
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        mm->num_hyper_samples = *static_cast<const int32_t*>(value);
        return ROCTENSORNET_STATUS_SUCCESS;
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_marginal_get_info(roctensornet_handle h, roctensornet_state_marginal m,
                               roctensornet_marginal_attribute attr, void* value, size_t size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* mm = cast<state_marginal_st>(m);
    if(mm == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    switch(attr)
    {
    case ROCTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int32_t*>(value) = mm->num_hyper_samples; return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_MARGINAL_INFO_FLOPS:
        if(size < sizeof(size_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<size_t*>(value) = mm->flops; return ROCTENSORNET_STATUS_SUCCESS;
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_marginal_prepare(roctensornet_handle h, roctensornet_state_marginal m,
                              size_t max_workspace_size_device,
                              roctensornet_workspace_descriptor workspace, hipStream_t stream)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* mm = cast<state_marginal_st>(m);
    if(mm == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    (void)max_workspace_size_device; (void)workspace; (void)stream;
    /* roctensornet_marginal_compute returns NOT_SUPPORTED in v0.1.0;
     * keep prepare consistent so callers do not silently advance to
     * a no-op compute. */
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_marginal_compute(roctensornet_handle h, roctensornet_state_marginal m,
                              const int64_t* projected_mode_values,
                              roctensornet_workspace_descriptor workspace,
                              void* marginal_tensor, hipStream_t stream)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    (void)m; (void)projected_mode_values; (void)workspace; (void)marginal_tensor; (void)stream;
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

/* ============================== SAMPLER ============================== */
roctensornet_status
roctensornet_create_sampler(roctensornet_handle h, roctensornet_state state,
                            int32_t num_modes_to_sample, const int32_t* modes_to_sample,
                            roctensornet_state_sampler* out)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    ROCTENSORNET_CHECK_PTR(out);
    if(state == nullptr) return ROCTENSORNET_STATUS_INVALID_VALUE;
    auto* sp = new (std::nothrow) state_sampler_st();
    if(sp == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    sp->state = state;
    if(num_modes_to_sample > 0)
        sp->modes_to_sample.assign(modes_to_sample, modes_to_sample + num_modes_to_sample);
    *out = reinterpret_cast<roctensornet_state_sampler>(sp);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_destroy_sampler(roctensornet_state_sampler s)
{ delete cast<state_sampler_st>(s); return ROCTENSORNET_STATUS_SUCCESS; }

roctensornet_status
roctensornet_sampler_configure(roctensornet_handle h, roctensornet_state_sampler s,
                               roctensornet_sampler_attribute attr, const void* value, size_t size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* sp = cast<state_sampler_st>(s);
    if(sp == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    switch(attr)
    {
    case ROCTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        sp->num_hyper_samples = *static_cast<const int32_t*>(value); return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_SAMPLER_CONFIG_DETERMINISTIC:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        sp->deterministic = *static_cast<const int32_t*>(value); return ROCTENSORNET_STATUS_SUCCESS;
    default:
        break;
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_sampler_get_info(roctensornet_handle h, roctensornet_state_sampler s,
                              roctensornet_sampler_attribute attr, void* value, size_t size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* sp = cast<state_sampler_st>(s);
    if(sp == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    switch(attr)
    {
    case ROCTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int32_t*>(value) = sp->num_hyper_samples; return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_SAMPLER_CONFIG_DETERMINISTIC:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int32_t*>(value) = sp->deterministic; return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_SAMPLER_INFO_FLOPS:
        if(size < sizeof(size_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<size_t*>(value) = sp->flops; return ROCTENSORNET_STATUS_SUCCESS;
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_sampler_prepare(roctensornet_handle h, roctensornet_state_sampler s,
                             size_t max_workspace_size_device,
                             roctensornet_workspace_descriptor workspace, hipStream_t stream)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* sp = cast<state_sampler_st>(s);
    if(sp == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    (void)max_workspace_size_device; (void)workspace; (void)stream;
    /* roctensornet_sampler_sample returns NOT_SUPPORTED in v0.1.0;
     * keep prepare consistent so callers do not silently advance. */
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_sampler_sample(roctensornet_handle h, roctensornet_state_sampler s,
                            int64_t num_shots, roctensornet_workspace_descriptor workspace,
                            int64_t* samples, hipStream_t stream)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    (void)s; (void)num_shots; (void)workspace; (void)samples; (void)stream;
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

/* ============================ EXPECTATION ============================ */
roctensornet_status
roctensornet_create_expectation(roctensornet_handle h, roctensornet_state state,
                                roctensornet_network_operator op,
                                roctensornet_state_expectation* out)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    ROCTENSORNET_CHECK_PTR(out);
    if(state == nullptr) return ROCTENSORNET_STATUS_INVALID_VALUE;
    auto* e = new (std::nothrow) state_expectation_st();
    if(e == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    e->state = state;
    e->op    = op;
    *out = reinterpret_cast<roctensornet_state_expectation>(e);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_destroy_expectation(roctensornet_state_expectation e)
{ delete cast<state_expectation_st>(e); return ROCTENSORNET_STATUS_SUCCESS; }

roctensornet_status
roctensornet_expectation_configure(roctensornet_handle h, roctensornet_state_expectation e,
                                   roctensornet_expectation_attribute attr,
                                   const void* value, size_t size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* ee = cast<state_expectation_st>(e);
    if(ee == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    if(attr == ROCTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES)
    {
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        ee->num_hyper_samples = *static_cast<const int32_t*>(value);
        return ROCTENSORNET_STATUS_SUCCESS;
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_expectation_get_info(roctensornet_handle h, roctensornet_state_expectation e,
                                  roctensornet_expectation_attribute attr, void* value, size_t size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* ee = cast<state_expectation_st>(e);
    if(ee == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    switch(attr)
    {
    case ROCTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int32_t*>(value) = ee->num_hyper_samples; return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_EXPECTATION_INFO_FLOPS:
        if(size < sizeof(size_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<size_t*>(value) = ee->flops; return ROCTENSORNET_STATUS_SUCCESS;
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_expectation_prepare(roctensornet_handle h, roctensornet_state_expectation e,
                                 size_t max_workspace_size_device,
                                 roctensornet_workspace_descriptor workspace, hipStream_t stream)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* ee = cast<state_expectation_st>(e);
    if(ee == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    (void)max_workspace_size_device; (void)workspace; (void)stream;
    /* roctensornet_expectation_compute returns NOT_SUPPORTED in v0.1.0;
     * keep prepare consistent so callers do not silently advance. */
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_expectation_compute(roctensornet_handle h, roctensornet_state_expectation e,
                                 roctensornet_workspace_descriptor workspace,
                                 void* expectation_value, void* state_norm, hipStream_t stream)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    (void)e; (void)workspace; (void)expectation_value; (void)state_norm; (void)stream;
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

/* ============================== ACCESSOR ============================== */
roctensornet_status
roctensornet_create_accessor(roctensornet_handle h, roctensornet_state state,
                             int32_t num_projected_modes, const int32_t* projected_modes,
                             const int64_t* amplitudes_tensor_strides,
                             roctensornet_state_accessor* out)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    ROCTENSORNET_CHECK_PTR(out);
    if(state == nullptr) return ROCTENSORNET_STATUS_INVALID_VALUE;
    auto* a = new (std::nothrow) state_accessor_st();
    if(a == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    a->state = state;
    if(num_projected_modes > 0)
        a->projected_modes.assign(projected_modes, projected_modes + num_projected_modes);
    if(amplitudes_tensor_strides != nullptr)
    {
        auto* st = cast<network_state_st>(state);
        if(st != nullptr)
            a->amplitudes_tensor_strides.assign(
                amplitudes_tensor_strides,
                amplitudes_tensor_strides + 2 * (st->num_state_modes - num_projected_modes));
    }
    *out = reinterpret_cast<roctensornet_state_accessor>(a);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_destroy_accessor(roctensornet_state_accessor a)
{ delete cast<state_accessor_st>(a); return ROCTENSORNET_STATUS_SUCCESS; }

roctensornet_status
roctensornet_accessor_configure(roctensornet_handle h, roctensornet_state_accessor a,
                                roctensornet_accessor_attribute attr, const void* value, size_t size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* aa = cast<state_accessor_st>(a);
    if(aa == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    if(attr == ROCTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES)
    {
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        aa->num_hyper_samples = *static_cast<const int32_t*>(value);
        return ROCTENSORNET_STATUS_SUCCESS;
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_accessor_get_info(roctensornet_handle h, roctensornet_state_accessor a,
                               roctensornet_accessor_attribute attr, void* value, size_t size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* aa = cast<state_accessor_st>(a);
    if(aa == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    switch(attr)
    {
    case ROCTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int32_t*>(value) = aa->num_hyper_samples; return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_ACCESSOR_INFO_FLOPS:
        if(size < sizeof(size_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<size_t*>(value) = aa->flops; return ROCTENSORNET_STATUS_SUCCESS;
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_accessor_prepare(roctensornet_handle h, roctensornet_state_accessor a,
                              size_t max_workspace_size_device,
                              roctensornet_workspace_descriptor workspace, hipStream_t stream)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* aa = cast<state_accessor_st>(a);
    if(aa == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    (void)max_workspace_size_device; (void)workspace; (void)stream;
    /* roctensornet_accessor_compute returns NOT_SUPPORTED in v0.1.0;
     * keep prepare consistent so callers do not silently advance. */
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_accessor_compute(roctensornet_handle h, roctensornet_state_accessor a,
                              const int64_t* projected_mode_values,
                              roctensornet_workspace_descriptor workspace,
                              void* amplitudes_tensor, void* state_norm, hipStream_t stream)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    (void)a; (void)projected_mode_values; (void)workspace;
    (void)amplitudes_tensor; (void)state_norm; (void)stream;
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

} // extern "C"
