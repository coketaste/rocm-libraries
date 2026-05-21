/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "rocdensitymat_descriptors.hpp"
#include "rocdensitymat_kernels.hpp"

#include <cmath>

namespace
{

template <typename Cmplx>
rocdensitymat_status fill_state(rocdensitymat_handle h,
                                rocdensitymat_state state,
                                Cmplx value)
{
    if(state->component_buffer == nullptr)
        return ROCDENSITYMAT_STATUS_NOT_INITIALIZED;
    int64_t n  = state->component_elems;
    int     bs = rocdensitymat::default_threads_per_block;
    int64_t gx = rocdensitymat::ceil_div<int64_t>(n, bs);

    auto* buf = static_cast<Cmplx*>(state->component_buffer);
    hipLaunchKernelGGL(rocdensitymat::fill_kernel<Cmplx>,
                       dim3(static_cast<unsigned int>(gx)),
                       dim3(bs),
                       0,
                       h->stream,
                       buf,
                       value,
                       n);
    return (hipGetLastError() == hipSuccess) ? ROCDENSITYMAT_STATUS_SUCCESS
                                             : ROCDENSITYMAT_STATUS_EXECUTION_FAILED;
}

template <typename Cmplx>
rocdensitymat_status set_one_element_to_one(rocdensitymat_handle h,
                                            rocdensitymat_state state,
                                            int64_t idx)
{
    Cmplx one;
    one.x = 1;
    one.y = 0;
    auto* buf = static_cast<Cmplx*>(state->component_buffer) + idx;
    if(hipMemcpyAsync(buf, &one, sizeof(Cmplx), hipMemcpyHostToDevice, h->stream)
       != hipSuccess)
        return ROCDENSITYMAT_STATUS_HIP_ERROR;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

} // namespace

extern "C" {

rocdensitymat_status rocdensitymat_create_state(
    rocdensitymat_handle handle,
    rocdensitymat_state_purity purity,
    int32_t num_space_modes,
    const int64_t* space_shape,
    int64_t batch_size,
    rocdensitymat_data_type data_type,
    rocdensitymat_state* state)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(state);
    if(num_space_modes <= 0 || space_shape == nullptr)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    if(purity == ROCDENSITYMAT_STATE_PURITY_MPS)
        return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;
    if(purity != ROCDENSITYMAT_STATE_PURITY_PURE
       && purity != ROCDENSITYMAT_STATE_PURITY_MIXED)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    if(batch_size <= 0) return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    if(batch_size != 1) return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;

    auto elem_bytes = rocdensitymat::element_size_bytes(data_type);
    if(elem_bytes == 0
       || (data_type != ROCDENSITYMAT_C_64F && data_type != ROCDENSITYMAT_C_32F))
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;

    auto* s = new(std::nothrow) _rocdensitymat_state();
    if(s == nullptr) return ROCDENSITYMAT_STATUS_ALLOC_FAILED;

    s->purity     = purity;
    s->data_type  = data_type;
    s->num_modes  = num_space_modes;
    s->space_shape.assign(space_shape, space_shape + num_space_modes);
    s->batch_size = batch_size;

    int64_t hdim = s->hilbert_dim();
    s->component_elems = (purity == ROCDENSITYMAT_STATE_PURITY_PURE)
                             ? hdim
                             : (hdim * hdim);
    s->component_bytes = static_cast<size_t>(s->component_elems) * elem_bytes;

    *state = s;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_destroy_state(rocdensitymat_state state)
{
    if(state == nullptr) return ROCDENSITYMAT_STATUS_SUCCESS;
    delete state;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_state_get_num_components(
    rocdensitymat_handle handle,
    rocdensitymat_state state,
    int32_t* num_components)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(state);
    ROCDENSITYMAT_CHECK_PTR(num_components);
    *num_components = 1;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_state_get_component_info(
    rocdensitymat_handle handle,
    rocdensitymat_state state,
    int32_t component_id,
    int32_t* num_modes,
    int64_t* mode_extents,
    size_t* component_size_bytes)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(state);
    if(component_id != 0) return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    if(num_modes != nullptr)
    {
        *num_modes = (state->purity == ROCDENSITYMAT_STATE_PURITY_PURE)
                         ? state->num_modes
                         : (state->num_modes * 2);
    }
    if(mode_extents != nullptr)
    {
        for(int32_t i = 0; i < state->num_modes; ++i)
            mode_extents[i] = state->space_shape[i];
        if(state->purity == ROCDENSITYMAT_STATE_PURITY_MIXED)
        {
            for(int32_t i = 0; i < state->num_modes; ++i)
                mode_extents[state->num_modes + i] = state->space_shape[i];
        }
    }
    if(component_size_bytes != nullptr)
        *component_size_bytes = state->component_bytes;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_state_attach_component_buffer(
    rocdensitymat_handle handle,
    rocdensitymat_state state,
    int32_t component_id,
    void* component_buffer,
    size_t component_buffer_size)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(state);
    if(component_id != 0) return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    if(component_buffer == nullptr) return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    if(component_buffer_size < state->component_bytes)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    state->component_buffer       = component_buffer;
    state->component_buffer_bytes = component_buffer_size;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_state_initialize_zero(
    rocdensitymat_handle handle, rocdensitymat_state state)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(state);
    if(state->data_type == ROCDENSITYMAT_C_64F)
    {
        hipDoubleComplex z;
        z.x = 0; z.y = 0;
        return fill_state<hipDoubleComplex>(handle, state, z);
    }
    if(state->data_type == ROCDENSITYMAT_C_32F)
    {
        hipFloatComplex z;
        z.x = 0; z.y = 0;
        return fill_state<hipFloatComplex>(handle, state, z);
    }
    return ROCDENSITYMAT_STATUS_INVALID_VALUE;
}

rocdensitymat_status rocdensitymat_state_initialize_uniform(
    rocdensitymat_handle handle, rocdensitymat_state state)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(state);
    int64_t hdim = state->hilbert_dim();
    if(hdim <= 0) return ROCDENSITYMAT_STATUS_INVALID_VALUE;

    if(state->purity == ROCDENSITYMAT_STATE_PURITY_PURE)
    {
        // Uniform pure state: amplitude 1/sqrt(D) on every basis vector.
        double amp = 1.0 / std::sqrt(static_cast<double>(hdim));
        if(state->data_type == ROCDENSITYMAT_C_64F)
        {
            hipDoubleComplex z;
            z.x = amp; z.y = 0;
            return fill_state<hipDoubleComplex>(handle, state, z);
        }
        hipFloatComplex z;
        z.x = static_cast<float>(amp); z.y = 0;
        return fill_state<hipFloatComplex>(handle, state, z);
    }

    // Mixed state: maximally-mixed = (1/D) I. Zero everything, then write
    // 1/D on the diagonal (indices i*D + i, i in [0,D)).
    rocdensitymat_status rc = rocdensitymat_state_initialize_zero(handle, state);
    if(rc != ROCDENSITYMAT_STATUS_SUCCESS) return rc;
    double inv = 1.0 / static_cast<double>(hdim);
    if(state->data_type == ROCDENSITYMAT_C_64F)
    {
        hipDoubleComplex one_over_d;
        one_over_d.x = inv; one_over_d.y = 0;
        auto* buf = static_cast<hipDoubleComplex*>(state->component_buffer);
        for(int64_t i = 0; i < hdim; ++i)
        {
            if(hipMemcpyAsync(buf + i * hdim + i, &one_over_d,
                              sizeof(hipDoubleComplex),
                              hipMemcpyHostToDevice, handle->stream) != hipSuccess)
                return ROCDENSITYMAT_STATUS_HIP_ERROR;
        }
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    hipFloatComplex one_over_d;
    one_over_d.x = static_cast<float>(inv); one_over_d.y = 0;
    auto* buf = static_cast<hipFloatComplex*>(state->component_buffer);
    for(int64_t i = 0; i < hdim; ++i)
    {
        if(hipMemcpyAsync(buf + i * hdim + i, &one_over_d,
                          sizeof(hipFloatComplex),
                          hipMemcpyHostToDevice, handle->stream) != hipSuccess)
            return ROCDENSITYMAT_STATUS_HIP_ERROR;
    }
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_state_initialize_basis(
    rocdensitymat_handle handle, rocdensitymat_state state,
    const int64_t* basis_indices)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(state);
    ROCDENSITYMAT_CHECK_PTR(basis_indices);

    int64_t flat = 0;
    int64_t stride = 1;
    for(int32_t k = state->num_modes - 1; k >= 0; --k)
    {
        if(basis_indices[k] < 0 || basis_indices[k] >= state->space_shape[k])
            return ROCDENSITYMAT_STATUS_INVALID_VALUE;
        flat   += basis_indices[k] * stride;
        stride *= state->space_shape[k];
    }
    auto rc = rocdensitymat_state_initialize_zero(handle, state);
    if(rc != ROCDENSITYMAT_STATUS_SUCCESS) return rc;

    int64_t target = (state->purity == ROCDENSITYMAT_STATE_PURITY_PURE)
                         ? flat
                         : flat * state->hilbert_dim() + flat;

    if(state->data_type == ROCDENSITYMAT_C_64F)
        return set_one_element_to_one<hipDoubleComplex>(handle, state, target);
    return set_one_element_to_one<hipFloatComplex>(handle, state, target);
}

} // extern "C"
