/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Network operator: a sum of products / MPOs over state modes.
 *
 * The operator is purely descriptive bookkeeping in v0.1.0; it is
 * consumed by `roctensornet_expectation_compute` which builds a
 * NetworkDescriptor that contracts <state | op | state>.
 * ************************************************************************ */

#include "roctensornet_descriptors.hpp"

#include <cstdlib>
#include <cstring>

using namespace roctensornet;

extern "C" {

roctensornet_status
roctensornet_create_network_operator(roctensornet_handle handle,
                                     int32_t                       num_state_modes,
                                     const roctensornet_index_t*   state_mode_extents,
                                     roctensornet_data_type        data_type,
                                     roctensornet_network_operator* out)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    ROCTENSORNET_CHECK_PTR(out);
    if(num_state_modes < 0) return ROCTENSORNET_STATUS_INVALID_VALUE;
    if(num_state_modes > 0) ROCTENSORNET_CHECK_PTR(state_mode_extents);
    if(element_size_bytes(data_type) == 0) return ROCTENSORNET_STATUS_NOT_SUPPORTED;
    auto* op = new (std::nothrow) network_operator_st();
    if(op == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    op->num_state_modes = num_state_modes;
    op->state_mode_extents.assign(state_mode_extents, state_mode_extents + num_state_modes);
    op->data_type = data_type;
    *out = reinterpret_cast<roctensornet_network_operator>(op);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_destroy_network_operator(roctensornet_network_operator op_in)
{
    auto* op = cast<network_operator_st>(op_in);
    if(op == nullptr) return ROCTENSORNET_STATUS_SUCCESS;
    for(auto& c : op->components)
        if(c.coefficient_mem != nullptr) std::free(c.coefficient_mem);
    delete op;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_network_operator_append_product(
    roctensornet_handle           h,
    roctensornet_network_operator op_in,
    const void*                   coefficient,
    int32_t                       num_tensors,
    const int32_t*                num_state_modes,
    const int32_t* const*         state_modes,
    const roctensornet_index_t* const* tensor_mode_strides,
    const void* const*            tensor_data,
    int64_t*                      component_id)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* op = cast<network_operator_st>(op_in);
    if(op == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    if(num_tensors < 0) return ROCTENSORNET_STATUS_INVALID_VALUE;
    if(num_tensors > 0)
    {
        ROCTENSORNET_CHECK_PTR(num_state_modes);
        ROCTENSORNET_CHECK_PTR(state_modes);
        ROCTENSORNET_CHECK_PTR(tensor_data);
    }
    (void)tensor_mode_strides;

    network_operator_component c;
    c.kind = 0;
    if(coefficient != nullptr)
    {
        size_t sz = element_size_bytes(op->data_type);
        c.coefficient_mem = std::malloc(sz);
        if(c.coefficient_mem == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
        std::memcpy(c.coefficient_mem, coefficient, sz);
    }
    c.per_tensor_modes.reserve(num_tensors);
    c.per_tensor_data.reserve(num_tensors);
    for(int32_t i = 0; i < num_tensors; ++i)
    {
        std::vector<int32_t> modes(state_modes[i], state_modes[i] + num_state_modes[i]);
        c.per_tensor_modes.push_back(std::move(modes));
        c.per_tensor_data.push_back(tensor_data[i]);
    }
    if(component_id) *component_id = static_cast<int64_t>(op->components.size());
    op->components.push_back(std::move(c));
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_network_operator_append_mpo(
    roctensornet_handle           h,
    roctensornet_network_operator op_in,
    const void*                   coefficient,
    int32_t                       num_state_modes,
    const int32_t*                state_modes,
    const int32_t*                tensor_mode_extents,
    const int64_t*                tensor_mode_strides,
    const void* const*            tensor_data,
    int32_t                       boundary_condition,
    int64_t*                      component_id)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* op = cast<network_operator_st>(op_in);
    if(op == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    if(num_state_modes < 0) return ROCTENSORNET_STATUS_INVALID_VALUE;
    if(num_state_modes > 0)
    {
        ROCTENSORNET_CHECK_PTR(state_modes);
        ROCTENSORNET_CHECK_PTR(tensor_data);
    }
    (void)tensor_mode_extents;
    (void)tensor_mode_strides;

    network_operator_component c;
    c.kind = 1;
    c.boundary_condition = boundary_condition;
    if(coefficient != nullptr)
    {
        size_t sz = element_size_bytes(op->data_type);
        c.coefficient_mem = std::malloc(sz);
        if(c.coefficient_mem == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
        std::memcpy(c.coefficient_mem, coefficient, sz);
    }
    for(int32_t i = 0; i < num_state_modes; ++i)
    {
        c.per_tensor_modes.push_back({state_modes[i]});
        c.per_tensor_data.push_back(tensor_data[i]);
    }
    if(component_id) *component_id = static_cast<int64_t>(op->components.size());
    op->components.push_back(std::move(c));
    return ROCTENSORNET_STATUS_SUCCESS;
}

} // extern "C"
