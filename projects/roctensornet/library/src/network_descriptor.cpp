/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Network descriptor lifecycle, attribute round-trip, and output
 * tensor descriptor inference.
 *
 * Output mode list: the cuTensorNet contract is that all modes
 * appearing exactly once across the inputs are output modes; modes
 * appearing more than once are contracted. The user can also pass an
 * explicit output mode list which we validate against the inputs.
 * ************************************************************************ */

#include "roctensornet_descriptors.hpp"

#include <algorithm>
#include <unordered_map>

using namespace roctensornet;

extern "C" {

roctensornet_status
roctensornet_create_network_descriptor(roctensornet_handle             h,
                                       int32_t                         num_inputs,
                                       const int32_t*                  num_modes_in,
                                       const roctensornet_index_t* const* extents_in,
                                       const roctensornet_index_t* const* strides_in,
                                       const int32_t* const*           modes_in,
                                       const uint32_t*                 align_in,
                                       int32_t                         num_modes_out,
                                       const roctensornet_index_t*     extents_out,
                                       const roctensornet_index_t*     strides_out,
                                       const int32_t*                  modes_out,
                                       uint32_t                        align_out,
                                       roctensornet_data_type          data_type,
                                       roctensornet_compute_type       compute_type,
                                       roctensornet_network_descriptor* out)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    ROCTENSORNET_CHECK_PTR(out);
    if(num_inputs < 1) return ROCTENSORNET_STATUS_INVALID_VALUE;
    ROCTENSORNET_CHECK_PTR(num_modes_in);
    ROCTENSORNET_CHECK_PTR(extents_in);
    ROCTENSORNET_CHECK_PTR(modes_in);
    if(element_size_bytes(data_type) == 0)
        return ROCTENSORNET_STATUS_NOT_SUPPORTED;

    auto* nd = new (std::nothrow) network_descriptor_st();
    if(nd == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    nd->num_inputs   = num_inputs;
    nd->data_type    = data_type;
    nd->compute_type = compute_type;

    /* ---- Capture inputs ---- */
    nd->inputs.resize(num_inputs);
    std::unordered_map<int32_t, roctensornet_index_t> mode_extent;
    for(int32_t i = 0; i < num_inputs; ++i)
    {
        int32_t nm = num_modes_in[i];
        if(nm < 0 || nm > max_supported_modes)
        {
            delete nd;
            return ROCTENSORNET_STATUS_INVALID_VALUE;
        }
        auto& td = nd->inputs[i];
        td.modes.assign(modes_in[i], modes_in[i] + nm);
        td.extents.assign(extents_in[i], extents_in[i] + nm);
        if(strides_in != nullptr && strides_in[i] != nullptr)
            td.strides.assign(strides_in[i], strides_in[i] + nm);
        else
            generalized_column_major_strides(td.extents, td.strides);
        td.data_type = data_type;
        if(align_in != nullptr) td.alignment = align_in[i];
        for(int32_t k = 0; k < nm; ++k)
        {
            auto m = td.modes[k];
            auto e = td.extents[k];
            auto it = mode_extent.find(m);
            if(it == mode_extent.end())
                mode_extent.emplace(m, e);
            else if(it->second != e)
            {
                delete nd;
                return ROCTENSORNET_STATUS_INVALID_VALUE;
            }
        }
    }

    /* ---- Capture / infer output ---- */
    if(modes_out != nullptr && num_modes_out >= 0)
    {
        auto& td = nd->output;
        td.modes.assign(modes_out, modes_out + num_modes_out);
        if(extents_out != nullptr)
            td.extents.assign(extents_out, extents_out + num_modes_out);
        else
        {
            td.extents.resize(num_modes_out);
            for(int32_t k = 0; k < num_modes_out; ++k)
            {
                auto it = mode_extent.find(td.modes[k]);
                if(it == mode_extent.end())
                {
                    delete nd;
                    return ROCTENSORNET_STATUS_INVALID_VALUE;
                }
                td.extents[k] = it->second;
            }
        }
        if(strides_out != nullptr)
            td.strides.assign(strides_out, strides_out + num_modes_out);
        else
            generalized_column_major_strides(td.extents, td.strides);
        td.data_type = data_type;
        td.alignment = align_out;
    }
    else
    {
        /* Infer: count occurrences across all inputs. */
        std::unordered_map<int32_t, int> cnt;
        for(const auto& td : nd->inputs)
            for(auto m : td.modes) cnt[m]++;
        auto& td = nd->output;
        for(const auto& kv : cnt)
            if(kv.second == 1) td.modes.push_back(kv.first);
        std::sort(td.modes.begin(), td.modes.end());
        td.extents.resize(td.modes.size());
        for(size_t k = 0; k < td.modes.size(); ++k)
            td.extents[k] = mode_extent[td.modes[k]];
        generalized_column_major_strides(td.extents, td.strides);
        td.data_type = data_type;
        td.alignment = align_out;
    }

    *out = reinterpret_cast<roctensornet_network_descriptor>(nd);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_destroy_network_descriptor(roctensornet_network_descriptor d)
{
    delete cast<network_descriptor_st>(d);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_get_output_tensor_descriptor(roctensornet_handle             h,
                                          roctensornet_network_descriptor d,
                                          roctensornet_tensor_descriptor* out)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    ROCTENSORNET_CHECK_PTR(out);
    auto* nd = cast<network_descriptor_st>(d);
    if(nd == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    auto* td = new (std::nothrow) tensor_descriptor_st(nd->output);
    if(td == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    *out = reinterpret_cast<roctensornet_tensor_descriptor>(td);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_network_get_attribute(roctensornet_handle             h,
                                   roctensornet_network_descriptor d,
                                   roctensornet_network_attribute  attr,
                                   void*                           value,
                                   size_t                          size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* nd = cast<network_descriptor_st>(d);
    if(nd == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    switch(attr)
    {
    case ROCTENSORNET_NETWORK_INPUT_TENSORS_DATA_TYPES:
    {
        if(size < sizeof(roctensornet_data_type) * (size_t)nd->num_inputs)
            return ROCTENSORNET_STATUS_INVALID_VALUE;
        auto* p = static_cast<roctensornet_data_type*>(value);
        for(int32_t i = 0; i < nd->num_inputs; ++i) p[i] = nd->inputs[i].data_type;
        return ROCTENSORNET_STATUS_SUCCESS;
    }
    case ROCTENSORNET_NETWORK_OUTPUT_TENSOR_DATA_TYPE:
        if(size < sizeof(roctensornet_data_type)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<roctensornet_data_type*>(value) = nd->output.data_type;
        return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_NETWORK_COMPUTE_TYPE:
        if(size < sizeof(roctensornet_compute_type)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<roctensornet_compute_type*>(value) = nd->compute_type;
        return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_NETWORK_INPUT_TENSORS_NUM_MODES:
    {
        if(size < sizeof(int32_t) * (size_t)nd->num_inputs)
            return ROCTENSORNET_STATUS_INVALID_VALUE;
        auto* p = static_cast<int32_t*>(value);
        for(int32_t i = 0; i < nd->num_inputs; ++i) p[i] = nd->inputs[i].num_modes();
        return ROCTENSORNET_STATUS_SUCCESS;
    }
    case ROCTENSORNET_NETWORK_OUTPUT_TENSOR_NUM_MODES:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int32_t*>(value) = nd->output.num_modes();
        return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_NETWORK_INPUT_TENSORS_NUM:
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<int32_t*>(value) = nd->num_inputs;
        return ROCTENSORNET_STATUS_SUCCESS;
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_network_set_attribute(roctensornet_handle             h,
                                   roctensornet_network_descriptor d,
                                   roctensornet_network_attribute  attr,
                                   const void*                     value,
                                   size_t                          size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* nd = cast<network_descriptor_st>(d);
    if(nd == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    switch(attr)
    {
    case ROCTENSORNET_NETWORK_COMPUTE_TYPE:
        if(size < sizeof(roctensornet_compute_type)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        nd->compute_type = *static_cast<const roctensornet_compute_type*>(value);
        return ROCTENSORNET_STATUS_SUCCESS;
    default:
        break;
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

} // extern "C"
