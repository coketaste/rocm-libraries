/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "roctensornet_descriptors.hpp"

#include <algorithm>

using namespace roctensornet;

extern "C" {

roctensornet_status
roctensornet_create_tensor_descriptor(roctensornet_handle              h,
                                      int32_t                          num_modes,
                                      const roctensornet_index_t*      extents,
                                      const roctensornet_index_t*      strides,
                                      const int32_t*                   modes,
                                      roctensornet_data_type           data_type,
                                      roctensornet_tensor_descriptor*  out)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    ROCTENSORNET_CHECK_PTR(out);
    if(num_modes < 0 || num_modes > max_supported_modes)
        return ROCTENSORNET_STATUS_INVALID_VALUE;
    if(num_modes > 0)
    {
        ROCTENSORNET_CHECK_PTR(extents);
        ROCTENSORNET_CHECK_PTR(modes);
    }
    if(element_size_bytes(data_type) == 0)
        return ROCTENSORNET_STATUS_NOT_SUPPORTED;

    auto* d = new (std::nothrow) tensor_descriptor_st();
    if(d == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    d->modes.assign(modes, modes + num_modes);
    d->extents.assign(extents, extents + num_modes);
    if(strides != nullptr)
        d->strides.assign(strides, strides + num_modes);
    else
        generalized_column_major_strides(d->extents, d->strides);
    d->data_type = data_type;

    *out = reinterpret_cast<roctensornet_tensor_descriptor>(d);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_destroy_tensor_descriptor(roctensornet_tensor_descriptor d)
{
    delete cast<tensor_descriptor_st>(d);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_get_tensor_details(roctensornet_handle           h,
                                roctensornet_tensor_descriptor d,
                                int32_t*                       num_modes,
                                size_t*                        data_size_bytes,
                                roctensornet_data_type*        data_type,
                                int32_t*                       modes,
                                roctensornet_index_t*          extents,
                                roctensornet_index_t*          strides)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* dd = cast<tensor_descriptor_st>(d);
    if(dd == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    if(num_modes)       *num_modes       = dd->num_modes();
    if(data_size_bytes) *data_size_bytes = dd->num_bytes();
    if(data_type)       *data_type       = dd->data_type;
    if(modes && !dd->modes.empty())     std::copy(dd->modes.begin(),   dd->modes.end(),   modes);
    if(extents && !dd->extents.empty()) std::copy(dd->extents.begin(), dd->extents.end(), extents);
    if(strides && !dd->strides.empty()) std::copy(dd->strides.begin(), dd->strides.end(), strides);
    return ROCTENSORNET_STATUS_SUCCESS;
}

} // extern "C"
