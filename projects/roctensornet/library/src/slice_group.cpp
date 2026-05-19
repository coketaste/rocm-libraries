/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "roctensornet_descriptors.hpp"

using namespace roctensornet;

extern "C" {

roctensornet_status
roctensornet_create_slice_group_from_id_range(roctensornet_handle      handle,
                                              int64_t                  start,
                                              int64_t                  stop,
                                              int64_t                  step,
                                              roctensornet_slice_group* out)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    ROCTENSORNET_CHECK_PTR(out);
    if(step == 0) return ROCTENSORNET_STATUS_INVALID_VALUE;
    auto* g = new (std::nothrow) slice_group_st();
    if(g == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    if(step > 0)
        for(int64_t v = start; v < stop; v += step) g->slice_ids.push_back(v);
    else
        for(int64_t v = start; v > stop; v += step) g->slice_ids.push_back(v);
    *out = reinterpret_cast<roctensornet_slice_group>(g);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_create_slice_group_from_ids(roctensornet_handle handle,
                                         const int64_t*      ids,
                                         int32_t             num,
                                         roctensornet_slice_group* out)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    ROCTENSORNET_CHECK_PTR(out);
    if(num < 0) return ROCTENSORNET_STATUS_INVALID_VALUE;
    if(num > 0) ROCTENSORNET_CHECK_PTR(ids);
    auto* g = new (std::nothrow) slice_group_st();
    if(g == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    g->slice_ids.assign(ids, ids + num);
    *out = reinterpret_cast<roctensornet_slice_group>(g);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_destroy_slice_group(roctensornet_slice_group g)
{
    delete cast<slice_group_st>(g);
    return ROCTENSORNET_STATUS_SUCCESS;
}

} // extern "C"
