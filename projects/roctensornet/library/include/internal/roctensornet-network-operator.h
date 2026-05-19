/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Network operator entry points.
 * ************************************************************************ */

#ifndef ROCTENSORNET_NETWORK_OPERATOR_H
#define ROCTENSORNET_NETWORK_OPERATOR_H

#include "../roctensornet-types.h"

#ifdef __cplusplus
extern "C" {
#endif

roctensornet_status
roctensornet_create_network_operator (roctensornet_handle handle,
                                      int32_t                       num_state_modes,
                                      const roctensornet_index_t*   state_mode_extents,
                                      roctensornet_data_type        data_type,
                                      roctensornet_network_operator* op);

roctensornet_status
roctensornet_destroy_network_operator (roctensornet_network_operator op);

roctensornet_status
roctensornet_network_operator_append_product (
    roctensornet_handle           handle,
    roctensornet_network_operator op,
    /* coefficient (host scalar, sized by op data_type) */ const void* coefficient,
    int32_t                       num_tensors,
    const int32_t*                num_state_modes,
    const int32_t* const*         state_modes,
    const roctensornet_index_t* const* tensor_mode_strides,
    const void* const*            tensor_data,
    int64_t*                      component_id);

roctensornet_status
roctensornet_network_operator_append_mpo (
    roctensornet_handle           handle,
    roctensornet_network_operator op,
    const void*                   coefficient,
    int32_t                       num_state_modes,
    const int32_t*                state_modes,
    const int32_t*                tensor_mode_extents,
    const int64_t*                tensor_mode_strides,
    const void* const*            tensor_data,
    int32_t                       boundary_condition,
    int64_t*                      component_id);

#ifdef __cplusplus
}
#endif

#endif /* ROCTENSORNET_NETWORK_OPERATOR_H */
