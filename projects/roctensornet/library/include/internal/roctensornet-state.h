/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Network state entry points.
 * ************************************************************************ */

#ifndef ROCTENSORNET_STATE_H
#define ROCTENSORNET_STATE_H

#include "../roctensornet-types.h"

#ifdef __cplusplus
extern "C" {
#endif

roctensornet_status
roctensornet_create_state (roctensornet_handle           handle,
                           roctensornet_state_purity     purity,
                           int32_t                       num_state_modes,
                           const roctensornet_index_t*   state_mode_extents,
                           roctensornet_data_type        data_type,
                           roctensornet_state*           state);

roctensornet_status
roctensornet_destroy_state (roctensornet_state state);

roctensornet_status
roctensornet_state_apply_tensor_operator (
    roctensornet_handle handle,
    roctensornet_state  state,
    int32_t             num_state_modes,
    const int32_t*      state_modes,
    void*               tensor_data,
    const int64_t*      tensor_mode_strides,
    int32_t             immutable,
    int32_t             adjoint,
    int32_t             unitary,
    int64_t*            tensor_id);

roctensornet_status
roctensornet_state_apply_controlled_tensor_operator (
    roctensornet_handle handle,
    roctensornet_state  state,
    int32_t             num_control_modes,
    const int32_t*      state_control_modes,
    const int64_t*      state_control_values,
    int32_t             num_target_modes,
    const int32_t*      state_target_modes,
    void*               tensor_data,
    const int64_t*      tensor_mode_strides,
    int32_t             immutable,
    int32_t             adjoint,
    int32_t             unitary,
    int64_t*            tensor_id);

roctensornet_status
roctensornet_state_apply_unitary_channel (
    roctensornet_handle handle,
    roctensornet_state  state,
    int32_t             num_state_modes,
    const int32_t*      state_modes,
    int32_t             num_tensors,
    void* const*        tensor_data,
    const int64_t* const* tensor_mode_strides,
    const double*       probabilities,
    int64_t*            channel_id);

roctensornet_status
roctensornet_state_apply_general_channel (
    roctensornet_handle handle,
    roctensornet_state  state,
    int32_t             num_state_modes,
    const int32_t*      state_modes,
    int32_t             num_tensors,
    void* const*        tensor_data,
    const int64_t* const* tensor_mode_strides,
    int64_t*            channel_id);

roctensornet_status
roctensornet_state_update_tensor_operator (
    roctensornet_handle handle,
    roctensornet_state  state,
    int64_t             tensor_id,
    void*               tensor_data,
    int32_t             unitary);

roctensornet_status
roctensornet_state_configure (roctensornet_handle             handle,
                              roctensornet_state              state,
                              roctensornet_state_attribute    attr,
                              const void*                     value,
                              size_t                          size);

roctensornet_status
roctensornet_state_get_info (roctensornet_handle             handle,
                             roctensornet_state              state,
                             roctensornet_state_attribute    attr,
                             void*                           value,
                             size_t                          size);

roctensornet_status
roctensornet_state_prepare (roctensornet_handle               handle,
                            roctensornet_state                state,
                            size_t                            max_workspace_size_device,
                            roctensornet_workspace_descriptor workspace,
                            hipStream_t                       stream);

roctensornet_status
roctensornet_state_compute (roctensornet_handle               handle,
                            roctensornet_state                state,
                            roctensornet_workspace_descriptor workspace,
                            void* const*                      state_tensors_out,
                            hipStream_t                       stream);

#ifdef __cplusplus
}
#endif

#endif /* ROCTENSORNET_STATE_H */
