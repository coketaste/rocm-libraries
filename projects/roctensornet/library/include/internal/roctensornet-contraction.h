/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Contraction plan, autotune, execute, and slice-group entry points.
 * ************************************************************************ */

#ifndef ROCTENSORNET_CONTRACTION_H
#define ROCTENSORNET_CONTRACTION_H

#include "../roctensornet-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Plan ---- */
roctensornet_status
roctensornet_create_contraction_plan (roctensornet_handle                     handle,
                                      roctensornet_network_descriptor         desc,
                                      roctensornet_contraction_optimizer_info info,
                                      roctensornet_workspace_descriptor       workspace,
                                      roctensornet_contraction_plan*          plan);

roctensornet_status
roctensornet_destroy_contraction_plan (roctensornet_contraction_plan plan);

/* ---- Autotune preference ---- */
roctensornet_status
roctensornet_create_contraction_autotune_preference (
    roctensornet_handle handle,
    roctensornet_contraction_autotune_preference* prefs);

roctensornet_status
roctensornet_destroy_contraction_autotune_preference (
    roctensornet_contraction_autotune_preference prefs);

roctensornet_status
roctensornet_contraction_autotune_preference_get_attribute (
    roctensornet_handle                                     handle,
    roctensornet_contraction_autotune_preference            prefs,
    roctensornet_contraction_autotune_preference_attribute  attr,
    void*                                                   value,
    size_t                                                  size);

roctensornet_status
roctensornet_contraction_autotune_preference_set_attribute (
    roctensornet_handle                                     handle,
    roctensornet_contraction_autotune_preference            prefs,
    roctensornet_contraction_autotune_preference_attribute  attr,
    const void*                                             value,
    size_t                                                  size);

/* ---- Autotune ---- */
roctensornet_status
roctensornet_contraction_autotune (roctensornet_handle                   handle,
                                   roctensornet_contraction_plan         plan,
                                   const void* const*                    raw_data_in,
                                   void*                                 raw_data_out,
                                   roctensornet_workspace_descriptor     workspace,
                                   roctensornet_contraction_autotune_preference prefs,
                                   hipStream_t                           stream);

/* ---- Execute ---- */
roctensornet_status
roctensornet_contraction (roctensornet_handle               handle,
                          roctensornet_contraction_plan     plan,
                          const void* const*                raw_data_in,
                          void*                             raw_data_out,
                          roctensornet_workspace_descriptor workspace,
                          int64_t                           slice_id,
                          hipStream_t                       stream);

roctensornet_status
roctensornet_contract_slices (roctensornet_handle               handle,
                              roctensornet_contraction_plan     plan,
                              const void* const*                raw_data_in,
                              void*                             raw_data_out,
                              int32_t                           accumulate_output,
                              roctensornet_workspace_descriptor workspace,
                              roctensornet_slice_group          slice_group,
                              hipStream_t                       stream);

/* ---- Slice group ---- */
roctensornet_status
roctensornet_create_slice_group_from_id_range (roctensornet_handle handle,
                                               int64_t slice_id_start,
                                               int64_t slice_id_stop,
                                               int64_t slice_id_step,
                                               roctensornet_slice_group* slice_group);

roctensornet_status
roctensornet_create_slice_group_from_ids (roctensornet_handle handle,
                                          const int64_t*      slice_ids,
                                          int32_t             num_slices,
                                          roctensornet_slice_group* slice_group);

roctensornet_status
roctensornet_destroy_slice_group (roctensornet_slice_group slice_group);

#ifdef __cplusplus
}
#endif

#endif /* ROCTENSORNET_CONTRACTION_H */
