/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Workspace descriptor entry points.
 * ************************************************************************ */

#ifndef ROCTENSORNET_WORKSPACE_H
#define ROCTENSORNET_WORKSPACE_H

#include "../roctensornet-types.h"

#ifdef __cplusplus
extern "C" {
#endif

roctensornet_status
roctensornet_create_workspace_descriptor (roctensornet_handle handle,
                                          roctensornet_workspace_descriptor* desc);

roctensornet_status
roctensornet_destroy_workspace_descriptor (roctensornet_workspace_descriptor desc);

roctensornet_status
roctensornet_workspace_compute_contraction_sizes (
    roctensornet_handle                     handle,
    roctensornet_network_descriptor         network_desc,
    roctensornet_contraction_optimizer_info info,
    roctensornet_workspace_descriptor       workspace_desc);

roctensornet_status
roctensornet_workspace_compute_svd_sizes (
    roctensornet_handle               handle,
    roctensornet_tensor_descriptor    desc_in,
    roctensornet_tensor_descriptor    desc_u,
    roctensornet_tensor_descriptor    desc_v,
    roctensornet_tensor_svd_config    svd_config,
    roctensornet_workspace_descriptor workspace_desc);

roctensornet_status
roctensornet_workspace_compute_qr_sizes (
    roctensornet_handle               handle,
    roctensornet_tensor_descriptor    desc_in,
    roctensornet_tensor_descriptor    desc_q,
    roctensornet_tensor_descriptor    desc_r,
    roctensornet_workspace_descriptor workspace_desc);

roctensornet_status
roctensornet_workspace_get_memory_size (
    roctensornet_handle               handle,
    roctensornet_workspace_descriptor desc,
    roctensornet_worksize_pref        pref,
    roctensornet_memspace             mem_space,
    roctensornet_workspace_kind       kind,
    int64_t*                          memory_size);

roctensornet_status
roctensornet_workspace_set_memory (roctensornet_handle               handle,
                                   roctensornet_workspace_descriptor desc,
                                   roctensornet_memspace             mem_space,
                                   roctensornet_workspace_kind       kind,
                                   void*                             memory_buffer,
                                   int64_t                           memory_size);

roctensornet_status
roctensornet_workspace_get_memory (roctensornet_handle               handle,
                                   roctensornet_workspace_descriptor desc,
                                   roctensornet_memspace             mem_space,
                                   roctensornet_workspace_kind       kind,
                                   void**                            memory_buffer,
                                   int64_t*                          memory_size);

roctensornet_status
roctensornet_workspace_purge_cache (roctensornet_handle               handle,
                                    roctensornet_workspace_descriptor desc,
                                    roctensornet_memspace             mem_space);

#ifdef __cplusplus
}
#endif

#endif /* ROCTENSORNET_WORKSPACE_H */
