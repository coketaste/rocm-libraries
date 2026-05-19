/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Contraction optimizer config / info / optimize entry points.
 * ************************************************************************ */

#ifndef ROCTENSORNET_OPTIMIZER_H
#define ROCTENSORNET_OPTIMIZER_H

#include "../roctensornet-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Optimizer config ---- */
roctensornet_status
roctensornet_create_contraction_optimizer_config (roctensornet_handle                          handle,
                                                  roctensornet_contraction_optimizer_config*   cfg);

roctensornet_status
roctensornet_destroy_contraction_optimizer_config (roctensornet_contraction_optimizer_config cfg);

roctensornet_status
roctensornet_contraction_optimizer_config_get_attribute (
    roctensornet_handle                                  handle,
    roctensornet_contraction_optimizer_config            cfg,
    roctensornet_contraction_optimizer_config_attribute  attr,
    void*                                                value,
    size_t                                               size);

roctensornet_status
roctensornet_contraction_optimizer_config_set_attribute (
    roctensornet_handle                                  handle,
    roctensornet_contraction_optimizer_config            cfg,
    roctensornet_contraction_optimizer_config_attribute  attr,
    const void*                                          value,
    size_t                                               size);

/* ---- Optimizer info ---- */
roctensornet_status
roctensornet_create_contraction_optimizer_info (roctensornet_handle             handle,
                                                roctensornet_network_descriptor desc,
                                                roctensornet_contraction_optimizer_info* info);

roctensornet_status
roctensornet_create_contraction_optimizer_info_from_packed_data (
    roctensornet_handle             handle,
    roctensornet_network_descriptor desc,
    const void*                     buffer,
    size_t                          size,
    roctensornet_contraction_optimizer_info* info);

roctensornet_status
roctensornet_destroy_contraction_optimizer_info (roctensornet_contraction_optimizer_info info);

roctensornet_status
roctensornet_contraction_optimizer_info_get_attribute (
    roctensornet_handle                                  handle,
    roctensornet_contraction_optimizer_info              info,
    roctensornet_contraction_optimizer_info_attribute    attr,
    void*                                                value,
    size_t                                               size);

roctensornet_status
roctensornet_contraction_optimizer_info_set_attribute (
    roctensornet_handle                                  handle,
    roctensornet_contraction_optimizer_info              info,
    roctensornet_contraction_optimizer_info_attribute    attr,
    const void*                                          value,
    size_t                                               size);

roctensornet_status
roctensornet_contraction_optimizer_info_get_packed_size (
    roctensornet_handle                                  handle,
    roctensornet_contraction_optimizer_info              info,
    size_t*                                              size);

roctensornet_status
roctensornet_contraction_optimizer_info_pack_data (
    roctensornet_handle                                  handle,
    roctensornet_contraction_optimizer_info              info,
    void*                                                buffer,
    size_t                                               size);

/* ---- Optimize ---- */
roctensornet_status
roctensornet_contraction_optimize (roctensornet_handle                       handle,
                                   roctensornet_network_descriptor           desc,
                                   roctensornet_contraction_optimizer_config cfg,
                                   uint64_t                                  workspace_size_constraint,
                                   roctensornet_contraction_optimizer_info   info);

#ifdef __cplusplus
}
#endif

#endif /* ROCTENSORNET_OPTIMIZER_H */
