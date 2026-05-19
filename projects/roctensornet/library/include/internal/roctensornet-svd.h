/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Tensor SVD / QR / Gate-split entry points.
 * ************************************************************************ */

#ifndef ROCTENSORNET_SVD_H
#define ROCTENSORNET_SVD_H

#include "../roctensornet-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- SVD config ---- */
roctensornet_status
roctensornet_create_tensor_svd_config (roctensornet_handle handle,
                                       roctensornet_tensor_svd_config* cfg);

roctensornet_status
roctensornet_destroy_tensor_svd_config (roctensornet_tensor_svd_config cfg);

roctensornet_status
roctensornet_tensor_svd_config_get_attribute (
    roctensornet_handle                          handle,
    roctensornet_tensor_svd_config               cfg,
    roctensornet_tensor_svd_config_attribute     attr,
    void*                                        value,
    size_t                                       size);

roctensornet_status
roctensornet_tensor_svd_config_set_attribute (
    roctensornet_handle                          handle,
    roctensornet_tensor_svd_config               cfg,
    roctensornet_tensor_svd_config_attribute     attr,
    const void*                                  value,
    size_t                                       size);

/* ---- SVD info ---- */
roctensornet_status
roctensornet_create_tensor_svd_info (roctensornet_handle handle,
                                     roctensornet_tensor_svd_info* info);

roctensornet_status
roctensornet_destroy_tensor_svd_info (roctensornet_tensor_svd_info info);

roctensornet_status
roctensornet_tensor_svd_info_get_attribute (
    roctensornet_handle                       handle,
    roctensornet_tensor_svd_info              info,
    roctensornet_tensor_svd_info_attribute    attr,
    void*                                     value,
    size_t                                    size);

/* ---- SVD execute ---- */
roctensornet_status
roctensornet_tensor_svd (roctensornet_handle              handle,
                         roctensornet_tensor_descriptor   desc_in,
                         const void*                      raw_data_in,
                         roctensornet_tensor_descriptor   desc_u,
                         void*                            u,
                         void*                            s,
                         roctensornet_tensor_descriptor   desc_v,
                         void*                            v,
                         roctensornet_tensor_svd_config   svd_config,
                         roctensornet_tensor_svd_info     svd_info,
                         roctensornet_workspace_descriptor workspace,
                         hipStream_t                      stream);

/* ---- QR execute ---- */
roctensornet_status
roctensornet_tensor_qr (roctensornet_handle              handle,
                        roctensornet_tensor_descriptor   desc_in,
                        const void*                      raw_data_in,
                        roctensornet_tensor_descriptor   desc_q,
                        void*                            q,
                        roctensornet_tensor_descriptor   desc_r,
                        void*                            r,
                        roctensornet_workspace_descriptor workspace,
                        hipStream_t                      stream);

/* ---- Gate split: SVD of the result of applying a gate ---- */
roctensornet_status
roctensornet_gate_split (roctensornet_handle              handle,
                         roctensornet_tensor_descriptor   desc_in_a,
                         const void*                      raw_data_in_a,
                         roctensornet_tensor_descriptor   desc_in_b,
                         const void*                      raw_data_in_b,
                         roctensornet_tensor_descriptor   desc_in_g,
                         const void*                      raw_data_in_g,
                         roctensornet_tensor_descriptor   desc_u,
                         void*                            u,
                         void*                            s,
                         roctensornet_tensor_descriptor   desc_v,
                         void*                            v,
                         roctensornet_tensor_svd_config   svd_config,
                         roctensornet_tensor_svd_info     svd_info,
                         roctensornet_compute_type        compute_type,
                         roctensornet_workspace_descriptor workspace,
                         hipStream_t                      stream);

#ifdef __cplusplus
}
#endif

#endif /* ROCTENSORNET_SVD_H */
