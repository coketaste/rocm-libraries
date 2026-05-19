/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * gate_split: contract A * G * B (with the appropriate index pattern
 * for a two-site gate split) and SVD-truncate the result back into
 * U (replacing A) and V (replacing B). The intermediate big-tensor
 * step uses the binary contraction kernel from contraction_execute;
 * the SVD step uses tensor_svd_host.
 *
 * For v0.1.0 the gate_split implementation is a documented stub that
 * returns ROCTENSORNET_STATUS_NOT_SUPPORTED. The composing pipeline
 * (manual contraction + tensor_svd) is fully supported and is what
 * the user should call directly until this entry point lands.
 * ************************************************************************ */

#include "roctensornet_descriptors.hpp"

using namespace roctensornet;

extern "C" {

roctensornet_status
roctensornet_gate_split(roctensornet_handle               h,
                        roctensornet_tensor_descriptor    desc_in_a,
                        const void*                       raw_data_in_a,
                        roctensornet_tensor_descriptor    desc_in_b,
                        const void*                       raw_data_in_b,
                        roctensornet_tensor_descriptor    desc_in_g,
                        const void*                       raw_data_in_g,
                        roctensornet_tensor_descriptor    desc_u,
                        void*                             u,
                        void*                             s,
                        roctensornet_tensor_descriptor    desc_v,
                        void*                             v,
                        roctensornet_tensor_svd_config    svd_config,
                        roctensornet_tensor_svd_info      svd_info,
                        roctensornet_compute_type         compute_type,
                        roctensornet_workspace_descriptor workspace,
                        hipStream_t                       stream)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    (void)desc_in_a; (void)desc_in_b; (void)desc_in_g;
    (void)raw_data_in_a; (void)raw_data_in_b; (void)raw_data_in_g;
    (void)desc_u; (void)desc_v;
    (void)u; (void)s; (void)v;
    (void)svd_config; (void)svd_info;
    (void)compute_type; (void)workspace; (void)stream;
    /* The composing operations (contract A*G*B via the binary
     * contraction kernel, then tensor_svd the result with truncation)
     * are fully implemented; only the convenience entry point is
     * left for follow-up. */
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

} // extern "C"
