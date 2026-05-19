/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Network descriptor + tensor descriptor entry points.
 * ************************************************************************ */

#ifndef ROCTENSORNET_NETWORK_H
#define ROCTENSORNET_NETWORK_H

#include "../roctensornet-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Network descriptor ---- */
roctensornet_status
roctensornet_create_network_descriptor (roctensornet_handle           handle,
                                        int32_t                       num_inputs,
                                        const int32_t*                num_modes_in,
                                        const roctensornet_index_t* const* extents_in,
                                        const roctensornet_index_t* const* strides_in,
                                        const int32_t* const*         modes_in,
                                        const uint32_t*               alignment_requirements_in,
                                        int32_t                       num_modes_out,
                                        const roctensornet_index_t*   extents_out,
                                        const roctensornet_index_t*   strides_out,
                                        const int32_t*                modes_out,
                                        uint32_t                      alignment_requirements_out,
                                        roctensornet_data_type        data_type,
                                        roctensornet_compute_type     compute_type,
                                        roctensornet_network_descriptor* desc);

roctensornet_status
roctensornet_destroy_network_descriptor (roctensornet_network_descriptor desc);

roctensornet_status
roctensornet_get_output_tensor_descriptor (roctensornet_handle             handle,
                                           roctensornet_network_descriptor desc,
                                           roctensornet_tensor_descriptor* out_desc);

roctensornet_status
roctensornet_network_get_attribute (roctensornet_handle             handle,
                                    roctensornet_network_descriptor desc,
                                    roctensornet_network_attribute  attr,
                                    void*                           value,
                                    size_t                          size);

roctensornet_status
roctensornet_network_set_attribute (roctensornet_handle             handle,
                                    roctensornet_network_descriptor desc,
                                    roctensornet_network_attribute  attr,
                                    const void*                     value,
                                    size_t                          size);

/* ---- Tensor descriptor ---- */
roctensornet_status
roctensornet_create_tensor_descriptor (roctensornet_handle           handle,
                                       int32_t                       num_modes,
                                       const roctensornet_index_t*   extents,
                                       const roctensornet_index_t*   strides,
                                       const int32_t*                modes,
                                       roctensornet_data_type        data_type,
                                       roctensornet_tensor_descriptor* desc);

roctensornet_status
roctensornet_destroy_tensor_descriptor (roctensornet_tensor_descriptor desc);

roctensornet_status
roctensornet_get_tensor_details (roctensornet_handle             handle,
                                 roctensornet_tensor_descriptor  desc,
                                 int32_t*                        num_modes,
                                 size_t*                         data_size_bytes,
                                 roctensornet_data_type*         data_type,
                                 int32_t*                        modes,
                                 roctensornet_index_t*           extents,
                                 roctensornet_index_t*           strides);

#ifdef __cplusplus
}
#endif

#endif /* ROCTENSORNET_NETWORK_H */
