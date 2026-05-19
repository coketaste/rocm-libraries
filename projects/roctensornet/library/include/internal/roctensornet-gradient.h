/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Contraction gradient entry point.
 * ************************************************************************ */

#ifndef ROCTENSORNET_GRADIENT_H
#define ROCTENSORNET_GRADIENT_H

#include "../roctensornet-types.h"

#ifdef __cplusplus
extern "C" {
#endif

roctensornet_status
roctensornet_compute_gradients_backward (
    roctensornet_handle               handle,
    roctensornet_contraction_plan     plan,
    const void* const*                raw_data_in,
    const void*                       output_gradient,
    void* const*                      gradients,
    int32_t                           accumulate_output,
    roctensornet_workspace_descriptor workspace,
    hipStream_t                       stream);

#ifdef __cplusplus
}
#endif

#endif /* ROCTENSORNET_GRADIENT_H */
