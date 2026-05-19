/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPSTATEVEC_ACCESSOR_H
#define HIPSTATEVEC_ACCESSOR_H

#include "../hipstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipstatevecStatus_t hipstatevecAccessorCreate(
    hipstatevecHandle_t              handle,
    void*                            state_vector,
    hipstatevecDataType_t            state_vector_data_type,
    uint32_t                         n_index_bits,
    hipstatevecAccessorDescriptor_t* accessor,
    const int32_t*                   bit_ordering,
    uint32_t                         bit_ordering_len,
    const int32_t*                   mask_bit_string,
    const int32_t*                   mask_ordering,
    uint32_t                         mask_len,
    size_t*                          extra_workspace_size_in_bytes);

hipstatevecStatus_t hipstatevecAccessorCreateView(
    hipstatevecHandle_t              handle,
    const void*                      state_vector,
    hipstatevecDataType_t            state_vector_data_type,
    uint32_t                         n_index_bits,
    hipstatevecAccessorDescriptor_t* accessor,
    const int32_t*                   bit_ordering,
    uint32_t                         bit_ordering_len,
    const int32_t*                   mask_bit_string,
    const int32_t*                   mask_ordering,
    uint32_t                         mask_len,
    size_t*                          extra_workspace_size_in_bytes);

hipstatevecStatus_t hipstatevecAccessorDestroy(hipstatevecAccessorDescriptor_t accessor);

hipstatevecStatus_t hipstatevecAccessorSetExtraWorkspace(
    hipstatevecHandle_t              handle,
    hipstatevecAccessorDescriptor_t  accessor,
    void*                            extra_workspace,
    size_t                           extra_workspace_size_in_bytes);

hipstatevecStatus_t hipstatevecAccessorGet(
    hipstatevecHandle_t              handle,
    hipstatevecAccessorDescriptor_t  accessor,
    void*                            external_buffer,
    hipstatevecIndex_t               begin,
    hipstatevecIndex_t               end);

hipstatevecStatus_t hipstatevecAccessorSet(
    hipstatevecHandle_t              handle,
    hipstatevecAccessorDescriptor_t  accessor,
    const void*                      external_buffer,
    hipstatevecIndex_t               begin,
    hipstatevecIndex_t               end);

#ifdef __cplusplus
}
#endif

#endif /* HIPSTATEVEC_ACCESSOR_H */
