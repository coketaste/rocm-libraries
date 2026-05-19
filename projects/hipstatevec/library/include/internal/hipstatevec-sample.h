/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPSTATEVEC_SAMPLE_H
#define HIPSTATEVEC_SAMPLE_H

#include "../hipstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipstatevecStatus_t hipstatevecSamplerCreate(
    hipstatevecHandle_t              handle,
    const void*                      state_vector,
    hipstatevecDataType_t            state_vector_data_type,
    uint32_t                         n_index_bits,
    hipstatevecSamplerDescriptor_t*  sampler,
    uint32_t                         n_max_shots,
    size_t*                          extra_workspace_size_in_bytes);

hipstatevecStatus_t hipstatevecSamplerDestroy(hipstatevecSamplerDescriptor_t sampler);

hipstatevecStatus_t hipstatevecSamplerPreprocess(
    hipstatevecHandle_t              handle,
    hipstatevecSamplerDescriptor_t   sampler,
    void*                            extra_workspace,
    size_t                           extra_workspace_size_in_bytes);

hipstatevecStatus_t hipstatevecSamplerGetSquaredNorm(
    hipstatevecHandle_t              handle,
    hipstatevecSamplerDescriptor_t   sampler,
    double*                          norm);

hipstatevecStatus_t hipstatevecSamplerApplySubSVOffset(
    hipstatevecHandle_t              handle,
    hipstatevecSamplerDescriptor_t   sampler,
    int32_t                          sub_sv_index,
    uint32_t                         n_sub_svs,
    double                           offset,
    double                           norm);

hipstatevecStatus_t hipstatevecSamplerSample(
    hipstatevecHandle_t              handle,
    hipstatevecSamplerDescriptor_t   sampler,
    hipstatevecIndex_t*              bit_strings,
    const int32_t*                   bit_ordering,
    uint32_t                         bit_string_len,
    const double*                    randnums,
    uint32_t                         n_shots,
    hipstatevecSamplerOutput_t       output);

#ifdef __cplusplus
}
#endif

#endif /* HIPSTATEVEC_SAMPLE_H */
