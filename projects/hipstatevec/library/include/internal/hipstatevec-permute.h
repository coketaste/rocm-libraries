/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPSTATEVEC_PERMUTE_H
#define HIPSTATEVEC_PERMUTE_H

#include "../hipstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipstatevecStatus_t hipstatevecSwapIndexBits(
    hipstatevecHandle_t              handle,
    void*                            state_vector,
    hipstatevecDataType_t            state_vector_data_type,
    uint32_t                         n_index_bits,
    const hipstatevecIndexPair_t*    bit_swaps,
    uint32_t                         n_bit_swaps,
    const int32_t*                   mask_bit_string,
    const int32_t*                   mask_ordering,
    uint32_t                         mask_len);

#ifdef __cplusplus
}
#endif

#endif /* HIPSTATEVEC_PERMUTE_H */
