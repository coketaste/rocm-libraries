/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPSTATEVEC_INIT_H
#define HIPSTATEVEC_INIT_H

#include "../hipstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipstatevecStatus_t hipstatevecInitializeStateVector(
    hipstatevecHandle_t          handle,
    void*                        state_vector,
    hipstatevecDataType_t        state_vector_data_type,
    uint32_t                     n_index_bits,
    hipstatevecStateVectorType_t state_vector_type);

#ifdef __cplusplus
}
#endif

#endif /* HIPSTATEVEC_INIT_H */
