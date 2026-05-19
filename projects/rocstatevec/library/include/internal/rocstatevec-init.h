/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Reverse-engineered from cuStateVec public API documentation
 * (cuQuantum 24.11 / cuStateVec 1.7.x); clean-room implementation.
 * ************************************************************************ */

#ifndef ROCSTATEVEC_INIT_H
#define ROCSTATEVEC_INIT_H

#include "../rocstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Initialize a state vector to one of the named presets
 *  (zero, uniform, GHZ, W). Bijection: \p custatevecInitializeStateVector. */
rocstatevec_status rocstatevec_initialize_state_vector(
    rocstatevec_handle             handle,
    void*                          state_vector,
    rocstatevec_data_type          state_vector_data_type,
    uint32_t                       n_index_bits,
    rocstatevec_state_vector_type  state_vector_type);

#ifdef __cplusplus
}
#endif

#endif /* ROCSTATEVEC_INIT_H */
