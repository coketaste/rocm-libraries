/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Reverse-engineered from cuStateVec public API documentation
 * (cuQuantum 24.11 / cuStateVec 1.7.x); clean-room implementation.
 * ************************************************************************ */

#ifndef ROCSTATEVEC_PERMUTE_H
#define ROCSTATEVEC_PERMUTE_H

#include "../rocstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Swap pairs of qubit index bits in place. Optionally restrict the
 *         swap to amplitudes whose mask-bit pattern matches the supplied mask.
 *
 *  Bijection: \p custatevecSwapIndexBits. */
rocstatevec_status rocstatevec_swap_index_bits(
    rocstatevec_handle      handle,
    void*                   state_vector,
    rocstatevec_data_type   state_vector_data_type,
    uint32_t                n_index_bits,
    const rocstatevec_index_pair_t* bit_swaps,
    uint32_t                n_bit_swaps,
    const int32_t*          mask_bit_string,
    const int32_t*          mask_ordering,
    uint32_t                mask_len);

#ifdef __cplusplus
}
#endif

#endif /* ROCSTATEVEC_PERMUTE_H */
