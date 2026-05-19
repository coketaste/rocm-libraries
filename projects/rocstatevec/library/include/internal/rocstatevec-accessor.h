/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Reverse-engineered from cuStateVec public API documentation
 * (cuQuantum 24.11 / cuStateVec 1.7.x); clean-room implementation.
 * ************************************************************************ */

#ifndef ROCSTATEVEC_ACCESSOR_H
#define ROCSTATEVEC_ACCESSOR_H

#include "../rocstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Create a read-write accessor descriptor over a sub-statevector slice.
 *  Bijection: \p custatevecAccessorCreate. */
rocstatevec_status rocstatevec_accessor_create(
    rocstatevec_handle               handle,
    void*                            state_vector,
    rocstatevec_data_type            state_vector_data_type,
    uint32_t                         n_index_bits,
    rocstatevec_accessor_descriptor* accessor,
    const int32_t*                   bit_ordering,
    uint32_t                         bit_ordering_len,
    const int32_t*                   mask_bit_string,
    const int32_t*                   mask_ordering,
    uint32_t                         mask_len,
    size_t*                          extra_workspace_size_in_bytes);

/*! \brief Create a read-only accessor descriptor over a sub-statevector slice.
 *  Bijection: \p custatevecAccessorCreateView. */
rocstatevec_status rocstatevec_accessor_create_view(
    rocstatevec_handle               handle,
    const void*                      state_vector,
    rocstatevec_data_type            state_vector_data_type,
    uint32_t                         n_index_bits,
    rocstatevec_accessor_descriptor* accessor,
    const int32_t*                   bit_ordering,
    uint32_t                         bit_ordering_len,
    const int32_t*                   mask_bit_string,
    const int32_t*                   mask_ordering,
    uint32_t                         mask_len,
    size_t*                          extra_workspace_size_in_bytes);

/*! \brief Destroy an accessor descriptor.
 *  Bijection: \p custatevecAccessorDestroy. */
rocstatevec_status rocstatevec_accessor_destroy(
    rocstatevec_accessor_descriptor accessor);

/*! \brief Hand a workspace buffer to an accessor descriptor.
 *  Bijection: \p custatevecAccessorSetExtraWorkspace. */
rocstatevec_status rocstatevec_accessor_set_extra_workspace(
    rocstatevec_handle              handle,
    rocstatevec_accessor_descriptor accessor,
    void*                           extra_workspace,
    size_t                          extra_workspace_size_in_bytes);

/*! \brief Read amplitudes from the accessor's window into a host buffer.
 *  Bijection: \p custatevecAccessorGet. */
rocstatevec_status rocstatevec_accessor_get(
    rocstatevec_handle              handle,
    rocstatevec_accessor_descriptor accessor,
    void*                           external_buffer,
    rocstatevec_index_t             begin,
    rocstatevec_index_t             end);

/*! \brief Write amplitudes from a host buffer into the accessor's window.
 *  Bijection: \p custatevecAccessorSet. */
rocstatevec_status rocstatevec_accessor_set(
    rocstatevec_handle              handle,
    rocstatevec_accessor_descriptor accessor,
    const void*                     external_buffer,
    rocstatevec_index_t             begin,
    rocstatevec_index_t             end);

#ifdef __cplusplus
}
#endif

#endif /* ROCSTATEVEC_ACCESSOR_H */
