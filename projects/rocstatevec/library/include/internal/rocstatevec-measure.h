/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Reverse-engineered from cuStateVec public API documentation
 * (cuQuantum 24.11 / cuStateVec 1.7.x); clean-room implementation.
 * ************************************************************************ */

#ifndef ROCSTATEVEC_MEASURE_H
#define ROCSTATEVEC_MEASURE_H

#include "../rocstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Sum |amp|^2 across one or more basis-state index axes.
 *  Bijection: \p custatevecAbs2SumArray. */
rocstatevec_status rocstatevec_abs2_sum_array(
    rocstatevec_handle      handle,
    const void*             state_vector,
    rocstatevec_data_type   state_vector_data_type,
    uint32_t                n_index_bits,
    double*                 abs2_sum,
    const int32_t*          bit_ordering,
    uint32_t                bit_ordering_len,
    const int32_t*          mask_bit_string,
    const int32_t*          mask_ordering,
    uint32_t                mask_len);

/*! \brief Sum |amp|^2 over Z-basis projections of the indicated qubits.
 *  Bijection: \p custatevecAbs2SumOnZBasis. */
rocstatevec_status rocstatevec_abs2_sum_on_z_basis(
    rocstatevec_handle      handle,
    const void*             state_vector,
    rocstatevec_data_type   state_vector_data_type,
    uint32_t                n_index_bits,
    double*                 abs2_sum0,
    double*                 abs2_sum1,
    const int32_t*          basis_bits,
    uint32_t                n_basis_bits);

/*! \brief Perform a Z-basis projective measurement on selected qubits and
 *         optionally collapse + renormalize the state. Bijection:
 *  \p custatevecMeasureOnZBasis. */
rocstatevec_status rocstatevec_measure_on_z_basis(
    rocstatevec_handle      handle,
    void*                   state_vector,
    rocstatevec_data_type   state_vector_data_type,
    uint32_t                n_index_bits,
    int32_t*                parity,
    const int32_t*          basis_bits,
    uint32_t                n_basis_bits,
    double                  randnum,
    rocstatevec_collapse_op collapse);

/*! \brief Sample independent bit-strings from the current statevector
 *         distribution, projecting on a chosen bit-ordering subset.
 *  Bijection: \p custatevecBatchMeasure. */
rocstatevec_status rocstatevec_batch_measure(
    rocstatevec_handle      handle,
    void*                   state_vector,
    rocstatevec_data_type   state_vector_data_type,
    uint32_t                n_index_bits,
    int32_t*                bit_string,
    const int32_t*          bit_ordering,
    uint32_t                bit_string_len,
    double                  randnum,
    rocstatevec_collapse_op collapse);

/*! \brief Variant of `batch_measure` accepting an offset/mask.
 *  Bijection: \p custatevecBatchMeasureWithOffset. */
rocstatevec_status rocstatevec_batch_measure_with_offset(
    rocstatevec_handle      handle,
    void*                   state_vector,
    rocstatevec_data_type   state_vector_data_type,
    uint32_t                n_index_bits,
    int32_t*                bit_string,
    const int32_t*          bit_ordering,
    uint32_t                bit_string_len,
    double                  randnum,
    rocstatevec_collapse_op collapse,
    double                  offset,
    double                  abs2_sum);

/*! \brief Collapse the statevector onto a fixed Z-basis outcome on a chosen
 *         set of qubits. Bijection: \p custatevecCollapseOnZBasis. */
rocstatevec_status rocstatevec_collapse_on_z_basis(
    rocstatevec_handle      handle,
    void*                   state_vector,
    rocstatevec_data_type   state_vector_data_type,
    uint32_t                n_index_bits,
    int32_t                 parity,
    const int32_t*          basis_bits,
    uint32_t                n_basis_bits,
    double                  norm);

/*! \brief Collapse the statevector onto a fixed bit-string outcome on a
 *         chosen set of qubits. Bijection: \p custatevecCollapseByBitString. */
rocstatevec_status rocstatevec_collapse_by_bit_string(
    rocstatevec_handle      handle,
    void*                   state_vector,
    rocstatevec_data_type   state_vector_data_type,
    uint32_t                n_index_bits,
    const int32_t*          bit_string,
    const int32_t*          bit_ordering,
    uint32_t                bit_string_len,
    double                  norm);

#ifdef __cplusplus
}
#endif

#endif /* ROCSTATEVEC_MEASURE_H */
