/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

/*! \file
 *  \brief Operator algebra: elementary operators, OperatorTerm, Operator.
 */

#ifndef ROCDENSITYMAT_OPERATOR_H
#define ROCDENSITYMAT_OPERATOR_H

#include "../rocdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Create an elementary operator acting on a contiguous mode block.
 *
 *  ``num_modes`` is the rank of the operator's local Hilbert space.
 *  ``mode_extents[k]`` is the dimension of mode \p k. ``kind`` selects an
 *  Identity / Pauli convenience operator, a dense matrix supplied through
 *  ``data`` (column-major, of size ``prod(extents)^2`` complex elements),
 *  or a diagonal operator (``data`` of size ``prod(extents)`` complex
 *  diagonal entries). ``tdep_callback`` is the optional time-dependent
 *  scalar coefficient; pass ``NULL`` for static operators.
 *
 *  Bijection: \p cudensitymatCreateElementaryOperator. */
rocdensitymat_status rocdensitymat_create_elementary_operator(
    rocdensitymat_handle handle,
    int32_t num_modes,
    const int64_t* mode_extents,
    rocdensitymat_elementary_kind kind,
    rocdensitymat_data_type data_type,
    const void* data,
    rocdensitymat_scalar_callback_t tdep_callback,
    rocdensitymat_elementary_operator* op);

/*! \brief Destroy an elementary operator descriptor. */
rocdensitymat_status rocdensitymat_destroy_elementary_operator(
    rocdensitymat_elementary_operator op);

/*! \brief Create an empty operator term over the supplied Hilbert space.
 *
 *  Bijection: \p cudensitymatCreateOperatorTerm. */
rocdensitymat_status rocdensitymat_create_operator_term(
    rocdensitymat_handle handle,
    int32_t num_space_modes,
    const int64_t* space_shape,
    rocdensitymat_operator_term* term);

/*! \brief Destroy an operator term. */
rocdensitymat_status rocdensitymat_destroy_operator_term(
    rocdensitymat_operator_term term);

/*! \brief Append a product of elementary operators to an operator term.
 *
 *  ``num_factors`` is the number of elementary operators in the product.
 *  ``operators`` is a length-``num_factors`` array of elementary-operator
 *  handles. ``state_modes`` is the corresponding length-``num_factors``
 *  array of mode indices into the operator-term Hilbert space.
 *  ``mode_action_duality`` selects \p KET (left) or \p BRA (right) action
 *  for each factor on a mixed state. ``coefficient`` is the static
 *  per-product coefficient; ``tdep_callback`` is an optional host
 *  time-dependent multiplier evaluated once per stepper call.
 *
 *  Bijection: \p cudensitymatOperatorTermAppendElementaryProduct. */
rocdensitymat_status rocdensitymat_operator_term_append_elementary_product(
    rocdensitymat_handle handle,
    rocdensitymat_operator_term term,
    int32_t num_factors,
    const rocdensitymat_elementary_operator* operators,
    const int32_t* state_modes,
    const int32_t* mode_action_duality,
    rocdensitymat_complex_double coefficient,
    rocdensitymat_scalar_callback_t tdep_callback);

/*! \brief Create an empty operator over the supplied Hilbert space.
 *
 *  Bijection: \p cudensitymatCreateOperator. */
rocdensitymat_status rocdensitymat_create_operator(
    rocdensitymat_handle handle,
    int32_t num_space_modes,
    const int64_t* space_shape,
    rocdensitymat_operator* op);

/*! \brief Destroy an operator. */
rocdensitymat_status rocdensitymat_destroy_operator(rocdensitymat_operator op);

/*! \brief Append an operator term to an operator with a per-term coefficient.
 *
 *  ``duality_offset`` is the action-side bias applied to all factors in
 *  the term; pass ``0`` for plain unitary action. A non-zero
 *  ``duality_offset`` flips the term into a Lindblad-collapse contribution
 *  whose compute path is \p NOT_SUPPORTED in v0.1.0.
 *
 *  Bijection: \p cudensitymatOperatorAppendTerm. */
rocdensitymat_status rocdensitymat_operator_append_term(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    rocdensitymat_operator_term term,
    int32_t duality_offset,
    rocdensitymat_complex_double coefficient,
    rocdensitymat_scalar_callback_t tdep_callback);

#ifdef __cplusplus
}
#endif

#endif /* ROCDENSITYMAT_OPERATOR_H */
