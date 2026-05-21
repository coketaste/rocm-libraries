/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "rocdensitymat_descriptors.hpp"

extern "C" {

rocdensitymat_status rocdensitymat_create_operator(
    rocdensitymat_handle handle,
    int32_t num_space_modes,
    const int64_t* space_shape,
    rocdensitymat_operator* op)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(op);
    if(num_space_modes <= 0 || space_shape == nullptr)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    auto* o = new(std::nothrow) _rocdensitymat_operator();
    if(o == nullptr) return ROCDENSITYMAT_STATUS_ALLOC_FAILED;
    o->num_space_modes = num_space_modes;
    o->space_shape.assign(space_shape, space_shape + num_space_modes);
    *op = o;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_destroy_operator(rocdensitymat_operator op)
{
    if(op == nullptr) return ROCDENSITYMAT_STATUS_SUCCESS;
    delete op;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_operator_append_term(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    rocdensitymat_operator_term term,
    int32_t duality_offset,
    rocdensitymat_complex_double coefficient,
    rocdensitymat_scalar_callback_t tdep_callback)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(op);
    ROCDENSITYMAT_CHECK_PTR(term);
    if(op->num_space_modes != term->num_space_modes)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    for(int32_t i = 0; i < op->num_space_modes; ++i)
    {
        if(op->space_shape[i] != term->space_shape[i])
            return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    }
    _rocdensitymat_operator::term_entry e;
    e.term            = term;
    e.duality_offset  = duality_offset;
    e.coefficient     = coefficient;
    e.tdep_callback   = tdep_callback;
    op->terms.push_back(e);
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

} // extern "C"
