/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "rocdensitymat_descriptors.hpp"

extern "C" {

rocdensitymat_status rocdensitymat_create_operator_term(
    rocdensitymat_handle handle,
    int32_t num_space_modes,
    const int64_t* space_shape,
    rocdensitymat_operator_term* term)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(term);
    if(num_space_modes <= 0 || space_shape == nullptr)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;

    auto* t = new(std::nothrow) _rocdensitymat_operator_term();
    if(t == nullptr) return ROCDENSITYMAT_STATUS_ALLOC_FAILED;
    t->num_space_modes = num_space_modes;
    t->space_shape.assign(space_shape, space_shape + num_space_modes);
    *term = t;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_destroy_operator_term(
    rocdensitymat_operator_term term)
{
    if(term == nullptr) return ROCDENSITYMAT_STATUS_SUCCESS;
    delete term;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_operator_term_append_elementary_product(
    rocdensitymat_handle handle,
    rocdensitymat_operator_term term,
    int32_t num_factors,
    const rocdensitymat_elementary_operator* operators,
    const int32_t* state_modes,
    const int32_t* mode_action_duality,
    rocdensitymat_complex_double coefficient,
    rocdensitymat_scalar_callback_t tdep_callback)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(term);
    if(num_factors < 0) return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    if(num_factors > rocdensitymat::max_factors_per_term)
        return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;
    if(num_factors > 0 && (operators == nullptr || state_modes == nullptr))
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;

    _rocdensitymat_operator_term::product p;
    p.coefficient   = coefficient;
    p.tdep_callback = tdep_callback;
    p.operators.assign(operators, operators + num_factors);
    p.state_modes.assign(state_modes, state_modes + num_factors);
    if(mode_action_duality != nullptr)
    {
        p.mode_action_duality.assign(mode_action_duality, mode_action_duality + num_factors);
    }
    else
    {
        p.mode_action_duality.assign(num_factors, ROCDENSITYMAT_DUALITY_KET);
    }

    // Validate that referenced modes are in range and that an elementary
    // operator's local Hilbert space matches the targeted state modes.
    for(int32_t k = 0; k < num_factors; ++k)
    {
        auto e = p.operators[k];
        if(e == nullptr) return ROCDENSITYMAT_STATUS_INVALID_VALUE;
        if(p.state_modes[k] < 0 || p.state_modes[k] >= term->num_space_modes)
            return ROCDENSITYMAT_STATUS_INVALID_VALUE;
        if(e->num_modes != 1)
            return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;
        if(term->space_shape[p.state_modes[k]] != e->mode_extents[0])
            return ROCDENSITYMAT_STATUS_INVALID_VALUE;
        if(p.mode_action_duality[k] != ROCDENSITYMAT_DUALITY_KET
           && p.mode_action_duality[k] != ROCDENSITYMAT_DUALITY_BRA)
            return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    }

    term->products.push_back(std::move(p));
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

} // extern "C"
