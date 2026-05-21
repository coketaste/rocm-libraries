/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Operator-action backward differentiation (NOT_SUPPORTED in v0.1.0).
 * v0.2 will provide a full reverse-mode AD path so cuQuantum-Q-style
 * gradient-based optimization runs against rocDENSITYMAT directly.
 * ************************************************************************ */

#include "rocdensitymat_descriptors.hpp"

extern "C" {

rocdensitymat_status rocdensitymat_operator_prepare_action_backward_diff(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    rocdensitymat_state state_in,
    rocdensitymat_state state_out,
    rocdensitymat_compute_type /*compute_type*/,
    size_t /*workspace_size_limit*/,
    rocdensitymat_workspace_descriptor /*workspace*/)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(op);
    ROCDENSITYMAT_CHECK_PTR(state_in);
    ROCDENSITYMAT_CHECK_PTR(state_out);
    return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;
}

rocdensitymat_status rocdensitymat_operator_compute_action_backward_diff(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    double /*t*/,
    int32_t /*num_params*/,
    const double* /*params*/,
    rocdensitymat_state state_in,
    rocdensitymat_state state_out_grad,
    rocdensitymat_state state_in_grad,
    double* /*params_grad*/,
    rocdensitymat_workspace_descriptor /*workspace*/)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(op);
    ROCDENSITYMAT_CHECK_PTR(state_in);
    ROCDENSITYMAT_CHECK_PTR(state_out_grad);
    ROCDENSITYMAT_CHECK_PTR(state_in_grad);
    return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;
}

} // extern "C"
