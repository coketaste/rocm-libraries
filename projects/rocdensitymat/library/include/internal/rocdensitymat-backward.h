/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

/*! \file
 *  \brief Backward-differentiation entry points (NOT_SUPPORTED in v0.1.0).
 */

#ifndef ROCDENSITYMAT_BACKWARD_H
#define ROCDENSITYMAT_BACKWARD_H

#include "../rocdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Query workspace for an operator-action backward-differentiation pass.
 *  v0.1.0 returns \p ROCDENSITYMAT_STATUS_NOT_SUPPORTED. */
rocdensitymat_status rocdensitymat_operator_prepare_action_backward_diff(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    rocdensitymat_state state_in,
    rocdensitymat_state state_out,
    rocdensitymat_compute_type compute_type,
    size_t workspace_size_limit,
    rocdensitymat_workspace_descriptor workspace);

/*! \brief Compute the gradient of an operator-action with respect to the
 *         input state and any callback parameters.
 *  v0.1.0 returns \p ROCDENSITYMAT_STATUS_NOT_SUPPORTED. */
rocdensitymat_status rocdensitymat_operator_compute_action_backward_diff(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    double t,
    int32_t num_params,
    const double* params,
    rocdensitymat_state state_in,
    rocdensitymat_state state_out_grad,
    rocdensitymat_state state_in_grad,
    double* params_grad,
    rocdensitymat_workspace_descriptor workspace);

#ifdef __cplusplus
}
#endif

#endif /* ROCDENSITYMAT_BACKWARD_H */
