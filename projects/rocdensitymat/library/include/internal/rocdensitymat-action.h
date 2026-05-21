/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

/*! \file
 *  \brief Operator action and expectation-value compute paths.
 */

#ifndef ROCDENSITYMAT_ACTION_H
#define ROCDENSITYMAT_ACTION_H

#include "../rocdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Query the workspace size required by \p rocdensitymat_operator_compute_action.
 *
 *  After this call \p rocdensitymat_workspace_get_memory_size returns a
 *  non-zero scratch size for \p ROCDENSITYMAT_MEMSPACE_DEVICE +
 *  \p ROCDENSITYMAT_WORKSPACE_SCRATCH that the caller must satisfy via
 *  \p rocdensitymat_workspace_set_memory before calling
 *  \p rocdensitymat_operator_compute_action.
 *
 *  Bijection: \p cudensitymatOperatorPrepareAction. */
rocdensitymat_status rocdensitymat_operator_prepare_action(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    rocdensitymat_state state_in,
    rocdensitymat_state state_out,
    rocdensitymat_compute_type compute_type,
    size_t workspace_size_limit,
    rocdensitymat_workspace_descriptor workspace);

/*! \brief Compute \f$|\psi_{\text{out}}\rangle = \alpha\, O |\psi_{\text{in}}\rangle\f$
 *         (pure) or \f$\rho_{\text{out}} = \alpha\, [O, \rho_{\text{in}}]\f$
 *         (mixed) at simulation time \p t with optional callback parameters.
 *
 *  Bijection: \p cudensitymatOperatorComputeAction. */
rocdensitymat_status rocdensitymat_operator_compute_action(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    double t,
    int32_t num_params,
    const double* params,
    rocdensitymat_state state_in,
    rocdensitymat_state state_out,
    rocdensitymat_workspace_descriptor workspace);

/*! \brief Compute \f$\langle\psi|O|\psi\rangle\f$ (pure) or
 *         \f$\mathrm{Tr}(O \rho)\f$ (mixed). */
rocdensitymat_status rocdensitymat_operator_compute_expectation(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    double t,
    int32_t num_params,
    const double* params,
    rocdensitymat_state state,
    rocdensitymat_workspace_descriptor workspace,
    rocdensitymat_complex_double* expectation);

#ifdef __cplusplus
}
#endif

#endif /* ROCDENSITYMAT_ACTION_H */
