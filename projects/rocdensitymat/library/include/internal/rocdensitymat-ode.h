/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

/*! \file
 *  \brief Master-equation ODE stepper. v0.1.0 ships fixed-step RK4.
 */

#ifndef ROCDENSITYMAT_ODE_H
#define ROCDENSITYMAT_ODE_H

#include "../rocdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Create a master-equation solver bound to a Liouvillian operator.
 *
 *  Bijection: \p cudensitymatCreateMasterEquationSolver. */
rocdensitymat_status rocdensitymat_create_master_equation_solver(
    rocdensitymat_handle handle,
    rocdensitymat_operator liouvillian,
    rocdensitymat_solver_kind kind,
    rocdensitymat_master_equation_solver* solver);

/*! \brief Destroy a master-equation solver. */
rocdensitymat_status rocdensitymat_destroy_master_equation_solver(
    rocdensitymat_master_equation_solver solver);

/*! \brief Query and bind the workspace required by the stepper.
 *
 *  After this call the workspace descriptor reports the per-step scratch
 *  size required by \p rocdensitymat_master_equation_step. The caller is
 *  expected to allocate and bind a buffer of at least the reported size. */
rocdensitymat_status rocdensitymat_master_equation_solver_prepare(
    rocdensitymat_handle handle,
    rocdensitymat_master_equation_solver solver,
    rocdensitymat_state state,
    rocdensitymat_compute_type compute_type,
    size_t workspace_size_limit,
    rocdensitymat_workspace_descriptor workspace);

/*! \brief Advance the supplied state by one fixed-step integration of size
 *         ``dt`` starting from time ``t0``. The state is updated in place.
 *
 *  Bijection: \p cudensitymatMasterEquationStep. */
rocdensitymat_status rocdensitymat_master_equation_step(
    rocdensitymat_handle handle,
    rocdensitymat_master_equation_solver solver,
    double t0,
    double dt,
    int32_t num_params,
    const double* params,
    rocdensitymat_state state,
    rocdensitymat_workspace_descriptor workspace);

/*! \brief Iterate \p rocdensitymat_master_equation_step \p num_steps times. */
rocdensitymat_status rocdensitymat_master_equation_step_n(
    rocdensitymat_handle handle,
    rocdensitymat_master_equation_solver solver,
    double t0,
    double dt,
    int64_t num_steps,
    int32_t num_params,
    const double* params,
    rocdensitymat_state state,
    rocdensitymat_workspace_descriptor workspace);

#ifdef __cplusplus
}
#endif

#endif /* ROCDENSITYMAT_ODE_H */
