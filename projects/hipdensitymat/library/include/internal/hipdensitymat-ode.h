/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPDENSITYMAT_ODE_H
#define HIPDENSITYMAT_ODE_H

#include "../hipdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipdensitymatStatus_t hipdensitymatCreateMasterEquationSolver(
    hipdensitymatHandle_t handle,
    hipdensitymatOperator_t liouvillian,
    hipdensitymatSolverKind_t kind,
    hipdensitymatMasterEquationSolver_t* solver);

hipdensitymatStatus_t hipdensitymatDestroyMasterEquationSolver(
    hipdensitymatMasterEquationSolver_t solver);

hipdensitymatStatus_t hipdensitymatMasterEquationSolverPrepare(
    hipdensitymatHandle_t handle,
    hipdensitymatMasterEquationSolver_t solver,
    hipdensitymatState_t state,
    hipdensitymatComputeType_t computeType,
    size_t workspaceSizeLimit,
    hipdensitymatWorkspaceDescriptor_t workspace);

hipdensitymatStatus_t hipdensitymatMasterEquationStep(
    hipdensitymatHandle_t handle,
    hipdensitymatMasterEquationSolver_t solver,
    double t0, double dt,
    int32_t numParams, const double* params,
    hipdensitymatState_t state,
    hipdensitymatWorkspaceDescriptor_t workspace);

hipdensitymatStatus_t hipdensitymatMasterEquationStepN(
    hipdensitymatHandle_t handle,
    hipdensitymatMasterEquationSolver_t solver,
    double t0, double dt, int64_t numSteps,
    int32_t numParams, const double* params,
    hipdensitymatState_t state,
    hipdensitymatWorkspaceDescriptor_t workspace);

#ifdef __cplusplus
}
#endif

#endif /* HIPDENSITYMAT_ODE_H */
