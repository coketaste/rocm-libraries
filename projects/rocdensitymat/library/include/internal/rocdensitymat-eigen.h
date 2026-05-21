/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

/*! \file
 *  \brief Operator eigenspectrum entry points (NOT_SUPPORTED in v0.1.0).
 */

#ifndef ROCDENSITYMAT_EIGEN_H
#define ROCDENSITYMAT_EIGEN_H

#include "../rocdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Query workspace for an eigenspectrum solve.
 *  v0.1.0 returns \p ROCDENSITYMAT_STATUS_NOT_SUPPORTED. */
rocdensitymat_status rocdensitymat_operator_prepare_eigenspectrum(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    rocdensitymat_state state,
    int32_t num_eigenpairs,
    rocdensitymat_compute_type compute_type,
    size_t workspace_size_limit,
    rocdensitymat_workspace_descriptor workspace);

/*! \brief Solve for the requested number of eigenpairs.
 *  v0.1.0 returns \p ROCDENSITYMAT_STATUS_NOT_SUPPORTED. */
rocdensitymat_status rocdensitymat_operator_compute_eigenspectrum(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    int32_t num_eigenpairs,
    rocdensitymat_complex_double* eigenvalues,
    rocdensitymat_state* eigenvectors,
    rocdensitymat_workspace_descriptor workspace);

#ifdef __cplusplus
}
#endif

#endif /* ROCDENSITYMAT_EIGEN_H */
