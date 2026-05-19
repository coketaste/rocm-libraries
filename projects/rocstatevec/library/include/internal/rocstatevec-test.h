/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Reverse-engineered from cuStateVec public API documentation
 * (cuQuantum 24.11 / cuStateVec 1.7.x); clean-room implementation.
 * ************************************************************************ */

#ifndef ROCSTATEVEC_TEST_H
#define ROCSTATEVEC_TEST_H

#include "../rocstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Workspace query for `rocstatevec_test_matrix_type`.
 *  Bijection: \p custatevecTestMatrixTypeGetWorkspaceSize. */
rocstatevec_status rocstatevec_test_matrix_type_get_workspace_size(
    rocstatevec_handle         handle,
    rocstatevec_matrix_type    matrix_type,
    const void*                matrix,
    rocstatevec_data_type      matrix_data_type,
    rocstatevec_matrix_layout  layout,
    uint32_t                   n_targets,
    int32_t                    adjoint,
    rocstatevec_compute_type   compute_type,
    size_t*                    extra_workspace_size_in_bytes);

/*! \brief Compute a residual that classifies how close a matrix is to the
 *         supplied matrix property (general / unitary / hermitian).
 *
 *  Bijection: \p custatevecTestMatrixType. */
rocstatevec_status rocstatevec_test_matrix_type(
    rocstatevec_handle         handle,
    double*                    residual_norm,
    rocstatevec_matrix_type    matrix_type,
    const void*                matrix,
    rocstatevec_data_type      matrix_data_type,
    rocstatevec_matrix_layout  layout,
    uint32_t                   n_targets,
    int32_t                    adjoint,
    rocstatevec_compute_type   compute_type,
    void*                      extra_workspace,
    size_t                     extra_workspace_size_in_bytes);

#ifdef __cplusplus
}
#endif

#endif /* ROCSTATEVEC_TEST_H */
