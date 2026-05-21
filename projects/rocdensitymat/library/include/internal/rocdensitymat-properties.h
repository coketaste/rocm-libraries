/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

/*! \file
 *  \brief Host-callable property queries layered over rocBLAS reductions.
 *
 *  These entry points are thin wrappers over the corresponding state
 *  routines in \p rocdensitymat-state.h, kept separate so the public
 *  reference docs match the cuDensityMat documentation grouping.
 */

#ifndef ROCDENSITYMAT_PROPERTIES_H
#define ROCDENSITYMAT_PROPERTIES_H

#include "../rocdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Query the device data-type tag of a state (precision + complexity). */
rocdensitymat_status rocdensitymat_state_get_data_type(
    rocdensitymat_handle handle, rocdensitymat_state state,
    rocdensitymat_data_type* data_type);

/*! \brief Query the purity of a state (\p PURE / \p MIXED / \p MPS). */
rocdensitymat_status rocdensitymat_state_get_purity(
    rocdensitymat_handle handle, rocdensitymat_state state,
    rocdensitymat_state_purity* purity);

/*! \brief Query the per-mode Hilbert-space dimensions of a state. */
rocdensitymat_status rocdensitymat_state_get_space_shape(
    rocdensitymat_handle handle, rocdensitymat_state state,
    int32_t* num_space_modes,
    int64_t* space_shape);

#ifdef __cplusplus
}
#endif

#endif /* ROCDENSITYMAT_PROPERTIES_H */
