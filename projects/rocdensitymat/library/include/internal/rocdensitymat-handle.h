/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

/*! \file
 *  \brief Library context handle lifecycle and stream binding.
 */

#ifndef ROCDENSITYMAT_HANDLE_H
#define ROCDENSITYMAT_HANDLE_H

#include "../rocdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Create a new rocDENSITYMAT library context handle.
 *
 *  Bijection: \p cudensitymatCreate. */
rocdensitymat_status rocdensitymat_create(rocdensitymat_handle* handle);

/*! \brief Destroy a rocDENSITYMAT library context handle.
 *
 *  Bijection: \p cudensitymatDestroy. */
rocdensitymat_status rocdensitymat_destroy(rocdensitymat_handle handle);

/*! \brief Reset the per-context random seed used by stochastic kernels.
 *
 *  Bijection: \p cudensitymatResetRandomSeed. */
rocdensitymat_status rocdensitymat_reset_random_seed(
    rocdensitymat_handle handle, uint32_t seed);

/*! \brief Bind a HIP stream to a library context. Subsequent operations are
 *         enqueued on this stream. */
rocdensitymat_status rocdensitymat_set_stream(rocdensitymat_handle handle,
                                              hipStream_t stream);

/*! \brief Retrieve the HIP stream currently bound to a library context. */
rocdensitymat_status rocdensitymat_get_stream(rocdensitymat_handle handle,
                                              hipStream_t* stream);

#ifdef __cplusplus
}
#endif

#endif /* ROCDENSITYMAT_HANDLE_H */
