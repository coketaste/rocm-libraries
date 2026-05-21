/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

/*! \file
 *  \brief Distributed (multi-GPU multi-node) configuration.
 *
 *  All entry points return \p ROCDENSITYMAT_STATUS_NOT_SUPPORTED in
 *  v0.1.0; the symbols are kept ABI-stable so v0.1.0 callers will not
 *  break when the v0.2 implementation lands.
 */

#ifndef ROCDENSITYMAT_DISTRIBUTED_H
#define ROCDENSITYMAT_DISTRIBUTED_H

#include "../rocdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Bind a distributed communicator to a library context. */
rocdensitymat_status rocdensitymat_reset_distributed_configuration(
    rocdensitymat_handle handle,
    rocdensitymat_distributed_provider provider,
    const void* comm_ptr,
    size_t comm_size_bytes);

/*! \brief Number of ranks participating in a previously-bound communicator. */
rocdensitymat_status rocdensitymat_get_num_ranks(
    rocdensitymat_handle handle, int32_t* num_ranks);

/*! \brief This rank's index in a previously-bound communicator. */
rocdensitymat_status rocdensitymat_get_proc_rank(
    rocdensitymat_handle handle, int32_t* rank);

#ifdef __cplusplus
}
#endif

#endif /* ROCDENSITYMAT_DISTRIBUTED_H */
