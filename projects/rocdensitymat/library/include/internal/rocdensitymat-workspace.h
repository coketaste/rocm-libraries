/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

/*! \file
 *  \brief Workspace descriptor lifecycle and memory routing.
 */

#ifndef ROCDENSITYMAT_WORKSPACE_H
#define ROCDENSITYMAT_WORKSPACE_H

#include "../rocdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Create a workspace descriptor for use with prepare/compute paths.
 *
 *  Bijection: \p cudensitymatCreateWorkspace. */
rocdensitymat_status rocdensitymat_create_workspace(
    rocdensitymat_handle handle,
    rocdensitymat_workspace_descriptor* workspace);

/*! \brief Destroy a workspace descriptor previously returned by
 *         \p rocdensitymat_create_workspace. */
rocdensitymat_status rocdensitymat_destroy_workspace(
    rocdensitymat_workspace_descriptor workspace);

/*! \brief Query the memory size required by a previously-prepared workspace
 *         for a given memory space and workspace kind.
 *
 *  Bijection: \p cudensitymatWorkspaceGetMemorySize. */
rocdensitymat_status rocdensitymat_workspace_get_memory_size(
    rocdensitymat_handle handle,
    rocdensitymat_workspace_descriptor workspace,
    rocdensitymat_memspace mem_space,
    rocdensitymat_workspace_kind kind,
    size_t* memory_size_bytes);

/*! \brief Bind a user-allocated buffer to a workspace descriptor.
 *
 *  Bijection: \p cudensitymatWorkspaceSetMemory. */
rocdensitymat_status rocdensitymat_workspace_set_memory(
    rocdensitymat_handle handle,
    rocdensitymat_workspace_descriptor workspace,
    rocdensitymat_memspace mem_space,
    rocdensitymat_workspace_kind kind,
    void* memory_ptr,
    size_t memory_size_bytes);

/*! \brief Retrieve the buffer currently bound to a workspace descriptor.
 *
 *  Bijection: \p cudensitymatWorkspaceGetMemory. */
rocdensitymat_status rocdensitymat_workspace_get_memory(
    rocdensitymat_handle handle,
    rocdensitymat_workspace_descriptor workspace,
    rocdensitymat_memspace mem_space,
    rocdensitymat_workspace_kind kind,
    void** memory_ptr,
    size_t* memory_size_bytes);

#ifdef __cplusplus
}
#endif

#endif /* ROCDENSITYMAT_WORKSPACE_H */
