/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

/*! \file
 *  \brief Pure / mixed quantum-state lifecycle and component access.
 */

#ifndef ROCDENSITYMAT_STATE_H
#define ROCDENSITYMAT_STATE_H

#include "../rocdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Create a quantum state of the requested purity over the supplied
 *         tensor-product Hilbert space.
 *
 *  ``space_shape`` is a length-``num_space_modes`` array of per-mode
 *  Hilbert-space dimensions (qubits use dimension 2). ``batch_size`` must
 *  be 1 in v0.1.0; values >1 return \p NOT_SUPPORTED.
 *  ``ROCDENSITYMAT_STATE_PURITY_MPS`` returns \p NOT_SUPPORTED in v0.1.0.
 *
 *  Bijection: \p cudensitymatCreateState. */
rocdensitymat_status rocdensitymat_create_state(
    rocdensitymat_handle handle,
    rocdensitymat_state_purity purity,
    int32_t num_space_modes,
    const int64_t* space_shape,
    int64_t batch_size,
    rocdensitymat_data_type data_type,
    rocdensitymat_state* state);

/*! \brief Destroy a quantum-state descriptor.
 *
 *  Bijection: \p cudensitymatDestroyState. */
rocdensitymat_status rocdensitymat_destroy_state(rocdensitymat_state state);

/*! \brief Number of dense components backing the state representation
 *         (1 for ``PURITY_PURE`` / ``PURITY_MIXED`` in v0.1.0). */
rocdensitymat_status rocdensitymat_state_get_num_components(
    rocdensitymat_handle handle,
    rocdensitymat_state state,
    int32_t* num_components);

/*! \brief Per-component shape and storage-size query, used together with
 *         \p rocdensitymat_state_attach_component_buffer. */
rocdensitymat_status rocdensitymat_state_get_component_info(
    rocdensitymat_handle handle,
    rocdensitymat_state state,
    int32_t component_id,
    int32_t* num_modes,
    int64_t* mode_extents,
    size_t* component_size_bytes);

/*! \brief Attach a user-owned device buffer as the storage for a single
 *         component of the state. The buffer must be at least the size
 *         returned by \p rocdensitymat_state_get_component_info. */
rocdensitymat_status rocdensitymat_state_attach_component_buffer(
    rocdensitymat_handle handle,
    rocdensitymat_state state,
    int32_t component_id,
    void* component_buffer,
    size_t component_buffer_size);

/*! \brief Initialize a state to the all-zeros tensor (vacuum amplitude
 *         and density matrix). */
rocdensitymat_status rocdensitymat_state_initialize_zero(
    rocdensitymat_handle handle, rocdensitymat_state state);

/*! \brief Initialize a state to the maximally-mixed (or maximally-uniform)
 *         distribution. */
rocdensitymat_status rocdensitymat_state_initialize_uniform(
    rocdensitymat_handle handle, rocdensitymat_state state);

/*! \brief Initialize a state to the Hilbert-space basis vector indexed by
 *         the mode-major coordinate ``basis_indices``. */
rocdensitymat_status rocdensitymat_state_initialize_basis(
    rocdensitymat_handle handle, rocdensitymat_state state,
    const int64_t* basis_indices);

/*! \brief Compute the L2 / Frobenius norm of the state.
 *
 *  ``norm`` is a host pointer receiving a single ``double``. */
rocdensitymat_status rocdensitymat_state_compute_norm(
    rocdensitymat_handle handle, rocdensitymat_state state,
    rocdensitymat_workspace_descriptor workspace, double* norm);

/*! \brief Compute the trace of a mixed state. Pure states return
 *         \p ROCDENSITYMAT_STATUS_INVALID_VALUE. */
rocdensitymat_status rocdensitymat_state_compute_trace(
    rocdensitymat_handle handle, rocdensitymat_state state,
    rocdensitymat_workspace_descriptor workspace,
    rocdensitymat_complex_double* trace);

/*! \brief Compute the inner product / fidelity-like overlap of two states.
 *
 *  Pure-pure: \p <psi1|psi2>. Mixed-mixed: \p Tr(rho1^dagger rho2). */
rocdensitymat_status rocdensitymat_state_compute_overlap(
    rocdensitymat_handle handle, rocdensitymat_state lhs,
    rocdensitymat_state rhs,
    rocdensitymat_workspace_descriptor workspace,
    rocdensitymat_complex_double* overlap);

#ifdef __cplusplus
}
#endif

#endif /* ROCDENSITYMAT_STATE_H */
