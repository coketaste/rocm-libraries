/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Operator eigenspectrum (NOT_SUPPORTED in v0.1.0). The full
 * rocSOLVER-backed solve lands in v0.2; the symbols are kept ABI-stable
 * so v0.1.0 callers do not break in the meantime.
 * ************************************************************************ */

#include "rocdensitymat_descriptors.hpp"

extern "C" {

rocdensitymat_status rocdensitymat_operator_prepare_eigenspectrum(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    rocdensitymat_state state,
    int32_t /*num_eigenpairs*/,
    rocdensitymat_compute_type /*compute_type*/,
    size_t /*workspace_size_limit*/,
    rocdensitymat_workspace_descriptor /*workspace*/)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(op);
    ROCDENSITYMAT_CHECK_PTR(state);
    return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;
}

rocdensitymat_status rocdensitymat_operator_compute_eigenspectrum(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    int32_t /*num_eigenpairs*/,
    rocdensitymat_complex_double* /*eigenvalues*/,
    rocdensitymat_state* /*eigenvectors*/,
    rocdensitymat_workspace_descriptor /*workspace*/)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(op);
    return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;
}

} // extern "C"
