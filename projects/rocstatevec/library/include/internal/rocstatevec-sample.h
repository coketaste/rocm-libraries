/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Reverse-engineered from cuStateVec public API documentation
 * (cuQuantum 24.11 / cuStateVec 1.7.x); clean-room implementation.
 * ************************************************************************ */

#ifndef ROCSTATEVEC_SAMPLE_H
#define ROCSTATEVEC_SAMPLE_H

#include "../rocstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Create a sampler descriptor and report the workspace size required
 *         for `rocstatevec_sampler_preprocess`.
 *  Bijection: \p custatevecSamplerCreate. */
rocstatevec_status rocstatevec_sampler_create(
    rocstatevec_handle               handle,
    const void*                      state_vector,
    rocstatevec_data_type            state_vector_data_type,
    uint32_t                         n_index_bits,
    rocstatevec_sampler_descriptor*  sampler,
    uint32_t                         n_max_shots,
    size_t*                          extra_workspace_size_in_bytes);

/*! \brief Destroy a sampler descriptor.
 *  Bijection: \p custatevecSamplerDestroy. */
rocstatevec_status rocstatevec_sampler_destroy(rocstatevec_sampler_descriptor sampler);

/*! \brief Build the cumulative-probability table inside the sampler descriptor.
 *  Bijection: \p custatevecSamplerPreprocess. */
rocstatevec_status rocstatevec_sampler_preprocess(
    rocstatevec_handle              handle,
    rocstatevec_sampler_descriptor  sampler,
    void*                           extra_workspace,
    size_t                          extra_workspace_size_in_bytes);

/*! \brief Read the squared norm of the statevector slice the sampler preprocessed.
 *  Bijection: \p custatevecSamplerGetSquaredNorm. */
rocstatevec_status rocstatevec_sampler_get_squared_norm(
    rocstatevec_handle              handle,
    rocstatevec_sampler_descriptor  sampler,
    double*                         norm);

/*! \brief Apply a sub-statevector probability offset to a sampler so that
 *         its drawn bit-strings can be combined with siblings on other devices.
 *  Bijection: \p custatevecSamplerApplySubSVOffset.
 *
 *  \note rocSTATEVEC v0.1.0 is single-node only and returns
 *        \c ROCSTATEVEC_STATUS_NOT_SUPPORTED unconditionally from this
 *        entry point. Single-node consumers should not call it. */
rocstatevec_status rocstatevec_sampler_apply_sub_sv_offset(
    rocstatevec_handle              handle,
    rocstatevec_sampler_descriptor  sampler,
    int32_t                         sub_sv_index,
    uint32_t                        n_sub_svs,
    double                          offset,
    double                          norm);

/*! \brief Draw \p n_shots independent bit-string samples and write them into
 *         the user-provided array. Bijection: \p custatevecSamplerSample. */
rocstatevec_status rocstatevec_sampler_sample(
    rocstatevec_handle              handle,
    rocstatevec_sampler_descriptor  sampler,
    rocstatevec_index_t*            bit_strings,
    const int32_t*                  bit_ordering,
    uint32_t                        bit_string_len,
    const double*                   randnums,
    uint32_t                        n_shots,
    rocstatevec_sampler_output      output);

#ifdef __cplusplus
}
#endif

#endif /* ROCSTATEVEC_SAMPLE_H */
