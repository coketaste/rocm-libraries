/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Marginal / Sampler / Expectation / Accessor entry points.
 * ************************************************************************ */

#ifndef ROCTENSORNET_DERIVED_H
#define ROCTENSORNET_DERIVED_H

#include "../roctensornet-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Marginal ---- */
roctensornet_status
roctensornet_create_marginal (roctensornet_handle handle,
                              roctensornet_state  state,
                              int32_t             num_marginal_modes,
                              const int32_t*      marginal_modes,
                              int32_t             num_projected_modes,
                              const int32_t*      projected_modes,
                              const int64_t*      marginal_tensor_strides,
                              roctensornet_state_marginal* marginal);

roctensornet_status
roctensornet_destroy_marginal (roctensornet_state_marginal marginal);

roctensornet_status
roctensornet_marginal_configure (roctensornet_handle             handle,
                                 roctensornet_state_marginal     marginal,
                                 roctensornet_marginal_attribute attr,
                                 const void*                     value,
                                 size_t                          size);

roctensornet_status
roctensornet_marginal_get_info (roctensornet_handle             handle,
                                roctensornet_state_marginal     marginal,
                                roctensornet_marginal_attribute attr,
                                void*                           value,
                                size_t                          size);

roctensornet_status
roctensornet_marginal_prepare (roctensornet_handle               handle,
                               roctensornet_state_marginal       marginal,
                               size_t                            max_workspace_size_device,
                               roctensornet_workspace_descriptor workspace,
                               hipStream_t                       stream);

roctensornet_status
roctensornet_marginal_compute (roctensornet_handle               handle,
                               roctensornet_state_marginal       marginal,
                               const int64_t*                    projected_mode_values,
                               roctensornet_workspace_descriptor workspace,
                               void*                             marginal_tensor,
                               hipStream_t                       stream);

/* ---- Sampler ---- */
roctensornet_status
roctensornet_create_sampler (roctensornet_handle handle,
                             roctensornet_state  state,
                             int32_t             num_modes_to_sample,
                             const int32_t*      modes_to_sample,
                             roctensornet_state_sampler* sampler);

roctensornet_status
roctensornet_destroy_sampler (roctensornet_state_sampler sampler);

roctensornet_status
roctensornet_sampler_configure (roctensornet_handle           handle,
                                roctensornet_state_sampler    sampler,
                                roctensornet_sampler_attribute attr,
                                const void*                   value,
                                size_t                        size);

roctensornet_status
roctensornet_sampler_get_info (roctensornet_handle             handle,
                               roctensornet_state_sampler      sampler,
                               roctensornet_sampler_attribute  attr,
                               void*                           value,
                               size_t                          size);

roctensornet_status
roctensornet_sampler_prepare (roctensornet_handle               handle,
                              roctensornet_state_sampler        sampler,
                              size_t                            max_workspace_size_device,
                              roctensornet_workspace_descriptor workspace,
                              hipStream_t                       stream);

roctensornet_status
roctensornet_sampler_sample (roctensornet_handle               handle,
                             roctensornet_state_sampler        sampler,
                             int64_t                           num_shots,
                             roctensornet_workspace_descriptor workspace,
                             int64_t*                          samples,
                             hipStream_t                       stream);

/* ---- Expectation ---- */
roctensornet_status
roctensornet_create_expectation (roctensornet_handle handle,
                                 roctensornet_state  state,
                                 roctensornet_network_operator op,
                                 roctensornet_state_expectation* expectation);

roctensornet_status
roctensornet_destroy_expectation (roctensornet_state_expectation expectation);

roctensornet_status
roctensornet_expectation_configure (roctensornet_handle             handle,
                                    roctensornet_state_expectation  expectation,
                                    roctensornet_expectation_attribute attr,
                                    const void*                     value,
                                    size_t                          size);

roctensornet_status
roctensornet_expectation_get_info (roctensornet_handle             handle,
                                   roctensornet_state_expectation  expectation,
                                   roctensornet_expectation_attribute attr,
                                   void*                           value,
                                   size_t                          size);

roctensornet_status
roctensornet_expectation_prepare (roctensornet_handle               handle,
                                  roctensornet_state_expectation    expectation,
                                  size_t                            max_workspace_size_device,
                                  roctensornet_workspace_descriptor workspace,
                                  hipStream_t                       stream);

roctensornet_status
roctensornet_expectation_compute (roctensornet_handle               handle,
                                  roctensornet_state_expectation    expectation,
                                  roctensornet_workspace_descriptor workspace,
                                  void*                             expectation_value,
                                  void*                             state_norm,
                                  hipStream_t                       stream);

/* ---- Accessor ---- */
roctensornet_status
roctensornet_create_accessor (roctensornet_handle handle,
                              roctensornet_state  state,
                              int32_t             num_projected_modes,
                              const int32_t*      projected_modes,
                              const int64_t*      amplitudes_tensor_strides,
                              roctensornet_state_accessor* accessor);

roctensornet_status
roctensornet_destroy_accessor (roctensornet_state_accessor accessor);

roctensornet_status
roctensornet_accessor_configure (roctensornet_handle             handle,
                                 roctensornet_state_accessor     accessor,
                                 roctensornet_accessor_attribute attr,
                                 const void*                     value,
                                 size_t                          size);

roctensornet_status
roctensornet_accessor_get_info (roctensornet_handle             handle,
                                roctensornet_state_accessor     accessor,
                                roctensornet_accessor_attribute attr,
                                void*                           value,
                                size_t                          size);

roctensornet_status
roctensornet_accessor_prepare (roctensornet_handle               handle,
                               roctensornet_state_accessor       accessor,
                               size_t                            max_workspace_size_device,
                               roctensornet_workspace_descriptor workspace,
                               hipStream_t                       stream);

roctensornet_status
roctensornet_accessor_compute (roctensornet_handle               handle,
                               roctensornet_state_accessor       accessor,
                               const int64_t*                    projected_mode_values,
                               roctensornet_workspace_descriptor workspace,
                               void*                             amplitudes_tensor,
                               void*                             state_norm,
                               hipStream_t                       stream);

#ifdef __cplusplus
}
#endif

#endif /* ROCTENSORNET_DERIVED_H */
