/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Concrete (non-opaque) definitions for the descriptor structs.
 * Only translation units inside roctensornet should include this.
 * ************************************************************************ */

#ifndef ROCTENSORNET_DESCRIPTORS_HPP
#define ROCTENSORNET_DESCRIPTORS_HPP

#include "roctensornet_internal.hpp"

#include <cstdint>
#include <vector>

namespace roctensornet
{

/* ---- Tensor descriptor (`roctensornet_tensor_descriptor`) ---- */
struct tensor_descriptor_st
{
    std::vector<int32_t>              modes;
    std::vector<roctensornet_index_t> extents;
    std::vector<roctensornet_index_t> strides;
    roctensornet_data_type            data_type = ROCTENSORNET_R_32F;
    uint32_t                          alignment = 256;

    int32_t num_modes() const { return static_cast<int32_t>(modes.size()); }
    size_t  num_elements() const
    {
        size_t n = 1;
        for(auto e : extents) n *= static_cast<size_t>(e);
        return n;
    }
    size_t  num_bytes() const { return num_elements() * element_size_bytes(data_type); }
};

/* ---- Generalized column-major packed strides for a given extent list. */
inline void generalized_column_major_strides(const std::vector<roctensornet_index_t>& extents,
                                             std::vector<roctensornet_index_t>&        strides)
{
    strides.resize(extents.size());
    roctensornet_index_t s = 1;
    for(size_t i = 0; i < extents.size(); ++i)
    {
        strides[i] = s;
        s *= extents[i];
    }
}

/* ---- Network descriptor (`roctensornet_network_descriptor`) ---- */
struct network_descriptor_st
{
    int32_t                                num_inputs = 0;
    std::vector<tensor_descriptor_st>      inputs;
    tensor_descriptor_st                   output;
    roctensornet_data_type                 data_type    = ROCTENSORNET_R_32F;
    roctensornet_compute_type              compute_type = ROCTENSORNET_COMPUTE_DEFAULT;
};

/* ---- Optimizer config (`roctensornet_contraction_optimizer_config`) ---- */
struct contraction_optimizer_config_st
{
    int32_t  num_graph_iterations    = 8;
    int32_t  num_graph_cuts          = 1;
    int32_t  graph_algorithm         = 0;
    int32_t  reconfig_num_iterations = 500;
    int32_t  reconfig_num_leaves     = 8;
    int32_t  slicer_disable_slicing  = 0;
    int32_t  slicer_memory_model     = 0;
    double   slicer_memory_factor    = 0.8;
    int32_t  seed                    = 42;
    int32_t  cost_function_objective = 0;
    int32_t  cache_reuse_nruns       = 0;
    int32_t  smart_option            = 0;
};

/* ---- Pair-tree node used in the optimizer info.
 *
 *  Each node is an entry in the contraction path: tensor IDs i, j
 *  contract into a virtual intermediate identified by the node's own
 *  index in the path array. The "tensor IDs" in a node are positive
 *  for original input tensors (0..num_inputs-1) and >= num_inputs for
 *  virtual intermediates (i.e. virtual intermediate k has id
 *  num_inputs + k).
 *
 *  We carry the resulting intermediate's mode list so the executor can
 *  feed binary contractions without a re-walk.
 */
struct pair_node
{
    int32_t              left  = -1; /* tensor id (input or intermediate) */
    int32_t              right = -1;
    std::vector<int32_t> out_modes;
    std::vector<roctensornet_index_t> out_extents;
    size_t               flops = 0;
    size_t               intermediate_bytes = 0;
};

/* ---- Optimizer info (`roctensornet_contraction_optimizer_info`) ---- */
struct contraction_optimizer_info_st
{
    std::vector<pair_node>            path;
    std::vector<int32_t>              sliced_modes;     /* mode IDs picked for slicing */
    std::vector<roctensornet_index_t> sliced_extents;   /* parallel array */
    int64_t                           num_slices    = 1;
    size_t                            flop_count    = 0;
    size_t                            largest_tensor = 0;
    /* Snapshot of the source network descriptor for cross-checks. */
    int32_t                           num_inputs    = 0;
    int32_t                           num_output_modes = 0;
    std::vector<int32_t>              output_modes;
    std::vector<roctensornet_index_t> output_extents;
    roctensornet_data_type            data_type     = ROCTENSORNET_R_32F;
    roctensornet_compute_type         compute_type  = ROCTENSORNET_COMPUTE_DEFAULT;
};

/* ---- Workspace descriptor ---- */
struct workspace_slot
{
    void*   buffer       = nullptr;
    int64_t buffer_bytes = 0;
    int64_t required     = 0;
    bool    user_owned   = false;
};

struct workspace_descriptor_st
{
    /* [memspace][kind] -> slot */
    workspace_slot device_scratch;
    workspace_slot device_cache;
    workspace_slot host_scratch;
    workspace_slot host_cache;

    workspace_slot* select(roctensornet_memspace ms, roctensornet_workspace_kind k)
    {
        if(ms == ROCTENSORNET_MEMSPACE_DEVICE && k == ROCTENSORNET_WORKSPACE_SCRATCH) return &device_scratch;
        if(ms == ROCTENSORNET_MEMSPACE_DEVICE && k == ROCTENSORNET_WORKSPACE_CACHE)   return &device_cache;
        if(ms == ROCTENSORNET_MEMSPACE_HOST   && k == ROCTENSORNET_WORKSPACE_SCRATCH) return &host_scratch;
        if(ms == ROCTENSORNET_MEMSPACE_HOST   && k == ROCTENSORNET_WORKSPACE_CACHE)   return &host_cache;
        return nullptr;
    }
};

/* ---- Slice group ---- */
struct slice_group_st
{
    std::vector<int64_t> slice_ids;
};

/* ---- Autotune preference ---- */
struct contraction_autotune_preference_st
{
    int32_t max_iterations    = 3;
    int32_t intermediate_modes = 0;
    int32_t gemm_algorithm    = 0;
};

/* ---- Contraction plan (pre-compiled pair sequence) ---- */
struct contraction_plan_st
{
    network_descriptor_st                   network;
    contraction_optimizer_info_st           info;
    /* For each pair_node in info.path: precomputed contraction signature. */
    struct pair_op
    {
        int32_t                           left;
        int32_t                           right;
        std::vector<int32_t>              left_modes;
        std::vector<int32_t>              right_modes;
        std::vector<int32_t>              out_modes;
        std::vector<roctensornet_index_t> left_extents;
        std::vector<roctensornet_index_t> right_extents;
        std::vector<roctensornet_index_t> out_extents;
        std::vector<int32_t>              contracted_modes; /* shared between L and R, removed in output */
    };
    std::vector<pair_op> pairs;
};

/* ---- SVD config / info ---- */
struct tensor_svd_config_st
{
    double                          abs_cutoff             = 0.0;
    double                          rel_cutoff             = 0.0;
    roctensornet_tensor_svd_normalization s_normalization  = ROCTENSORNET_TENSOR_SVD_NORMALIZATION_NONE;
    roctensornet_tensor_svd_partition     s_partition      = ROCTENSORNET_TENSOR_SVD_PARTITION_NONE;
    roctensornet_tensor_svd_algo          algo             = ROCTENSORNET_TENSOR_SVD_ALGO_GESVD;
    int64_t                         max_extent             = 0; /* 0 = no limit */
    double                          discarded_weight_cutoff = 0.0;
};

struct tensor_svd_info_st
{
    int64_t full_extent      = 0;
    int64_t reduced_extent   = 0;
    double  discarded_weight = 0.0;
    int32_t algo             = 0;
    int32_t algo_status      = 0;
};

/* ---- Network operator ---- */
struct network_operator_component
{
    void*                                    coefficient_mem = nullptr; /* heap-owned, dtype-sized */
    std::vector<std::vector<int32_t>>        per_tensor_modes;
    std::vector<const void*>                 per_tensor_data;
    int32_t                                  kind = 0; /* 0 = product, 1 = MPO */
    int32_t                                  boundary_condition = 0;
};

struct network_operator_st
{
    int32_t                                num_state_modes = 0;
    std::vector<roctensornet_index_t>      state_mode_extents;
    roctensornet_data_type                 data_type    = ROCTENSORNET_R_32F;
    std::vector<network_operator_component> components;
};

/* ---- Network state ---- */
struct state_gate_record
{
    std::vector<int32_t>              state_modes;
    std::vector<int32_t>              control_modes;
    std::vector<int64_t>              control_values;
    void*                             tensor_data        = nullptr;
    std::vector<int64_t>              tensor_mode_strides;
    int32_t                           immutable          = 1;
    int32_t                           adjoint            = 0;
    int32_t                           unitary            = 1;
    int32_t                           kind               = 0; /* 0=tensor, 1=ctrl, 2=unitary channel, 3=general channel */
    /* For channel kinds, store the multiple Kraus tensors. */
    std::vector<void*>                channel_tensors;
    std::vector<std::vector<int64_t>> channel_strides;
    std::vector<double>               channel_probs;
    int64_t                           id = -1;
};

struct network_state_st
{
    roctensornet_state_purity              purity = ROCTENSORNET_STATE_PURITY_PURE;
    int32_t                                num_state_modes = 0;
    std::vector<roctensornet_index_t>      state_mode_extents;
    roctensornet_data_type                 data_type = ROCTENSORNET_R_32F;
    std::vector<state_gate_record>         gates;
    /* Configurable attributes. */
    int32_t                                num_hyper_samples = 0;
    int32_t                                mps_canonical_center = -1;
    double                                 mps_svd_abs_cutoff = 0.0;
    double                                 mps_svd_rel_cutoff = 0.0;
    int32_t                                mps_svd_s_norm   = 0;
    int32_t                                mps_svd_s_part   = 0;
    int32_t                                mps_svd_algo     = 0;
    int32_t                                mps_mpo_app      = 0;
    int32_t                                mps_gauge_option = 0;
    size_t                                 prepared_flops   = 0;
    bool                                   prepared         = false;
};

/* ---- Marginal / Sampler / Expectation / Accessor ---- */
struct state_marginal_st
{
    roctensornet_state            state          = nullptr;
    std::vector<int32_t>          marginal_modes;
    std::vector<int32_t>          projected_modes;
    std::vector<int64_t>          marginal_tensor_strides;
    int32_t                       num_hyper_samples = 0;
    size_t                        flops             = 0;
    bool                          prepared          = false;
};

struct state_sampler_st
{
    roctensornet_state            state             = nullptr;
    std::vector<int32_t>          modes_to_sample;
    int32_t                       num_hyper_samples = 0;
    int32_t                       deterministic     = 0;
    size_t                        flops             = 0;
    bool                          prepared          = false;
};

struct state_expectation_st
{
    roctensornet_state            state          = nullptr;
    roctensornet_network_operator op             = nullptr;
    int32_t                       num_hyper_samples = 0;
    size_t                        flops             = 0;
    bool                          prepared          = false;
};

struct state_accessor_st
{
    roctensornet_state            state          = nullptr;
    std::vector<int32_t>          projected_modes;
    std::vector<int64_t>          amplitudes_tensor_strides;
    int32_t                       num_hyper_samples = 0;
    size_t                        flops             = 0;
    bool                          prepared          = false;
};

/* ---- Cast helpers from opaque to concrete. */
template <typename Concrete, typename Opaque>
inline Concrete* cast(Opaque o) { return reinterpret_cast<Concrete*>(o); }

} // namespace roctensornet

#endif /* ROCTENSORNET_DESCRIPTORS_HPP */
