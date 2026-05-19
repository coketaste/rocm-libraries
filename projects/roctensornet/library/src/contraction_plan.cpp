/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Contraction plan: lower a `contraction_optimizer_info` (a list of
 * pair_node entries) into a sequence of binary contraction ops that
 * the executor can dispatch in order. The plan also captures the
 * source network descriptor so the executor can look up input
 * extents/strides.
 *
 * Note: pair_node entries reference tensor IDs in [0, num_inputs +
 * len(path)). IDs < num_inputs map to original inputs; IDs >=
 * num_inputs map to the (i - num_inputs)-th virtual intermediate.
 * ************************************************************************ */

#include "roctensornet_descriptors.hpp"

#include <algorithm>
#include <unordered_map>

using namespace roctensornet;

extern "C" {

roctensornet_status
roctensornet_create_contraction_plan(roctensornet_handle                     h,
                                     roctensornet_network_descriptor         desc,
                                     roctensornet_contraction_optimizer_info info,
                                     roctensornet_workspace_descriptor       workspace,
                                     roctensornet_contraction_plan*          out)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    ROCTENSORNET_CHECK_PTR(out);
    auto* nd = cast<network_descriptor_st>(desc);
    auto* ii = cast<contraction_optimizer_info_st>(info);
    if(nd == nullptr || ii == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    (void)workspace; /* required-size cross-check is the workspace API's job */

    auto* plan = new (std::nothrow) contraction_plan_st();
    if(plan == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    plan->network = *nd;
    plan->info    = *ii;

    /* Track current tensors in a virtual pool indexed by id. */
    struct entry { std::vector<int32_t> modes; std::vector<roctensornet_index_t> extents; std::vector<roctensornet_index_t> strides; };
    std::unordered_map<int32_t, entry> pool;
    for(int32_t i = 0; i < nd->num_inputs; ++i)
        pool[i] = entry{nd->inputs[i].modes, nd->inputs[i].extents, nd->inputs[i].strides};
    int32_t next_intermediate_id = nd->num_inputs;

    plan->pairs.reserve(ii->path.size());
    for(const auto& node : ii->path)
    {
        contraction_plan_st::pair_op p;
        p.left  = node.left;
        p.right = node.right;
        p.out_modes   = node.out_modes;
        p.out_extents = node.out_extents;

        auto itL = pool.find(node.left);
        auto itR = pool.find(node.right);
        if(itL == pool.end() || itR == pool.end()) { delete plan; return ROCTENSORNET_STATUS_INTERNAL_ERROR; }
        const auto& L = itL->second;
        const auto& R = itR->second;
        p.left_modes    = L.modes;
        p.right_modes   = R.modes;
        p.left_extents  = L.extents;
        p.right_extents = R.extents;

        /* contracted = modes appearing in both L and R, absent from out */
        for(auto m : L.modes)
        {
            bool inR   = std::find(R.modes.begin(), R.modes.end(), m) != R.modes.end();
            bool inOut = std::find(p.out_modes.begin(), p.out_modes.end(), m) != p.out_modes.end();
            if(inR && !inOut) p.contracted_modes.push_back(m);
        }

        /* Compute strides for the new intermediate (generalized column-major over out_extents) */
        std::vector<roctensornet_index_t> out_strides;
        generalized_column_major_strides(p.out_extents, out_strides);

        /* Insert virtual intermediate into the pool */
        pool.erase(itL); pool.erase(itR);
        pool[next_intermediate_id++] = entry{p.out_modes, p.out_extents, out_strides};
        plan->pairs.push_back(std::move(p));
    }

    *out = reinterpret_cast<roctensornet_contraction_plan>(plan);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_destroy_contraction_plan(roctensornet_contraction_plan plan)
{
    delete cast<contraction_plan_st>(plan);
    return ROCTENSORNET_STATUS_SUCCESS;
}

} // extern "C"
