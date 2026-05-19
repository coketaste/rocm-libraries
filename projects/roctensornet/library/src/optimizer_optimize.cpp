/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Contraction path finder.
 *
 * The v0.1.0 implementation is a deterministic greedy heuristic:
 *
 *   1. Build a working pool of (modes, extents, dtype) records, one per
 *      input tensor (id 0..n-1).
 *   2. Pick the pair (i, j) with the smallest resulting intermediate
 *      tensor (size = product of the union-minus-contracted extents).
 *      Tie-break on the lowest FLOP count for the pair.
 *   3. Replace tensors i and j in the pool with the new intermediate;
 *      record the pair in `info.path`.
 *   4. Repeat until one tensor remains; that's the network output.
 *
 * Slicing: if the largest intermediate exceeds the workspace
 * constraint, pick mode(s) common across the largest pair and slice
 * over their extent until the size fits. The path itself is unchanged
 * by slicing - only `sliced_modes` and `num_slices` are populated.
 *
 * The heuristic is intentionally simple. The plan documents that a
 * better optimizer is a planned follow-up.
 * ************************************************************************ */

#include "roctensornet_descriptors.hpp"

#include <algorithm>
#include <set>
#include <vector>

using namespace roctensornet;

namespace
{

/* A tensor in the working pool. */
struct pool_entry
{
    int32_t              id;       /* >= 0 if input, else assigned by us */
    std::vector<int32_t> modes;
    std::vector<roctensornet_index_t> extents;
};

/* Compute the contraction outcome of two pool entries:
 *   - output modes = symmetric difference of their mode sets
 *   - contracted modes = intersection
 *   - FLOPs ~ product(extents over union of modes) * 2
 */
struct pair_eval
{
    std::vector<int32_t>              out_modes;
    std::vector<roctensornet_index_t> out_extents;
    std::vector<int32_t>              contracted;
    size_t                            flops = 0;
    size_t                            out_size = 1;
};

pair_eval evaluate_pair(const pool_entry& a, const pool_entry& b)
{
    pair_eval r;
    std::set<int32_t> all;
    all.insert(a.modes.begin(), a.modes.end());
    all.insert(b.modes.begin(), b.modes.end());

    /* Extents are uniquely keyed by mode id (guaranteed by descriptor). */
    auto extent_of = [&](int32_t m) -> roctensornet_index_t {
        for(size_t k = 0; k < a.modes.size(); ++k)
            if(a.modes[k] == m) return a.extents[k];
        for(size_t k = 0; k < b.modes.size(); ++k)
            if(b.modes[k] == m) return b.extents[k];
        return 1;
    };

    size_t prod_all = 1;
    for(auto m : all) prod_all *= static_cast<size_t>(extent_of(m));
    r.flops = prod_all * 2;

    /* contracted = modes shared between a and b */
    std::set<int32_t> sb(b.modes.begin(), b.modes.end());
    for(auto m : a.modes)
        if(sb.count(m)) r.contracted.push_back(m);
    std::sort(r.contracted.begin(), r.contracted.end());

    /* output modes = symmetric difference, sorted by mode id */
    std::vector<int32_t> outm;
    for(auto m : a.modes)
        if(!sb.count(m)) outm.push_back(m);
    std::set<int32_t> sa(a.modes.begin(), a.modes.end());
    for(auto m : b.modes)
        if(!sa.count(m)) outm.push_back(m);
    std::sort(outm.begin(), outm.end());
    r.out_modes = std::move(outm);

    r.out_extents.reserve(r.out_modes.size());
    r.out_size = 1;
    for(auto m : r.out_modes)
    {
        auto e = extent_of(m);
        r.out_extents.push_back(e);
        r.out_size *= static_cast<size_t>(e);
    }
    return r;
}

} // anon

extern "C" {

roctensornet_status
roctensornet_contraction_optimize(roctensornet_handle                       handle,
                                  roctensornet_network_descriptor           desc,
                                  roctensornet_contraction_optimizer_config cfg_in,
                                  uint64_t                                  workspace_size_constraint,
                                  roctensornet_contraction_optimizer_info   info_in)
{
    ROCTENSORNET_CHECK_HANDLE(handle);
    auto* nd   = cast<network_descriptor_st>(desc);
    auto* info = cast<contraction_optimizer_info_st>(info_in);
    if(nd == nullptr || info == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;

    contraction_optimizer_config_st default_cfg;
    auto* cfg = cast<contraction_optimizer_config_st>(cfg_in);
    if(cfg == nullptr) cfg = &default_cfg;

    /* ---- Build working pool from inputs. ---- */
    std::vector<pool_entry> pool;
    pool.reserve(nd->num_inputs);
    for(int32_t i = 0; i < nd->num_inputs; ++i)
    {
        pool.push_back({i, nd->inputs[i].modes, nd->inputs[i].extents});
    }
    int32_t next_id = nd->num_inputs;

    info->path.clear();
    info->path.reserve(static_cast<size_t>(std::max(0, nd->num_inputs - 1)));
    info->flop_count     = 0;
    info->largest_tensor = 0;

    while(pool.size() > 1)
    {
        size_t best_i = 0, best_j = 1;
        pair_eval best = evaluate_pair(pool[0], pool[1]);
        for(size_t i = 0; i < pool.size(); ++i)
            for(size_t j = i + 1; j < pool.size(); ++j)
            {
                auto e = evaluate_pair(pool[i], pool[j]);
                bool better = e.out_size < best.out_size
                              || (e.out_size == best.out_size && e.flops < best.flops);
                if(better) { best = std::move(e); best_i = i; best_j = j; }
            }

        pair_node node;
        node.left  = pool[best_i].id;
        node.right = pool[best_j].id;
        node.out_modes   = best.out_modes;
        node.out_extents = best.out_extents;
        node.flops       = best.flops;
        node.intermediate_bytes = best.out_size * element_size_bytes(nd->data_type);
        info->flop_count    += best.flops;
        if(node.intermediate_bytes > info->largest_tensor)
            info->largest_tensor = node.intermediate_bytes;
        info->path.push_back(node);

        pool_entry merged;
        merged.id = next_id++;
        merged.modes   = std::move(best.out_modes);
        merged.extents = std::move(best.out_extents);
        /* Drop the higher index first to keep best_i valid. */
        pool.erase(pool.begin() + best_j);
        pool.erase(pool.begin() + best_i);
        pool.push_back(std::move(merged));
    }

    /* The final pool entry is the network output; trust the descriptor's
     * mode ordering (path-finder used sorted modes for stable testing,
     * but the public output shape follows descriptor order). */
    info->num_output_modes = nd->output.num_modes();
    info->output_modes     = nd->output.modes;
    info->output_extents   = nd->output.extents;
    info->num_inputs       = nd->num_inputs;
    info->data_type        = nd->data_type;
    info->compute_type     = nd->compute_type;

    /* ---- Slicing pass ---- */
    info->sliced_modes.clear();
    info->sliced_extents.clear();
    info->num_slices = 1;

    if(cfg->slicer_disable_slicing == 0 && workspace_size_constraint > 0
       && info->largest_tensor > workspace_size_constraint)
    {
        /* Pick contracted modes from the largest pair, slice in extent
         * order until the largest intermediate fits. We slice over
         * shared modes between the inputs of the largest pair node. */
        size_t   biggest_idx = 0;
        size_t   biggest_sz  = 0;
        for(size_t i = 0; i < info->path.size(); ++i)
            if(info->path[i].intermediate_bytes > biggest_sz)
            {
                biggest_sz = info->path[i].intermediate_bytes;
                biggest_idx = i;
            }
        const auto& biggest = info->path[biggest_idx];
        (void)biggest; /* used implicitly via pool snapshot below */
        /* For v0.1.0 we slice on the *first* mode of the biggest
         * intermediate's output (heuristic; usually a shared mode). */
        size_t need = info->largest_tensor;
        size_t allowed = static_cast<size_t>(workspace_size_constraint);
        for(size_t k = 0; k < info->path[biggest_idx].out_modes.size() && need > allowed; ++k)
        {
            info->sliced_modes.push_back(info->path[biggest_idx].out_modes[k]);
            info->sliced_extents.push_back(info->path[biggest_idx].out_extents[k]);
            need /= std::max<roctensornet_index_t>(1, info->path[biggest_idx].out_extents[k]);
            info->num_slices *= info->path[biggest_idx].out_extents[k];
        }
    }

    return ROCTENSORNET_STATUS_SUCCESS;
}

} // extern "C"
