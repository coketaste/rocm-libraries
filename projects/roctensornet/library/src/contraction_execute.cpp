/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Contraction executor: runs the binary-pair sequence captured in a
 * `contraction_plan_st`. Each pair is dispatched through a generic
 * loop-nest kernel (see `roctensornet_kernels.hpp`).
 *
 * Memory: intermediates and per-pair stride arrays are sourced from a
 * `dev_arena` rooted at the workspace descriptor's device SCRATCH
 * slot. If that slot has a user-supplied buffer the arena consumes it
 * directly; otherwise the arena routes through `dev_alloc` which in
 * turn honors the handle's `roctensornet_device_mem_handler_t`.
 *
 * Slicing: when the optimizer info marks one or more modes for
 * slicing, this executor produces a single slice (the slice_id passed
 * in for `_contraction`) or the full set (for `_contract_slices`).
 * `_contract_slices` accumulates output slices.
 * ************************************************************************ */

#include "roctensornet_descriptors.hpp"
#include "roctensornet_kernels.hpp"

#include <algorithm>
#include <unordered_map>

using namespace roctensornet;

namespace
{

/* ---- Strides (element units) for a tensor descriptor in generalized
 * column-major order, used when the user-supplied strides are 0/null. */
inline std::vector<roctensornet_index_t>
default_strides(const std::vector<roctensornet_index_t>& extents)
{
    std::vector<roctensornet_index_t> s;
    generalized_column_major_strides(extents, s);
    return s;
}

/* ---- Compute output strides for a pair_op intermediate. */
inline std::vector<roctensornet_index_t>
intermediate_strides(const contraction_plan_st::pair_op& p)
{
    return default_strides(p.out_extents);
}

/* ---- For a pair, evaluate one binary contraction:
 *      c = sum_k a[..., k] * b[..., k]
 * Generic over element type. */
template <typename T>
roctensornet_status
run_pair(const T*                                  a,
         const T*                                  b,
         T*                                        c,
         const std::vector<int32_t>&               a_modes,
         const std::vector<roctensornet_index_t>&  a_extents,
         const std::vector<roctensornet_index_t>&  a_strides,
         const std::vector<int32_t>&               b_modes,
         const std::vector<roctensornet_index_t>&  b_extents,
         const std::vector<roctensornet_index_t>&  b_strides,
         const std::vector<int32_t>&               out_modes,
         const std::vector<roctensornet_index_t>&  out_extents,
         const std::vector<roctensornet_index_t>&  out_strides,
         dev_arena&                                arena,
         hipStream_t                               stream,
         int                                       accumulate)
{
    /* Build the per-output-axis stride lookups. */
    auto find_stride = [&](const std::vector<int32_t>& m, const std::vector<roctensornet_index_t>& s, int32_t mode) -> roctensornet_index_t
    {
        for(size_t i = 0; i < m.size(); ++i) if(m[i] == mode) return s[i];
        return 0;
    };
    std::vector<roctensornet_index_t> out_a(out_modes.size()), out_b(out_modes.size());
    for(size_t i = 0; i < out_modes.size(); ++i)
    {
        out_a[i] = find_stride(a_modes, a_strides, out_modes[i]);
        out_b[i] = find_stride(b_modes, b_strides, out_modes[i]);
    }

    /* Contracted modes: in both A and B, not in output. */
    std::vector<int32_t>              k_modes;
    std::vector<roctensornet_index_t> k_extents, k_a, k_b;
    for(size_t i = 0; i < a_modes.size(); ++i)
    {
        int32_t m = a_modes[i];
        bool inB = std::find(b_modes.begin(), b_modes.end(), m) != b_modes.end();
        bool inOut = std::find(out_modes.begin(), out_modes.end(), m) != out_modes.end();
        if(inB && !inOut)
        {
            k_modes.push_back(m);
            k_extents.push_back(a_extents[i]);
            k_a.push_back(a_strides[i]);
            roctensornet_index_t bs = 0;
            for(size_t j = 0; j < b_modes.size(); ++j) if(b_modes[j] == m) { bs = b_strides[j]; break; }
            k_b.push_back(bs);
        }
    }

    /* Copy the small index arrays to device. */
    size_t r_out = out_extents.size();
    size_t r_k   = k_extents.size();
    auto bytes_idx = [](size_t n){ return n * sizeof(roctensornet_index_t); };

    void* d_oe = nullptr; void* d_oa = nullptr; void* d_ob = nullptr; void* d_oc = nullptr;
    void* d_ke = nullptr; void* d_ka = nullptr; void* d_kb = nullptr;
    ROCTENSORNET_RC_CHECK(arena.alloc_and_copy(&d_oe, out_extents.data(), bytes_idx(r_out)));
    ROCTENSORNET_RC_CHECK(arena.alloc_and_copy(&d_oa, out_a.data(),       bytes_idx(r_out)));
    ROCTENSORNET_RC_CHECK(arena.alloc_and_copy(&d_ob, out_b.data(),       bytes_idx(r_out)));
    ROCTENSORNET_RC_CHECK(arena.alloc_and_copy(&d_oc, out_strides.data(), bytes_idx(r_out)));
    ROCTENSORNET_RC_CHECK(arena.alloc_and_copy(&d_ke, k_extents.data(),   bytes_idx(r_k)));
    ROCTENSORNET_RC_CHECK(arena.alloc_and_copy(&d_ka, k_a.data(),         bytes_idx(r_k)));
    ROCTENSORNET_RC_CHECK(arena.alloc_and_copy(&d_kb, k_b.data(),         bytes_idx(r_k)));

    /* Total output element count. */
    size_t total_out = 1;
    for(auto e : out_extents) total_out *= static_cast<size_t>(e);

    hipError_t le = launch_binary_contract_kernel<T>(
        stream, a, b, c,
        static_cast<int32_t>(r_out),
        static_cast<const roctensornet_index_t*>(d_oe),
        static_cast<const roctensornet_index_t*>(d_oa),
        static_cast<const roctensornet_index_t*>(d_ob),
        static_cast<const roctensornet_index_t*>(d_oc),
        static_cast<int32_t>(r_k),
        static_cast<const roctensornet_index_t*>(d_ke),
        static_cast<const roctensornet_index_t*>(d_ka),
        static_cast<const roctensornet_index_t*>(d_kb),
        total_out, accumulate);
    if(le != hipSuccess) return ROCTENSORNET_STATUS_EXECUTION_FAILED;
    return ROCTENSORNET_STATUS_SUCCESS;
}

/* Dispatch a pair by element type. */
roctensornet_status
dispatch_pair(const void*                              a,
              const void*                              b,
              void*                                    c,
              const contraction_plan_st::pair_op&      p,
              const std::vector<roctensornet_index_t>& a_strides,
              const std::vector<roctensornet_index_t>& b_strides,
              const std::vector<roctensornet_index_t>& out_strides,
              roctensornet_data_type                   dt,
              dev_arena&                               arena,
              hipStream_t                              stream,
              int                                      accumulate)
{
    switch(dt)
    {
    case ROCTENSORNET_R_32F:
        return run_pair<float>(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(c),
                               p.left_modes, p.left_extents, a_strides,
                               p.right_modes, p.right_extents, b_strides,
                               p.out_modes, p.out_extents, out_strides,
                               arena, stream, accumulate);
    case ROCTENSORNET_R_64F:
        return run_pair<double>(static_cast<const double*>(a), static_cast<const double*>(b), static_cast<double*>(c),
                                p.left_modes, p.left_extents, a_strides,
                                p.right_modes, p.right_extents, b_strides,
                                p.out_modes, p.out_extents, out_strides,
                                arena, stream, accumulate);
    case ROCTENSORNET_C_32F:
        return run_pair<hipFloatComplex>(static_cast<const hipFloatComplex*>(a),
                                         static_cast<const hipFloatComplex*>(b),
                                         static_cast<hipFloatComplex*>(c),
                                         p.left_modes, p.left_extents, a_strides,
                                         p.right_modes, p.right_extents, b_strides,
                                         p.out_modes, p.out_extents, out_strides,
                                         arena, stream, accumulate);
    case ROCTENSORNET_C_64F:
        return run_pair<hipDoubleComplex>(static_cast<const hipDoubleComplex*>(a),
                                          static_cast<const hipDoubleComplex*>(b),
                                          static_cast<hipDoubleComplex*>(c),
                                          p.left_modes, p.left_extents, a_strides,
                                          p.right_modes, p.right_extents, b_strides,
                                          p.out_modes, p.out_extents, out_strides,
                                          arena, stream, accumulate);
    default:
        return ROCTENSORNET_STATUS_NOT_SUPPORTED;
    }
}

/* Execute the entire plan once into `out_buffer`. Intermediates are
 * sourced from `arena`. `intermediate_ptrs` keys virtual-IDs to either
 * an input pointer or an arena-owned device buffer. */
roctensornet_status
run_plan_once(const contraction_plan_st& plan,
              const void* const*         raw_data_in,
              void*                      out_buffer,
              dev_arena&                 arena,
              hipStream_t                stream,
              int                        accumulate_output)
{
    size_t es = element_size_bytes(plan.network.data_type);
    int32_t n_inputs = plan.network.num_inputs;

    std::unordered_map<int32_t, void*>                       buf_of;
    std::unordered_map<int32_t, std::vector<roctensornet_index_t>> strides_of;
    for(int32_t i = 0; i < n_inputs; ++i)
    {
        buf_of[i]     = const_cast<void*>(raw_data_in[i]);
        strides_of[i] = plan.network.inputs[i].strides;
    }

    /* IDs for virtual intermediates start at n_inputs. */
    int32_t next_id = n_inputs;
    size_t  num_pairs = plan.pairs.size();
    for(size_t k = 0; k < num_pairs; ++k)
    {
        const auto& p = plan.pairs[k];
        bool is_last = (k + 1 == num_pairs);

        void* out_ptr = nullptr;
        std::vector<roctensornet_index_t> out_strides;
        if(is_last)
        {
            out_ptr = out_buffer;
            /* Remap output strides so they're parallel to the pair's
             * out_modes (the order the kernel iterates) but reference
             * the network output buffer's layout. For each pair-out
             * mode, find the same mode in the network output and
             * adopt its stride; the kernel will then index correctly. */
            out_strides.resize(p.out_modes.size());
            for(size_t i = 0; i < p.out_modes.size(); ++i)
            {
                int32_t m  = p.out_modes[i];
                bool found = false;
                for(size_t j = 0; j < plan.network.output.modes.size(); ++j)
                {
                    if(plan.network.output.modes[j] == m)
                    {
                        out_strides[i] = plan.network.output.strides[j];
                        found = true;
                        break;
                    }
                }
                if(!found) return ROCTENSORNET_STATUS_INVALID_VALUE;
            }
        }
        else
        {
            size_t bytes = 1;
            for(auto e : p.out_extents) bytes *= static_cast<size_t>(e);
            bytes *= es;
            ROCTENSORNET_RC_CHECK(arena.alloc(&out_ptr, bytes));
            out_strides = default_strides(p.out_extents);
        }

        const void* a = buf_of[p.left];
        const void* b = buf_of[p.right];

        int acc = (is_last && accumulate_output != 0) ? 1 : 0;
        if(!is_last) acc = 0; /* fresh intermediate */

        auto rc = dispatch_pair(a, b, out_ptr, p,
                                strides_of[p.left], strides_of[p.right],
                                out_strides,
                                plan.network.data_type, arena, stream, acc);
        if(rc != ROCTENSORNET_STATUS_SUCCESS) return rc;

        buf_of[next_id]     = out_ptr;
        strides_of[next_id] = out_strides;
        next_id++;
    }
    return ROCTENSORNET_STATUS_SUCCESS;
}

} // anon

extern "C" {

roctensornet_status
roctensornet_contraction(roctensornet_handle               h,
                         roctensornet_contraction_plan     plan_in,
                         const void* const*                raw_data_in,
                         void*                             raw_data_out,
                         roctensornet_workspace_descriptor workspace,
                         int64_t                           slice_id,
                         hipStream_t                       stream)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* hh   = reinterpret_cast<handle*>(h);
    auto* plan = cast<contraction_plan_st>(plan_in);
    if(plan == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(raw_data_in);
    ROCTENSORNET_CHECK_PTR(raw_data_out);
    (void)slice_id; /* v0.1.0 ignores; slicing is summed in contract_slices */

    hipStream_t s = stream ? stream : hh->stream;
    workspace_descriptor_st* w = cast<workspace_descriptor_st>(workspace);
    void*   ws_buf   = (w ? w->device_scratch.buffer       : nullptr);
    size_t  ws_bytes = (w ? static_cast<size_t>(w->device_scratch.buffer_bytes) : 0);
    dev_arena arena(hh, s, ws_buf, ws_bytes);

    return run_plan_once(*plan, raw_data_in, raw_data_out, arena, s, 0);
}

roctensornet_status
roctensornet_contract_slices(roctensornet_handle               h,
                             roctensornet_contraction_plan     plan_in,
                             const void* const*                raw_data_in,
                             void*                             raw_data_out,
                             int32_t                           accumulate_output,
                             roctensornet_workspace_descriptor workspace,
                             roctensornet_slice_group          slice_group_in,
                             hipStream_t                       stream)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* hh   = reinterpret_cast<handle*>(h);
    auto* plan = cast<contraction_plan_st>(plan_in);
    if(plan == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(raw_data_in);
    ROCTENSORNET_CHECK_PTR(raw_data_out);

    auto* sg = cast<slice_group_st>(slice_group_in);
    hipStream_t s = stream ? stream : hh->stream;
    workspace_descriptor_st* w = cast<workspace_descriptor_st>(workspace);
    void*   ws_buf   = (w ? w->device_scratch.buffer       : nullptr);
    size_t  ws_bytes = (w ? static_cast<size_t>(w->device_scratch.buffer_bytes) : 0);

    int64_t total = (sg && !sg->slice_ids.empty()) ? static_cast<int64_t>(sg->slice_ids.size())
                                                    : 1;
    for(int64_t k = 0; k < total; ++k)
    {
        dev_arena arena(hh, s, ws_buf, ws_bytes);
        int acc = (k == 0) ? accumulate_output : 1;
        auto rc = run_plan_once(*plan, raw_data_in, raw_data_out, arena, s, acc);
        if(rc != ROCTENSORNET_STATUS_SUCCESS) return rc;
    }
    return ROCTENSORNET_STATUS_SUCCESS;
}

} // extern "C"
