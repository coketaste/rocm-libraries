/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Test fixtures for the rocTENSORNET test suite. Provides utilities to
 * build small example networks (matrix-matrix, chain-of-three, etc).
 * ************************************************************************ */

#pragma once

#include "test_helpers.hpp"

#include <vector>

struct MatMulNetwork
{
    roctensornet_handle              h    = nullptr;
    roctensornet_network_descriptor  nd   = nullptr;
    roctensornet_contraction_optimizer_config cfg = nullptr;
    roctensornet_contraction_optimizer_info   info = nullptr;
    roctensornet_workspace_descriptor          ws  = nullptr;
    roctensornet_contraction_plan              plan = nullptr;

    int32_t mA = 2, kAB = 2, nB = 2;
    /* extents indexed by mode IDs 0, 1, 2 */
    int32_t modes_a[2] = {0, 1};
    int32_t modes_b[2] = {1, 2};
    int32_t modes_c[2] = {0, 2};

    void setup_dims(int32_t M, int32_t K, int32_t N)
    {
        mA = M; kAB = K; nB = N;
    }

    ~MatMulNetwork()
    {
        if(plan) roctensornet_destroy_contraction_plan(plan);
        if(ws)   roctensornet_destroy_workspace_descriptor(ws);
        if(info) roctensornet_destroy_contraction_optimizer_info(info);
        if(cfg)  roctensornet_destroy_contraction_optimizer_config(cfg);
        if(nd)   roctensornet_destroy_network_descriptor(nd);
        if(h)    roctensornet_destroy(h);
    }

    void build()
    {
        ROC_TN_ASSERT_OK(roctensornet_create(&h));
        int32_t num_modes_in[2] = {2, 2};
        roctensornet_index_t ea[2] = {mA, kAB};
        roctensornet_index_t eb[2] = {kAB, nB};
        roctensornet_index_t ec[2] = {mA, nB};
        const int32_t* modes_in[2]                = {modes_a, modes_b};
        const roctensornet_index_t* extents_in[2] = {ea, eb};
        ROC_TN_ASSERT_OK(roctensornet_create_network_descriptor(
            h, 2, num_modes_in, extents_in, nullptr, modes_in, nullptr,
            2, ec, nullptr, modes_c, 256,
            ROCTENSORNET_R_64F, ROCTENSORNET_COMPUTE_64F, &nd));
        ROC_TN_ASSERT_OK(roctensornet_create_contraction_optimizer_config(h, &cfg));
        ROC_TN_ASSERT_OK(roctensornet_create_contraction_optimizer_info(h, nd, &info));
        ROC_TN_ASSERT_OK(roctensornet_contraction_optimize(h, nd, cfg, 1ull<<30, info));
        ROC_TN_ASSERT_OK(roctensornet_create_workspace_descriptor(h, &ws));
        ROC_TN_ASSERT_OK(roctensornet_workspace_compute_contraction_sizes(h, nd, info, ws));
        ROC_TN_ASSERT_OK(roctensornet_create_contraction_plan(h, nd, info, ws, &plan));
    }
};
