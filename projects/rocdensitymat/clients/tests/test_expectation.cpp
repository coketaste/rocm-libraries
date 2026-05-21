/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * <Z> on |0> equals +1; <Z> on |1> equals -1.
 * ************************************************************************ */

#include "test_helpers.hpp"

namespace
{
double measure_z_on_basis(int64_t basis_idx)
{
    rocdensitymat_handle h = nullptr;
    rocdensitymat_create(&h);
    int64_t shape[1] = {2};
    rocdensitymat_elementary_operator z = nullptr;
    rocdensitymat_create_elementary_operator(
        h, 1, shape, ROCDENSITYMAT_ELEMENTARY_PAULI_Z,
        ROCDENSITYMAT_C_64F, nullptr, nullptr, &z);
    rocdensitymat_operator_term t = nullptr;
    rocdensitymat_create_operator_term(h, 1, shape, &t);
    int32_t modes[1] = {0};
    int32_t dual[1]  = {ROCDENSITYMAT_DUALITY_KET};
    rocdensitymat_complex_double one{1,0};
    rocdensitymat_operator_term_append_elementary_product(
        h, t, 1, &z, modes, dual, one, nullptr);
    rocdensitymat_operator op = nullptr;
    rocdensitymat_create_operator(h, 1, shape, &op);
    rocdensitymat_operator_append_term(h, op, t, 0, one, nullptr);

    rocdensitymat_state s = nullptr;
    rocdensitymat_create_state(
        h, ROCDENSITYMAT_STATE_PURITY_PURE, 1, shape, 1,
        ROCDENSITYMAT_C_64F, &s);
    void* d = nullptr;
    hipMalloc(&d, 2 * sizeof(hipDoubleComplex));
    rocdensitymat_state_attach_component_buffer(
        h, s, 0, d, 2 * sizeof(hipDoubleComplex));
    int64_t basis[1] = {basis_idx};
    rocdensitymat_state_initialize_basis(h, s, basis);

    rocdensitymat_workspace_descriptor ws = nullptr;
    rocdensitymat_create_workspace(h, &ws);
    void* d_ws = nullptr;
    hipMalloc(&d_ws, 6 * sizeof(hipDoubleComplex));
    rocdensitymat_workspace_set_memory(
        h, ws, ROCDENSITYMAT_MEMSPACE_DEVICE,
        ROCDENSITYMAT_WORKSPACE_SCRATCH, d_ws, 6 * sizeof(hipDoubleComplex));
    rocdensitymat_complex_double v{0,0};
    rocdensitymat_operator_compute_expectation(
        h, op, 0.0, 0, nullptr, s, ws, &v);
    hipDeviceSynchronize();

    hipFree(d);
    hipFree(d_ws);
    rocdensitymat_destroy_workspace(ws);
    rocdensitymat_destroy_state(s);
    rocdensitymat_destroy_operator(op);
    rocdensitymat_destroy_operator_term(t);
    rocdensitymat_destroy_elementary_operator(z);
    rocdensitymat_destroy(h);
    return v.x;
}
} // namespace

TEST(ExpectationTest, ZOnZeroIsPlusOne)
{
    EXPECT_NEAR(measure_z_on_basis(0), 1.0, 1e-9);
}

TEST(ExpectationTest, ZOnOneIsMinusOne)
{
    EXPECT_NEAR(measure_z_on_basis(1), -1.0, 1e-9);
}
