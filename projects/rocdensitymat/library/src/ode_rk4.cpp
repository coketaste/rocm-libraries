/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Fixed-step RK4 master-equation stepper.
 *
 * Integrates dy/dt = L[y] with L[y] = -i [H, y] (or -i H |y> for pure
 * states). Each step requires four evaluations of L[.] across separate
 * stage buffers (k1..k4) plus one combined-update buffer y_tmp, all of
 * the same shape as y. The total scratch demand is therefore reported
 * to the workspace as 5 * component_bytes plus the inner-apply scratch.
 * ************************************************************************ */

#include "rocdensitymat_descriptors.hpp"
#include "rocdensitymat_kernels.hpp"

#include <complex>

namespace rocdensitymat
{

template <typename Cmplx>
rocdensitymat_status
apply_op_pure_into(handle_impl* h,
                   rocdensitymat_operator op,
                   double t,
                   int32_t num_params,
                   const double* params,
                   const Cmplx* x,
                   Cmplx* y,
                   bool accumulate,
                   Cmplx* buf_a,
                   Cmplx* buf_b,
                   int64_t n_elems);

template <typename Cmplx>
rocdensitymat_status
apply_op_mixed_into(handle_impl* h,
                    rocdensitymat_operator op,
                    double t,
                    int32_t num_params,
                    const double* params,
                    const Cmplx* rho_in,
                    Cmplx* rho_out,
                    Cmplx* buf_a,
                    Cmplx* buf_b,
                    Cmplx* rho_T,
                    Cmplx* col_acc,
                    int64_t D);

size_t apply_op_pure_scratch_bytes_per_component(rocdensitymat_state state);
size_t apply_op_mixed_scratch_bytes(rocdensitymat_state state);

namespace
{

template <typename Cmplx>
rocdensitymat_status liouvillian_eval(handle_impl*       h,
                                      rocdensitymat_operator op,
                                      rocdensitymat_state state,
                                      double             t,
                                      int32_t            num_params,
                                      const double*      params,
                                      const Cmplx*       in,
                                      Cmplx*             out,
                                      Cmplx*             scratch_base,
                                      int64_t            n_elems)
{
    if(state->purity == ROCDENSITYMAT_STATE_PURITY_PURE)
    {
        Cmplx* sa = scratch_base;
        Cmplx* sb = sa + n_elems;
        ROCDENSITYMAT_RC_CHECK(apply_op_pure_into<Cmplx>(
            h, op, t, num_params, params, in, out, /*accumulate*/false,
            sa, sb, n_elems));
        // out *= -i
        int     bs = default_threads_per_block;
        int64_t gx = ceil_div<int64_t>(n_elems, bs);
        Cmplx neg_i; neg_i.x = 0; neg_i.y = -1;
        hipLaunchKernelGGL(scale_kernel<Cmplx>,
                           dim3(static_cast<unsigned int>(gx)),
                           dim3(bs), 0, h->stream, n_elems, neg_i, out);
        return (hipGetLastError() == hipSuccess)
                   ? ROCDENSITYMAT_STATUS_SUCCESS
                   : ROCDENSITYMAT_STATUS_EXECUTION_FAILED;
    }
    int64_t D       = state->hilbert_dim();
    Cmplx*  buf_a   = scratch_base;
    Cmplx*  buf_b   = buf_a + D;
    Cmplx*  rho_T   = buf_b + D;
    Cmplx*  col_ac  = rho_T + D * D;
    return apply_op_mixed_into<Cmplx>(h, op, t, num_params, params,
                                      in, out, buf_a, buf_b, rho_T, col_ac, D);
}

template <typename Cmplx>
__global__ void rk4_combine_kernel(int64_t n,
                                   double dt,
                                   const Cmplx* __restrict__ y0,
                                   const Cmplx* __restrict__ k1,
                                   const Cmplx* __restrict__ k2,
                                   const Cmplx* __restrict__ k3,
                                   const Cmplx* __restrict__ k4,
                                   Cmplx*       __restrict__ y_out)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(tid >= n) return;
    using R = decltype(Cmplx{}.x);
    R d = static_cast<R>(dt) * static_cast<R>(1.0 / 6.0);
    Cmplx a = k1[tid], b = k2[tid], c = k3[tid], e = k4[tid], y = y0[tid];
    y_out[tid].x = y.x + d * (a.x + 2 * b.x + 2 * c.x + e.x);
    y_out[tid].y = y.y + d * (a.y + 2 * b.y + 2 * c.y + e.y);
}

template <typename Cmplx>
__global__ void rk4_stage_axpy_kernel(int64_t n,
                                      double  dt_scale,
                                      const Cmplx* __restrict__ y0,
                                      const Cmplx* __restrict__ k,
                                      Cmplx*       __restrict__ y_tmp)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(tid >= n) return;
    using R = decltype(Cmplx{}.x);
    R d = static_cast<R>(dt_scale);
    Cmplx kk = k[tid], yy = y0[tid];
    y_tmp[tid].x = yy.x + d * kk.x;
    y_tmp[tid].y = yy.y + d * kk.y;
}

template <typename Cmplx>
rocdensitymat_status rk4_step_typed(handle_impl* h,
                                    rocdensitymat_operator op,
                                    rocdensitymat_state state,
                                    double t0, double dt,
                                    int32_t num_params,
                                    const double* params,
                                    Cmplx* y_state,
                                    Cmplx* k1, Cmplx* k2, Cmplx* k3, Cmplx* k4,
                                    Cmplx* y_tmp,
                                    Cmplx* inner_scratch,
                                    int64_t n_elems)
{
    int     bs = default_threads_per_block;
    int64_t gx = ceil_div<int64_t>(n_elems, bs);

    // k1 = L[t0, y]
    ROCDENSITYMAT_RC_CHECK(liouvillian_eval<Cmplx>(
        h, op, state, t0, num_params, params, y_state, k1, inner_scratch, n_elems));

    // y_tmp = y + (dt/2) * k1
    hipLaunchKernelGGL(rk4_stage_axpy_kernel<Cmplx>,
                       dim3(static_cast<unsigned int>(gx)),
                       dim3(bs), 0, h->stream,
                       n_elems, dt * 0.5, y_state, k1, y_tmp);
    if(hipGetLastError() != hipSuccess)
        return ROCDENSITYMAT_STATUS_EXECUTION_FAILED;

    // k2 = L[t0 + dt/2, y_tmp]
    ROCDENSITYMAT_RC_CHECK(liouvillian_eval<Cmplx>(
        h, op, state, t0 + 0.5 * dt, num_params, params, y_tmp, k2, inner_scratch, n_elems));

    // y_tmp = y + (dt/2) * k2
    hipLaunchKernelGGL(rk4_stage_axpy_kernel<Cmplx>,
                       dim3(static_cast<unsigned int>(gx)),
                       dim3(bs), 0, h->stream,
                       n_elems, dt * 0.5, y_state, k2, y_tmp);
    if(hipGetLastError() != hipSuccess)
        return ROCDENSITYMAT_STATUS_EXECUTION_FAILED;

    // k3 = L[t0 + dt/2, y_tmp]
    ROCDENSITYMAT_RC_CHECK(liouvillian_eval<Cmplx>(
        h, op, state, t0 + 0.5 * dt, num_params, params, y_tmp, k3, inner_scratch, n_elems));

    // y_tmp = y + dt * k3
    hipLaunchKernelGGL(rk4_stage_axpy_kernel<Cmplx>,
                       dim3(static_cast<unsigned int>(gx)),
                       dim3(bs), 0, h->stream,
                       n_elems, dt, y_state, k3, y_tmp);
    if(hipGetLastError() != hipSuccess)
        return ROCDENSITYMAT_STATUS_EXECUTION_FAILED;

    // k4 = L[t0 + dt, y_tmp]
    ROCDENSITYMAT_RC_CHECK(liouvillian_eval<Cmplx>(
        h, op, state, t0 + dt, num_params, params, y_tmp, k4, inner_scratch, n_elems));

    // y <- y + (dt/6) (k1 + 2 k2 + 2 k3 + k4)
    hipLaunchKernelGGL(rk4_combine_kernel<Cmplx>,
                       dim3(static_cast<unsigned int>(gx)),
                       dim3(bs), 0, h->stream,
                       n_elems, dt, y_state, k1, k2, k3, k4, y_state);
    if(hipGetLastError() != hipSuccess)
        return ROCDENSITYMAT_STATUS_EXECUTION_FAILED;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

size_t solver_inner_scratch_bytes(rocdensitymat_state state)
{
    if(state->purity == ROCDENSITYMAT_STATE_PURITY_PURE)
        return apply_op_pure_scratch_bytes_per_component(state);
    return apply_op_mixed_scratch_bytes(state);
}

size_t solver_total_scratch_bytes(rocdensitymat_state state)
{
    // 5 stage buffers (k1..k4 + y_tmp) of component_bytes each, plus inner.
    return 5 * state->component_bytes + solver_inner_scratch_bytes(state);
}

} // namespace
} // namespace rocdensitymat

extern "C" {

rocdensitymat_status rocdensitymat_create_master_equation_solver(
    rocdensitymat_handle handle,
    rocdensitymat_operator liouvillian,
    rocdensitymat_solver_kind kind,
    rocdensitymat_master_equation_solver* solver)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(liouvillian);
    ROCDENSITYMAT_CHECK_PTR(solver);
    if(kind != ROCDENSITYMAT_SOLVER_RK4)
        return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;
    auto* s = new(std::nothrow) _rocdensitymat_master_equation_solver();
    if(s == nullptr) return ROCDENSITYMAT_STATUS_ALLOC_FAILED;
    s->kind        = kind;
    s->liouvillian = liouvillian;
    *solver        = s;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_destroy_master_equation_solver(
    rocdensitymat_master_equation_solver solver)
{
    if(solver == nullptr) return ROCDENSITYMAT_STATUS_SUCCESS;
    delete solver;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_master_equation_solver_prepare(
    rocdensitymat_handle handle,
    rocdensitymat_master_equation_solver solver,
    rocdensitymat_state state,
    rocdensitymat_compute_type compute_type,
    size_t workspace_size_limit,
    rocdensitymat_workspace_descriptor workspace)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(solver);
    ROCDENSITYMAT_CHECK_PTR(state);
    ROCDENSITYMAT_CHECK_PTR(workspace);
    if(solver->liouvillian == nullptr) return ROCDENSITYMAT_STATUS_NOT_INITIALIZED;
    if(solver->liouvillian->has_collapse_term())
        return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;
    if(state->purity == ROCDENSITYMAT_STATE_PURITY_MPS)
        return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;
    if(compute_type != ROCDENSITYMAT_COMPUTE_DEFAULT
       && compute_type != ROCDENSITYMAT_COMPUTE_64F
       && compute_type != ROCDENSITYMAT_COMPUTE_32F)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;

    size_t need = rocdensitymat::solver_total_scratch_bytes(state);
    if(workspace_size_limit != 0 && need > workspace_size_limit)
        return ROCDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE;
    workspace->required_device_scratch_bytes = need;
    solver->step_scratch_bytes               = need;
    solver->compute_type                     = compute_type;
    solver->prepared                         = true;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_master_equation_step(
    rocdensitymat_handle handle,
    rocdensitymat_master_equation_solver solver,
    double t0,
    double dt,
    int32_t num_params,
    const double* params,
    rocdensitymat_state state,
    rocdensitymat_workspace_descriptor workspace)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(solver);
    ROCDENSITYMAT_CHECK_PTR(state);
    ROCDENSITYMAT_CHECK_PTR(workspace);
    if(!solver->prepared) return ROCDENSITYMAT_STATUS_NOT_INITIALIZED;
    if(state->component_buffer == nullptr) return ROCDENSITYMAT_STATUS_NOT_INITIALIZED;
    if(workspace->device_scratch_bytes < solver->step_scratch_bytes
       || workspace->device_scratch_ptr == nullptr)
        return ROCDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE;

    auto* hi = static_cast<rocdensitymat::handle_impl*>(handle);
    int64_t  n  = state->component_elems;

    if(state->data_type == ROCDENSITYMAT_C_64F)
    {
        auto* base   = static_cast<hipDoubleComplex*>(workspace->device_scratch_ptr);
        auto* k1     = base;
        auto* k2     = k1 + n;
        auto* k3     = k2 + n;
        auto* k4     = k3 + n;
        auto* y_tmp  = k4 + n;
        auto* inner  = y_tmp + n;
        return rocdensitymat::rk4_step_typed<hipDoubleComplex>(
            hi, solver->liouvillian, state, t0, dt,
            num_params, params,
            static_cast<hipDoubleComplex*>(state->component_buffer),
            k1, k2, k3, k4, y_tmp, inner, n);
    }
    auto* base   = static_cast<hipFloatComplex*>(workspace->device_scratch_ptr);
    auto* k1     = base;
    auto* k2     = k1 + n;
    auto* k3     = k2 + n;
    auto* k4     = k3 + n;
    auto* y_tmp  = k4 + n;
    auto* inner  = y_tmp + n;
    return rocdensitymat::rk4_step_typed<hipFloatComplex>(
        hi, solver->liouvillian, state, t0, dt,
        num_params, params,
        static_cast<hipFloatComplex*>(state->component_buffer),
        k1, k2, k3, k4, y_tmp, inner, n);
}

rocdensitymat_status rocdensitymat_master_equation_step_n(
    rocdensitymat_handle handle,
    rocdensitymat_master_equation_solver solver,
    double t0,
    double dt,
    int64_t num_steps,
    int32_t num_params,
    const double* params,
    rocdensitymat_state state,
    rocdensitymat_workspace_descriptor workspace)
{
    if(num_steps < 0) return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    double t = t0;
    for(int64_t i = 0; i < num_steps; ++i)
    {
        auto rc = rocdensitymat_master_equation_step(
            handle, solver, t, dt, num_params, params, state, workspace);
        if(rc != ROCDENSITYMAT_STATUS_SUCCESS) return rc;
        t += dt;
    }
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

} // extern "C"
