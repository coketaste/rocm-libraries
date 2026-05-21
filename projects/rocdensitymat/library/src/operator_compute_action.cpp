/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
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

/*! \brief Multiply a state buffer in-place by ``-i`` (used to convert
 *         ``H |psi>`` into ``-i H |psi>`` so the ODE driver receives
 *         the Liouvillian rate of change.) */
template <typename Cmplx>
static rocdensitymat_status apply_neg_i_inplace(handle_impl* h,
                                                Cmplx* y, int64_t n)
{
    int     bs = default_threads_per_block;
    int64_t gx = ceil_div<int64_t>(n, bs);
    Cmplx neg_i; neg_i.x = 0; neg_i.y = -1;
    hipLaunchKernelGGL(scale_kernel<Cmplx>,
                       dim3(static_cast<unsigned int>(gx)),
                       dim3(bs), 0, h->stream, n, neg_i, y);
    return (hipGetLastError() == hipSuccess)
               ? ROCDENSITYMAT_STATUS_SUCCESS
               : ROCDENSITYMAT_STATUS_EXECUTION_FAILED;
}

/*! \brief Compute the per-state scratch requirement of compute_action and
 *         imprint it on the supplied workspace descriptor. */
static size_t compute_action_scratch_bytes(rocdensitymat_state state)
{
    if(state->purity == ROCDENSITYMAT_STATE_PURITY_PURE)
        return apply_op_pure_scratch_bytes_per_component(state);
    return apply_op_mixed_scratch_bytes(state);
}

} // namespace rocdensitymat

extern "C" {

rocdensitymat_status rocdensitymat_operator_prepare_action(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    rocdensitymat_state state_in,
    rocdensitymat_state state_out,
    rocdensitymat_compute_type compute_type,
    size_t workspace_size_limit,
    rocdensitymat_workspace_descriptor workspace)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(op);
    ROCDENSITYMAT_CHECK_PTR(state_in);
    ROCDENSITYMAT_CHECK_PTR(state_out);
    ROCDENSITYMAT_CHECK_PTR(workspace);
    if(state_in->purity == ROCDENSITYMAT_STATE_PURITY_MPS
       || state_out->purity == ROCDENSITYMAT_STATE_PURITY_MPS)
        return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;
    if(state_in->purity != state_out->purity
       || state_in->data_type != state_out->data_type
       || state_in->num_modes != state_out->num_modes)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    if(compute_type != ROCDENSITYMAT_COMPUTE_DEFAULT
       && compute_type != ROCDENSITYMAT_COMPUTE_64F
       && compute_type != ROCDENSITYMAT_COMPUTE_32F)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    if(op->has_collapse_term())
        return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;

    size_t need = rocdensitymat::compute_action_scratch_bytes(state_in);
    if(workspace_size_limit != 0 && need > workspace_size_limit)
        return ROCDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE;
    workspace->required_device_scratch_bytes = need;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_operator_compute_action(
    rocdensitymat_handle handle,
    rocdensitymat_operator op,
    double t,
    int32_t num_params,
    const double* params,
    rocdensitymat_state state_in,
    rocdensitymat_state state_out,
    rocdensitymat_workspace_descriptor workspace)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(op);
    ROCDENSITYMAT_CHECK_PTR(state_in);
    ROCDENSITYMAT_CHECK_PTR(state_out);
    ROCDENSITYMAT_CHECK_PTR(workspace);
    if(state_in->component_buffer == nullptr || state_out->component_buffer == nullptr)
        return ROCDENSITYMAT_STATUS_NOT_INITIALIZED;
    if(op->has_collapse_term())
        return ROCDENSITYMAT_STATUS_NOT_SUPPORTED;
    if(state_in->purity != state_out->purity
       || state_in->data_type != state_out->data_type)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;

    if(workspace->device_scratch_bytes < workspace->required_device_scratch_bytes
       || workspace->device_scratch_ptr == nullptr)
        return ROCDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE;

    auto* hi = static_cast<rocdensitymat::handle_impl*>(handle);

    if(state_in->purity == ROCDENSITYMAT_STATE_PURITY_PURE)
    {
        int64_t n = state_in->component_elems;
        if(state_in->data_type == ROCDENSITYMAT_C_64F)
        {
            auto* sa  = static_cast<hipDoubleComplex*>(workspace->device_scratch_ptr);
            auto* sb  = sa + n;
            auto  rc  = rocdensitymat::apply_op_pure_into<hipDoubleComplex>(
                hi, op, t, num_params, params,
                static_cast<const hipDoubleComplex*>(state_in->component_buffer),
                static_cast<hipDoubleComplex*>(state_out->component_buffer),
                /*accumulate*/false, sa, sb, n);
            if(rc != ROCDENSITYMAT_STATUS_SUCCESS) return rc;
            return rocdensitymat::apply_neg_i_inplace<hipDoubleComplex>(
                hi, static_cast<hipDoubleComplex*>(state_out->component_buffer), n);
        }
        auto* sa = static_cast<hipFloatComplex*>(workspace->device_scratch_ptr);
        auto* sb = sa + n;
        auto rc  = rocdensitymat::apply_op_pure_into<hipFloatComplex>(
            hi, op, t, num_params, params,
            static_cast<const hipFloatComplex*>(state_in->component_buffer),
            static_cast<hipFloatComplex*>(state_out->component_buffer),
            /*accumulate*/false, sa, sb, n);
        if(rc != ROCDENSITYMAT_STATUS_SUCCESS) return rc;
        return rocdensitymat::apply_neg_i_inplace<hipFloatComplex>(
            hi, static_cast<hipFloatComplex*>(state_out->component_buffer), n);
    }

    int64_t D = state_in->hilbert_dim();
    if(state_in->data_type == ROCDENSITYMAT_C_64F)
    {
        auto* base   = static_cast<hipDoubleComplex*>(workspace->device_scratch_ptr);
        auto* buf_a  = base;
        auto* buf_b  = buf_a + D;
        auto* rho_T  = buf_b + D;
        auto* col_ac = rho_T + D * D;
        return rocdensitymat::apply_op_mixed_into<hipDoubleComplex>(
            hi, op, t, num_params, params,
            static_cast<const hipDoubleComplex*>(state_in->component_buffer),
            static_cast<hipDoubleComplex*>(state_out->component_buffer),
            buf_a, buf_b, rho_T, col_ac, D);
    }
    auto* base   = static_cast<hipFloatComplex*>(workspace->device_scratch_ptr);
    auto* buf_a  = base;
    auto* buf_b  = buf_a + D;
    auto* rho_T  = buf_b + D;
    auto* col_ac = rho_T + D * D;
    return rocdensitymat::apply_op_mixed_into<hipFloatComplex>(
        hi, op, t, num_params, params,
        static_cast<const hipFloatComplex*>(state_in->component_buffer),
        static_cast<hipFloatComplex*>(state_out->component_buffer),
        buf_a, buf_b, rho_T, col_ac, D);
}

} // extern "C"
