/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Internal struct definitions for opaque rocDENSITYMAT handles.
 * ************************************************************************ */

#ifndef ROCDENSITYMAT_DESCRIPTORS_HPP
#define ROCDENSITYMAT_DESCRIPTORS_HPP

#include "rocdensitymat_internal.hpp"

#include <numeric>

struct _rocdensitymat_handle : public rocdensitymat::handle_impl
{
};

struct _rocdensitymat_workspace_descriptor
{
    void*  device_scratch_ptr   = nullptr;
    size_t device_scratch_bytes = 0;

    void*  host_scratch_ptr     = nullptr;
    size_t host_scratch_bytes   = 0;

    /*! \brief Required scratch size as last reported by a prepare() call. */
    size_t required_device_scratch_bytes = 0;
};

struct _rocdensitymat_state
{
    rocdensitymat_state_purity   purity      = ROCDENSITYMAT_STATE_PURITY_PURE;
    rocdensitymat_data_type      data_type   = ROCDENSITYMAT_C_64F;
    int32_t                      num_modes   = 0;
    std::vector<int64_t>         space_shape;
    int64_t                      batch_size  = 1;

    /*! \brief Total element count in a single component
     *  (\p prod(space_shape) for pure, \p prod(space_shape)^2 for mixed). */
    int64_t                      component_elems = 0;

    /*! \brief Storage size of a single component in bytes. */
    size_t                       component_bytes = 0;

    /*! \brief External device buffer attached by the caller; rocDENSITYMAT
     *  never owns or frees this pointer. */
    void*                        component_buffer = nullptr;
    size_t                       component_buffer_bytes = 0;

    int64_t hilbert_dim() const
    {
        int64_t prod = 1;
        for(auto d : space_shape) prod *= d;
        return prod;
    }
};

struct _rocdensitymat_elementary_operator
{
    rocdensitymat_elementary_kind   kind        = ROCDENSITYMAT_ELEMENTARY_IDENTITY;
    rocdensitymat_data_type         data_type   = ROCDENSITYMAT_C_64F;
    int32_t                         num_modes   = 0;
    std::vector<int64_t>            mode_extents;

    /*! \brief Host-resident copy of the data. Pauli / Identity ops keep a
     *  cached 2x2 representation so the apply path does not need to special
     *  case them; dense and diagonal ops keep the full caller-supplied
     *  matrix here. */
    std::vector<std::complex<double>> host_data;

    /*! \brief Element count of host_data. Dense: prod(extents)^2.
     *  Diagonal: prod(extents). */
    int64_t                         elem_count  = 0;

    rocdensitymat_scalar_callback_t tdep_callback = nullptr;

    /*! \brief Lazily-created device copy of host_data (uploaded the first
     *  time the operator is used in an apply); freed in the destructor. */
    void*                           device_data       = nullptr;
    size_t                          device_data_bytes = 0;
    hipStream_t                     device_data_stream = nullptr;
};

struct _rocdensitymat_operator_term
{
    int32_t                                num_space_modes = 0;
    std::vector<int64_t>                   space_shape;

    /*! \brief One \p product per call to \p OperatorTermAppendElementaryProduct. */
    struct product
    {
        std::vector<rocdensitymat_elementary_operator> operators;
        std::vector<int32_t>                           state_modes;
        std::vector<int32_t>                           mode_action_duality;
        rocdensitymat_complex_double                   coefficient;
        rocdensitymat_scalar_callback_t                tdep_callback = nullptr;
    };
    std::vector<product>                   products;
};

struct _rocdensitymat_operator
{
    int32_t                                num_space_modes = 0;
    std::vector<int64_t>                   space_shape;

    struct term_entry
    {
        rocdensitymat_operator_term     term            = nullptr;
        int32_t                         duality_offset  = 0;
        rocdensitymat_complex_double    coefficient;
        rocdensitymat_scalar_callback_t tdep_callback   = nullptr;
    };
    std::vector<term_entry>                terms;

    /*! \brief Returns true if any term is a Lindblad-collapse (duality_offset != 0). */
    bool has_collapse_term() const
    {
        for(auto& t : terms)
            if(t.duality_offset != 0) return true;
        return false;
    }
};

struct _rocdensitymat_master_equation_solver
{
    rocdensitymat_solver_kind kind         = ROCDENSITYMAT_SOLVER_RK4;
    rocdensitymat_operator    liouvillian  = nullptr;

    /*! \brief Required scratch size for one RK4 step over the bound state. */
    size_t                    step_scratch_bytes = 0;
    bool                      prepared           = false;
    rocdensitymat_compute_type compute_type      = ROCDENSITYMAT_COMPUTE_DEFAULT;
};

#endif /* ROCDENSITYMAT_DESCRIPTORS_HPP */
