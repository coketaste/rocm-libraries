/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "rocdensitymat_descriptors.hpp"

#include <complex>

namespace
{

void seed_pauli_data(rocdensitymat_elementary_kind kind,
                     std::vector<std::complex<double>>& host)
{
    using cd = std::complex<double>;
    host.assign(4, cd(0, 0));
    switch(kind)
    {
    case ROCDENSITYMAT_ELEMENTARY_IDENTITY:
        host[0] = cd(1, 0);
        host[3] = cd(1, 0);
        break;
    case ROCDENSITYMAT_ELEMENTARY_PAULI_X:
        host[1] = cd(1, 0);
        host[2] = cd(1, 0);
        break;
    case ROCDENSITYMAT_ELEMENTARY_PAULI_Y:
        host[1] = cd(0, -1);
        host[2] = cd(0,  1);
        break;
    case ROCDENSITYMAT_ELEMENTARY_PAULI_Z:
        host[0] = cd( 1, 0);
        host[3] = cd(-1, 0);
        break;
    default:
        break;
    }
}

} // namespace

extern "C" {

rocdensitymat_status rocdensitymat_create_elementary_operator(
    rocdensitymat_handle handle,
    int32_t num_modes,
    const int64_t* mode_extents,
    rocdensitymat_elementary_kind kind,
    rocdensitymat_data_type data_type,
    const void* data,
    rocdensitymat_scalar_callback_t tdep_callback,
    rocdensitymat_elementary_operator* op)
{
    ROCDENSITYMAT_CHECK_HANDLE(handle);
    ROCDENSITYMAT_CHECK_PTR(op);
    if(num_modes <= 0 || mode_extents == nullptr)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    if(data_type != ROCDENSITYMAT_C_64F && data_type != ROCDENSITYMAT_C_32F)
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;

    int64_t local_dim = 1;
    for(int32_t i = 0; i < num_modes; ++i)
    {
        if(mode_extents[i] <= 0) return ROCDENSITYMAT_STATUS_INVALID_VALUE;
        local_dim *= mode_extents[i];
    }
    if(local_dim <= 0) return ROCDENSITYMAT_STATUS_INVALID_VALUE;

    if((kind == ROCDENSITYMAT_ELEMENTARY_PAULI_X
        || kind == ROCDENSITYMAT_ELEMENTARY_PAULI_Y
        || kind == ROCDENSITYMAT_ELEMENTARY_PAULI_Z)
       && (num_modes != 1 || mode_extents[0] != 2))
    {
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    }

    auto* e = new(std::nothrow) _rocdensitymat_elementary_operator();
    if(e == nullptr) return ROCDENSITYMAT_STATUS_ALLOC_FAILED;

    e->kind          = kind;
    e->data_type     = data_type;
    e->num_modes     = num_modes;
    e->mode_extents.assign(mode_extents, mode_extents + num_modes);
    e->tdep_callback = tdep_callback;

    using cd = std::complex<double>;

    switch(kind)
    {
    case ROCDENSITYMAT_ELEMENTARY_IDENTITY:
    case ROCDENSITYMAT_ELEMENTARY_PAULI_X:
    case ROCDENSITYMAT_ELEMENTARY_PAULI_Y:
    case ROCDENSITYMAT_ELEMENTARY_PAULI_Z:
        seed_pauli_data(kind, e->host_data);
        e->elem_count = 4;
        break;
    case ROCDENSITYMAT_ELEMENTARY_DENSE:
    {
        int64_t total = local_dim * local_dim;
        e->host_data.assign(static_cast<size_t>(total), cd(0, 0));
        if(data == nullptr)
        {
            delete e;
            return ROCDENSITYMAT_STATUS_INVALID_VALUE;
        }
        if(data_type == ROCDENSITYMAT_C_64F)
        {
            const auto* src = reinterpret_cast<const cd*>(data);
            for(int64_t i = 0; i < total; ++i) e->host_data[i] = src[i];
        }
        else
        {
            const auto* src = reinterpret_cast<const std::complex<float>*>(data);
            for(int64_t i = 0; i < total; ++i)
                e->host_data[i] = cd(src[i].real(), src[i].imag());
        }
        e->elem_count = total;
        break;
    }
    case ROCDENSITYMAT_ELEMENTARY_DIAGONAL:
    {
        int64_t total = local_dim;
        e->host_data.assign(static_cast<size_t>(total), cd(0, 0));
        if(data == nullptr)
        {
            delete e;
            return ROCDENSITYMAT_STATUS_INVALID_VALUE;
        }
        if(data_type == ROCDENSITYMAT_C_64F)
        {
            const auto* src = reinterpret_cast<const cd*>(data);
            for(int64_t i = 0; i < total; ++i) e->host_data[i] = src[i];
        }
        else
        {
            const auto* src = reinterpret_cast<const std::complex<float>*>(data);
            for(int64_t i = 0; i < total; ++i)
                e->host_data[i] = cd(src[i].real(), src[i].imag());
        }
        e->elem_count = total;
        break;
    }
    default:
        delete e;
        return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    }

    *op = e;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_destroy_elementary_operator(
    rocdensitymat_elementary_operator op)
{
    if(op == nullptr) return ROCDENSITYMAT_STATUS_SUCCESS;
    if(op->device_data != nullptr)
    {
        // Best-effort free; we deliberately do not propagate failures, since
        // this path runs from a destructor with no surviving handle context.
        (void)hipFreeAsync(op->device_data, op->device_data_stream);
    }
    delete op;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

} // extern "C"
