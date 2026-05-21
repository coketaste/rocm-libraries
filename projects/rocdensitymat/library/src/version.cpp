/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "rocdensitymat_internal.hpp"

extern "C" {

size_t rocdensitymat_get_version(void)
{
    return static_cast<size_t>(ROCDENSITYMAT_VERSION_MAJOR) * 10000
         + static_cast<size_t>(ROCDENSITYMAT_VERSION_MINOR) * 100
         + static_cast<size_t>(ROCDENSITYMAT_VERSION_PATCH);
}

rocdensitymat_status rocdensitymat_get_property(
    rocdensitymat_library_property_type type, int32_t* value)
{
    if(value == nullptr) return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    switch(type)
    {
    case ROCDENSITYMAT_PROPERTY_MAJOR_VERSION:
        *value = ROCDENSITYMAT_VERSION_MAJOR;
        return ROCDENSITYMAT_STATUS_SUCCESS;
    case ROCDENSITYMAT_PROPERTY_MINOR_VERSION:
        *value = ROCDENSITYMAT_VERSION_MINOR;
        return ROCDENSITYMAT_STATUS_SUCCESS;
    case ROCDENSITYMAT_PROPERTY_PATCH_LEVEL:
        *value = ROCDENSITYMAT_VERSION_PATCH;
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }
    return ROCDENSITYMAT_STATUS_INVALID_VALUE;
}

} // extern "C"
