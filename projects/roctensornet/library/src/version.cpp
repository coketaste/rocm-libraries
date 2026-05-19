/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "roctensornet_internal.hpp"

#include <hip/hip_version.h>

extern "C" {

roctensornet_status roctensornet_get_version(int* version)
{
    ROCTENSORNET_CHECK_PTR(version);
    *version = ROCTENSORNET_VERSION;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status roctensornet_get_hip_runtime_version(int* version)
{
    ROCTENSORNET_CHECK_PTR(version);
    int v = 0;
    if(hipRuntimeGetVersion(&v) != hipSuccess)
    {
#ifdef HIP_VERSION
        v = HIP_VERSION;
#endif
    }
    *version = v;
    return ROCTENSORNET_STATUS_SUCCESS;
}

} // extern "C"
