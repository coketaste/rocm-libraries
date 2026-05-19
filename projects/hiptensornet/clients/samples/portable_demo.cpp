/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Smoke demo of the hipTENSORNET handle and version-query API.
 * The library forwards to rocTENSORNET on a HIP runtime; this sample
 * exercises the AMD pass-through.
 * ************************************************************************ */

#include "hiptensornet.h"

#include <hip/hip_runtime.h>

#include <cstdio>

int main()
{
    int n_dev = 0;
    if(hipGetDeviceCount(&n_dev) != hipSuccess || n_dev <= 0)
    {
        std::printf("no HIP device, skipping demo\n");
        return 0;
    }

    hiptensornet_handle h = nullptr;
    if(hiptensornet_create(&h) != HIPTENSORNET_STATUS_SUCCESS)
    {
        std::printf("hiptensornet_create failed; skipping\n");
        return 0;
    }

    int version = 0;
    hiptensornet_get_version(&version);
    int hip_rt = 0;
    hiptensornet_get_hip_runtime_version(&hip_rt);
    std::printf("hipTENSORNET version=%d, HIP runtime=%d\n", version, hip_rt);

    hiptensornet_destroy(h);
    return 0;
}
