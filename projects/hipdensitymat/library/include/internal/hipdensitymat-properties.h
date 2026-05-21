/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPDENSITYMAT_PROPERTIES_H
#define HIPDENSITYMAT_PROPERTIES_H

#include "../hipdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipdensitymatStatus_t hipdensitymatStateGetDataType(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    hipdensitymatDataType_t* dataType);

hipdensitymatStatus_t hipdensitymatStateGetPurity(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    hipdensitymatStatePurity_t* purity);

hipdensitymatStatus_t hipdensitymatStateGetSpaceShape(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    int32_t* numSpaceModes, int64_t* spaceShape);

#ifdef __cplusplus
}
#endif

#endif /* HIPDENSITYMAT_PROPERTIES_H */
