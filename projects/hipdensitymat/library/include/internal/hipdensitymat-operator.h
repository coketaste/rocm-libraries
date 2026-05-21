/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPDENSITYMAT_OPERATOR_H
#define HIPDENSITYMAT_OPERATOR_H

#include "../hipdensitymat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipdensitymatStatus_t hipdensitymatCreateElementaryOperator(
    hipdensitymatHandle_t handle,
    int32_t numModes, const int64_t* modeExtents,
    hipdensitymatElementaryKind_t kind,
    hipdensitymatDataType_t dataType,
    const void* data,
    hipdensitymatScalarCallback_t tdepCallback,
    hipdensitymatElementaryOperator_t* op);

hipdensitymatStatus_t hipdensitymatDestroyElementaryOperator(
    hipdensitymatElementaryOperator_t op);

hipdensitymatStatus_t hipdensitymatCreateOperatorTerm(
    hipdensitymatHandle_t handle,
    int32_t numSpaceModes, const int64_t* spaceShape,
    hipdensitymatOperatorTerm_t* term);

hipdensitymatStatus_t hipdensitymatDestroyOperatorTerm(
    hipdensitymatOperatorTerm_t term);

hipdensitymatStatus_t hipdensitymatOperatorTermAppendElementaryProduct(
    hipdensitymatHandle_t handle,
    hipdensitymatOperatorTerm_t term,
    int32_t numFactors,
    const hipdensitymatElementaryOperator_t* operators,
    const int32_t* stateModes,
    const int32_t* modeActionDuality,
    hipdensitymatComplexDouble_t coefficient,
    hipdensitymatScalarCallback_t tdepCallback);

hipdensitymatStatus_t hipdensitymatCreateOperator(
    hipdensitymatHandle_t handle,
    int32_t numSpaceModes, const int64_t* spaceShape,
    hipdensitymatOperator_t* op);

hipdensitymatStatus_t hipdensitymatDestroyOperator(hipdensitymatOperator_t op);

hipdensitymatStatus_t hipdensitymatOperatorAppendTerm(
    hipdensitymatHandle_t handle,
    hipdensitymatOperator_t op,
    hipdensitymatOperatorTerm_t term,
    int32_t dualityOffset,
    hipdensitymatComplexDouble_t coefficient,
    hipdensitymatScalarCallback_t tdepCallback);

#ifdef __cplusplus
}
#endif

#endif /* HIPDENSITYMAT_OPERATOR_H */
