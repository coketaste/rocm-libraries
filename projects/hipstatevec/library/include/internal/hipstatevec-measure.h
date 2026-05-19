/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#ifndef HIPSTATEVEC_MEASURE_H
#define HIPSTATEVEC_MEASURE_H

#include "../hipstatevec-types.h"

#ifdef __cplusplus
extern "C" {
#endif

hipstatevecStatus_t hipstatevecAbs2SumArray(
    hipstatevecHandle_t      handle,
    const void*              state_vector,
    hipstatevecDataType_t    state_vector_data_type,
    uint32_t                 n_index_bits,
    double*                  abs2_sum,
    const int32_t*           bit_ordering,
    uint32_t                 bit_ordering_len,
    const int32_t*           mask_bit_string,
    const int32_t*           mask_ordering,
    uint32_t                 mask_len);

hipstatevecStatus_t hipstatevecAbs2SumOnZBasis(
    hipstatevecHandle_t      handle,
    const void*              state_vector,
    hipstatevecDataType_t    state_vector_data_type,
    uint32_t                 n_index_bits,
    double*                  abs2_sum0,
    double*                  abs2_sum1,
    const int32_t*           basis_bits,
    uint32_t                 n_basis_bits);

hipstatevecStatus_t hipstatevecMeasureOnZBasis(
    hipstatevecHandle_t      handle,
    void*                    state_vector,
    hipstatevecDataType_t    state_vector_data_type,
    uint32_t                 n_index_bits,
    int32_t*                 parity,
    const int32_t*           basis_bits,
    uint32_t                 n_basis_bits,
    double                   randnum,
    hipstatevecCollapseOp_t  collapse);

hipstatevecStatus_t hipstatevecBatchMeasure(
    hipstatevecHandle_t      handle,
    void*                    state_vector,
    hipstatevecDataType_t    state_vector_data_type,
    uint32_t                 n_index_bits,
    int32_t*                 bit_string,
    const int32_t*           bit_ordering,
    uint32_t                 bit_string_len,
    double                   randnum,
    hipstatevecCollapseOp_t  collapse);

hipstatevecStatus_t hipstatevecBatchMeasureWithOffset(
    hipstatevecHandle_t      handle,
    void*                    state_vector,
    hipstatevecDataType_t    state_vector_data_type,
    uint32_t                 n_index_bits,
    int32_t*                 bit_string,
    const int32_t*           bit_ordering,
    uint32_t                 bit_string_len,
    double                   randnum,
    hipstatevecCollapseOp_t  collapse,
    double                   offset,
    double                   abs2_sum);

hipstatevecStatus_t hipstatevecCollapseOnZBasis(
    hipstatevecHandle_t      handle,
    void*                    state_vector,
    hipstatevecDataType_t    state_vector_data_type,
    uint32_t                 n_index_bits,
    int32_t                  parity,
    const int32_t*           basis_bits,
    uint32_t                 n_basis_bits,
    double                   norm);

hipstatevecStatus_t hipstatevecCollapseByBitString(
    hipstatevecHandle_t      handle,
    void*                    state_vector,
    hipstatevecDataType_t    state_vector_data_type,
    uint32_t                 n_index_bits,
    const int32_t*           bit_string,
    const int32_t*           bit_ordering,
    uint32_t                 bit_string_len,
    double                   norm);

#ifdef __cplusplus
}
#endif

#endif /* HIPSTATEVEC_MEASURE_H */
