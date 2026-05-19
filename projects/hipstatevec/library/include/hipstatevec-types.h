/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Public C type definitions for hipSTATEVEC.
 *
 * Function-name shape and enum numeric values mirror NVIDIA's cuStateVec
 * (cuQuantum 24.11 / cuStateVec 1.7.x) so that consumer code written
 * against `<custatevec.h>` ports to `<hipstatevec.h>` with a mechanical
 * `s/cu/hip/` rename. The library is built and runs only against AMD
 * ROCm via the rocSTATEVEC backend; it does not link to any NVIDIA
 * runtime.
 * ************************************************************************ */

#ifndef HIPSTATEVEC_TYPES_H
#define HIPSTATEVEC_TYPES_H

#include <stddef.h>
#include <stdint.h>

/// \cond DO_NOT_DOCUMENT
typedef struct ihipStream_t* hipStream_t;
/// \endcond

/*! \brief 64-bit signed integer used for state-vector indices. */
typedef int64_t hipstatevecIndex_t;

/*! \brief Opaque hipSTATEVEC library context handle. */
typedef struct _hipstatevecHandle*    hipstatevecHandle_t;
typedef struct _hipstatevecSampler*   hipstatevecSamplerDescriptor_t;
typedef struct _hipstatevecAccessor*  hipstatevecAccessorDescriptor_t;

/*! \brief Status codes returned by all hipSTATEVEC API functions. */
typedef enum {
    HIPSTATEVEC_STATUS_SUCCESS                  = 0,
    HIPSTATEVEC_STATUS_NOT_INITIALIZED          = 1,
    HIPSTATEVEC_STATUS_ALLOC_FAILED             = 2,
    HIPSTATEVEC_STATUS_INVALID_VALUE            = 3,
    HIPSTATEVEC_STATUS_ARCH_MISMATCH            = 4,
    HIPSTATEVEC_STATUS_EXECUTION_FAILED         = 5,
    HIPSTATEVEC_STATUS_INTERNAL_ERROR           = 6,
    HIPSTATEVEC_STATUS_NOT_SUPPORTED            = 7,
    HIPSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE   = 8,
    HIPSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED = 9,
    HIPSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR      = 10,
    HIPSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR   = 11,
    HIPSTATEVEC_STATUS_COMMUNICATOR_ERROR       = 12,
    HIPSTATEVEC_STATUS_LOADING_LIBRARY_FAILED   = 13,
    HIPSTATEVEC_STATUS_MAX_VALUE                = 14
} hipstatevecStatus_t;

typedef enum {
    HIPSTATEVEC_PAULI_I = 0,
    HIPSTATEVEC_PAULI_X = 1,
    HIPSTATEVEC_PAULI_Y = 2,
    HIPSTATEVEC_PAULI_Z = 3
} hipstatevecPauli_t;

typedef enum {
    HIPSTATEVEC_COMPUTE_DEFAULT = 0,
    HIPSTATEVEC_COMPUTE_32F     = (1U << 2),
    HIPSTATEVEC_COMPUTE_64F     = (1U << 4),
    HIPSTATEVEC_COMPUTE_TF32    = (1U << 12)
} hipstatevecComputeType_t;

typedef enum {
    HIPSTATEVEC_R_32F = 0,
    HIPSTATEVEC_R_64F = 1,
    HIPSTATEVEC_C_32F = 4,
    HIPSTATEVEC_C_64F = 5
} hipstatevecDataType_t;

typedef enum {
    HIPSTATEVEC_MATRIX_LAYOUT_COL = 0,
    HIPSTATEVEC_MATRIX_LAYOUT_ROW = 1
} hipstatevecMatrixLayout_t;

typedef enum {
    HIPSTATEVEC_MATRIX_TYPE_GENERAL   = 0,
    HIPSTATEVEC_MATRIX_TYPE_UNITARY   = 1,
    HIPSTATEVEC_MATRIX_TYPE_HERMITIAN = 2
} hipstatevecMatrixType_t;

typedef enum {
    HIPSTATEVEC_MATRIX_MAP_TYPE_BROADCAST      = 0,
    HIPSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED = 1
} hipstatevecMatrixMapType_t;

typedef enum {
    HIPSTATEVEC_COLLAPSE_NONE                   = 0,
    HIPSTATEVEC_COLLAPSE_NORMALIZE_AS_SPECIFIED = 1
} hipstatevecCollapseOp_t;

typedef enum {
    HIPSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER   = 0,
    HIPSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER = 1
} hipstatevecSamplerOutput_t;

typedef enum {
    HIPSTATEVEC_STATE_VECTOR_TYPE_ZERO    = 0,
    HIPSTATEVEC_STATE_VECTOR_TYPE_UNIFORM = 1,
    HIPSTATEVEC_STATE_VECTOR_TYPE_GHZ     = 2,
    HIPSTATEVEC_STATE_VECTOR_TYPE_W       = 3
} hipstatevecStateVectorType_t;

typedef enum {
    HIPSTATEVEC_PROPERTY_MAJOR_VERSION = 0,
    HIPSTATEVEC_PROPERTY_MINOR_VERSION = 1,
    HIPSTATEVEC_PROPERTY_PATCH_LEVEL   = 2
} hipstatevecLibraryPropertyType_t;

typedef struct _hipstatevecDeviceMemHandler {
    void* ctx;
    int (*device_alloc)(void* ctx, void** ptr, size_t size, hipStream_t stream);
    int (*device_free)(void* ctx, void* ptr, size_t size, hipStream_t stream);
    char name[64];
} hipstatevecDeviceMemHandler_t;

typedef void (*hipstatevecLoggerCallback_t)(int32_t level, const char* function_name, const char* message);
typedef void (*hipstatevecLoggerCallbackData_t)(int32_t level, const char* function_name, const char* message, void* user_data);

/*! \brief Pair of qubit indices used by `hipstatevecSwapIndexBits`. */
typedef struct _hipstatevecIndexPair {
    int32_t x;
    int32_t y;
} hipstatevecIndexPair_t;

#endif /* HIPSTATEVEC_TYPES_H */
