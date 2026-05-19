/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Reverse-engineered from cuStateVec public API documentation
 * (cuQuantum 24.11 / cuStateVec 1.7.x); clean-room implementation.
 * ************************************************************************ */

/*! \file
 *  \brief rocstatevec-types.h defines data types used by rocSTATEVEC.
 */

#ifndef ROCSTATEVEC_TYPES_H
#define ROCSTATEVEC_TYPES_H

#include <stddef.h>
#include <stdint.h>

/// \cond DO_NOT_DOCUMENT
typedef struct ihipStream_t* hipStream_t;
/// \endcond

/*! \brief 64-bit signed integer used for state-vector indices and shot counts. */
typedef int64_t rocstatevec_index_t;

/*! \brief Opaque rocSTATEVEC library context handle. */
typedef struct _rocstatevec_handle* rocstatevec_handle;

/*! \brief Opaque sampler descriptor. */
typedef struct _rocstatevec_sampler_descriptor* rocstatevec_sampler_descriptor;

/*! \brief Opaque accessor descriptor. */
typedef struct _rocstatevec_accessor_descriptor* rocstatevec_accessor_descriptor;

/*! \brief Status codes returned by all rocSTATEVEC API functions.
 *
 *  Numeric values match the documented stable values in
 *  `custatevecStatus_t` so that downstream code that interprets status codes
 *  numerically continues to work after a `s/cu/hip/` rename via hipSTATEVEC.
 */
typedef enum {
    ROCSTATEVEC_STATUS_SUCCESS                  = 0,
    ROCSTATEVEC_STATUS_NOT_INITIALIZED          = 1,
    ROCSTATEVEC_STATUS_ALLOC_FAILED             = 2,
    ROCSTATEVEC_STATUS_INVALID_VALUE            = 3,
    ROCSTATEVEC_STATUS_ARCH_MISMATCH            = 4,
    ROCSTATEVEC_STATUS_EXECUTION_FAILED         = 5,
    ROCSTATEVEC_STATUS_INTERNAL_ERROR           = 6,
    ROCSTATEVEC_STATUS_NOT_SUPPORTED            = 7,
    ROCSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE   = 8,
    ROCSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED = 9,
    ROCSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR      = 10,
    ROCSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR   = 11,
    ROCSTATEVEC_STATUS_COMMUNICATOR_ERROR       = 12,
    ROCSTATEVEC_STATUS_LOADING_LIBRARY_FAILED   = 13,
    ROCSTATEVEC_STATUS_MAX_VALUE                = 14
} rocstatevec_status;

/*! \brief Pauli matrix identifiers. Numeric values match `custatevecPauli_t`. */
typedef enum {
    ROCSTATEVEC_PAULI_I = 0,
    ROCSTATEVEC_PAULI_X = 1,
    ROCSTATEVEC_PAULI_Y = 2,
    ROCSTATEVEC_PAULI_Z = 3
} rocstatevec_pauli;

/*! \brief Compute precision used for state-vector and matrix data.
 *
 *  Numeric values match the published `custatevecComputeType_t` for the
 *  precisions rocSTATEVEC supports. Mixed-precision tensor-core paths
 *  exposed by cuStateVec on Hopper are not implemented in v1.0; consumers
 *  that pass them get `ROCSTATEVEC_STATUS_NOT_SUPPORTED`.
 */
typedef enum {
    ROCSTATEVEC_COMPUTE_DEFAULT = 0,
    ROCSTATEVEC_COMPUTE_32F     = (1U << 2),
    ROCSTATEVEC_COMPUTE_64F     = (1U << 4),
    ROCSTATEVEC_COMPUTE_TF32    = (1U << 12)
} rocstatevec_compute_type;

/*! \brief Floating-point complex scalar type tag.
 *
 *  Mirrors `cudaDataType_t`. Only the four complex types that cuStateVec
 *  actually consumes are listed.
 */
typedef enum {
    ROCSTATEVEC_R_32F = 0,
    ROCSTATEVEC_R_64F = 1,
    ROCSTATEVEC_C_32F = 4,
    ROCSTATEVEC_C_64F = 5
} rocstatevec_data_type;

/*! \brief Storage layout for a dense matrix passed by the host. */
typedef enum {
    ROCSTATEVEC_MATRIX_LAYOUT_COL = 0,
    ROCSTATEVEC_MATRIX_LAYOUT_ROW = 1
} rocstatevec_matrix_layout;

/*! \brief Matrix property hint used by validation and apply paths. */
typedef enum {
    ROCSTATEVEC_MATRIX_TYPE_GENERAL   = 0,
    ROCSTATEVEC_MATRIX_TYPE_UNITARY   = 1,
    ROCSTATEVEC_MATRIX_TYPE_HERMITIAN = 2
} rocstatevec_matrix_type;

/*! \brief Index-mapping flavor for generalized permutation matrix apply. */
typedef enum {
    ROCSTATEVEC_MATRIX_MAP_TYPE_BROADCAST = 0,
    ROCSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED = 1
} rocstatevec_matrix_map_type;

/*! \brief Z-basis collapse selector. */
typedef enum {
    ROCSTATEVEC_COLLAPSE_NONE         = 0,
    ROCSTATEVEC_COLLAPSE_NORMALIZE_AS_SPECIFIED = 1
} rocstatevec_collapse_op;

/*! \brief Sampler output format. */
typedef enum {
    ROCSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER = 0,
    ROCSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER = 1
} rocstatevec_sampler_output;

/*! \brief Initial state preset used by `rocstatevec_initialize_state_vector`. */
typedef enum {
    ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO    = 0,
    ROCSTATEVEC_STATE_VECTOR_TYPE_UNIFORM = 1,
    ROCSTATEVEC_STATE_VECTOR_TYPE_GHZ     = 2,
    ROCSTATEVEC_STATE_VECTOR_TYPE_W       = 3
} rocstatevec_state_vector_type;

/*! \brief Library-property selector for `rocstatevec_get_property`. */
typedef enum {
    ROCSTATEVEC_PROPERTY_MAJOR_VERSION = 0,
    ROCSTATEVEC_PROPERTY_MINOR_VERSION = 1,
    ROCSTATEVEC_PROPERTY_PATCH_LEVEL   = 2
} rocstatevec_library_property_type;

/*! \brief User-supplied device memory handler.
 *
 *  Mirrors `custatevecDeviceMemHandler_t`. Functions are async-on-stream;
 *  rocSTATEVEC stores the handler on the handle and uses it whenever a
 *  workspace allocation would otherwise be required.
 */
typedef struct _rocstatevec_device_mem_handler {
    void* ctx;
    int (*device_alloc)(void* ctx, void** ptr, size_t size, hipStream_t stream);
    int (*device_free)(void* ctx, void* ptr, size_t size, hipStream_t stream);
    char name[64];
} rocstatevec_device_mem_handler_t;

/*! \brief Logger callback signature mirroring `custatevecLoggerCallback_t`. */
typedef void (*rocstatevec_logger_callback_t)(int32_t level,
                                              const char* function_name,
                                              const char* message);

/*! \brief Logger callback signature with a user-data pointer. */
typedef void (*rocstatevec_logger_callback_data_t)(int32_t level,
                                                   const char* function_name,
                                                   const char* message,
                                                   void* user_data);

/*! \brief Pair of qubit indices used by \p rocstatevec_swap_index_bits.
 *
 *  Layout-compatible with the CUDA `int2` vector type so a hipSTATEVEC
 *  pass-through to cuStateVec on NVIDIA can `reinterpret_cast` directly.
 */
typedef struct _rocstatevec_index_pair {
    int32_t x;
    int32_t y;
} rocstatevec_index_pair_t;

#endif /* ROCSTATEVEC_TYPES_H */
