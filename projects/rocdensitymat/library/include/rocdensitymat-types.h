/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Reverse-engineered from cuDensityMat public API documentation
 * (cuQuantum 24.11 / cuDensityMat 0.1.0); clean-room implementation.
 * ************************************************************************ */

/*! \file
 *  \brief rocdensitymat-types.h defines data types used by rocDENSITYMAT.
 */

#ifndef ROCDENSITYMAT_TYPES_H
#define ROCDENSITYMAT_TYPES_H

#include <stddef.h>
#include <stdint.h>

/// \cond DO_NOT_DOCUMENT
typedef struct ihipStream_t* hipStream_t;
/// \endcond

/*! \brief Opaque rocDENSITYMAT library context handle. */
typedef struct _rocdensitymat_handle* rocdensitymat_handle;

/*! \brief Opaque workspace descriptor handle. */
typedef struct _rocdensitymat_workspace_descriptor* rocdensitymat_workspace_descriptor;

/*! \brief Opaque quantum-state descriptor handle. */
typedef struct _rocdensitymat_state* rocdensitymat_state;

/*! \brief Opaque elementary-operator descriptor handle. */
typedef struct _rocdensitymat_elementary_operator* rocdensitymat_elementary_operator;

/*! \brief Opaque operator-term descriptor handle. */
typedef struct _rocdensitymat_operator_term* rocdensitymat_operator_term;

/*! \brief Opaque operator descriptor handle. */
typedef struct _rocdensitymat_operator* rocdensitymat_operator;

/*! \brief Opaque master-equation-solver descriptor handle. */
typedef struct _rocdensitymat_master_equation_solver* rocdensitymat_master_equation_solver;

/*! \brief Opaque distributed-communicator descriptor handle. */
typedef struct _rocdensitymat_distributed_comm* rocdensitymat_distributed_comm;

/*! \brief Opaque distributed-request descriptor handle. */
typedef struct _rocdensitymat_distributed_request* rocdensitymat_distributed_request;

/*! \brief Status codes returned by all rocDENSITYMAT API functions.
 *
 *  Numeric values are aligned with the documented stable values of
 *  `cudensitymatStatus_t` so that downstream code that interprets status
 *  codes numerically continues to work after a `s/cu/hip/` rename via
 *  hipDENSITYMAT.
 */
typedef enum {
    ROCDENSITYMAT_STATUS_SUCCESS                  = 0,
    ROCDENSITYMAT_STATUS_NOT_INITIALIZED          = 1,
    ROCDENSITYMAT_STATUS_ALLOC_FAILED             = 2,
    ROCDENSITYMAT_STATUS_INVALID_VALUE            = 3,
    ROCDENSITYMAT_STATUS_ARCH_MISMATCH            = 4,
    ROCDENSITYMAT_STATUS_EXECUTION_FAILED         = 5,
    ROCDENSITYMAT_STATUS_INTERNAL_ERROR           = 6,
    ROCDENSITYMAT_STATUS_NOT_SUPPORTED            = 7,
    ROCDENSITYMAT_STATUS_CALLBACK_ERROR           = 8,
    ROCDENSITYMAT_STATUS_BLAS_ERROR               = 9,
    ROCDENSITYMAT_STATUS_HIP_ERROR                = 10,
    ROCDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE   = 11,
    ROCDENSITYMAT_STATUS_INSUFFICIENT_DRIVER      = 12,
    ROCDENSITYMAT_STATUS_IO_ERROR                 = 13,
    ROCDENSITYMAT_STATUS_NO_DEVICE_ALLOCATOR      = 15,
    ROCDENSITYMAT_STATUS_DEVICE_ALLOCATOR_ERROR   = 19,
    ROCDENSITYMAT_STATUS_DISTRIBUTED_FAILURE      = 20,
    ROCDENSITYMAT_STATUS_INTERRUPTED              = 21,
    ROCDENSITYMAT_STATUS_DRIVER_VERSION_MISMATCH  = 23,
    ROCDENSITYMAT_STATUS_MAX_VALUE                = 24
} rocdensitymat_status;

/*! \brief Compute precision used for state and operator data.
 *
 *  Numeric values mirror `cudensitymatComputeType_t` for the precisions
 *  rocDENSITYMAT supports. Mixed-precision tensor-core paths are not
 *  implemented in v0.1.0; consumers that pass them get
 *  `ROCDENSITYMAT_STATUS_NOT_SUPPORTED`.
 */
typedef enum {
    ROCDENSITYMAT_COMPUTE_64F     = (1U << 4),
    ROCDENSITYMAT_COMPUTE_32F     = (1U << 2),
    ROCDENSITYMAT_COMPUTE_DEFAULT = 0
} rocdensitymat_compute_type;

/*! \brief Floating-point complex scalar type tag.
 *
 *  Mirrors `cudaDataType_t`. Only the four complex types that
 *  cuDensityMat actually consumes are listed.
 */
typedef enum {
    ROCDENSITYMAT_R_32F = 0,
    ROCDENSITYMAT_R_64F = 1,
    ROCDENSITYMAT_C_32F = 4,
    ROCDENSITYMAT_C_64F = 5
} rocdensitymat_data_type;

/*! \brief Library-property selector for `rocdensitymat_get_property`. */
typedef enum {
    ROCDENSITYMAT_PROPERTY_MAJOR_VERSION = 0,
    ROCDENSITYMAT_PROPERTY_MINOR_VERSION = 1,
    ROCDENSITYMAT_PROPERTY_PATCH_LEVEL   = 2
} rocdensitymat_library_property_type;

/*! \brief Quantum-state purity. PURITY_MPS is reserved for v0.2. */
typedef enum {
    ROCDENSITYMAT_STATE_PURITY_PURE  = 0,
    ROCDENSITYMAT_STATE_PURITY_MIXED = 1,
    ROCDENSITYMAT_STATE_PURITY_MPS   = 2
} rocdensitymat_state_purity;

/*! \brief Memory space for workspace buffers. */
typedef enum {
    ROCDENSITYMAT_MEMSPACE_DEVICE = 0,
    ROCDENSITYMAT_MEMSPACE_HOST   = 1
} rocdensitymat_memspace;

/*! \brief Workspace kind: scratch (per-call) or cache (cross-call). */
typedef enum {
    ROCDENSITYMAT_WORKSPACE_SCRATCH = 0,
    ROCDENSITYMAT_WORKSPACE_CACHE   = 1
} rocdensitymat_workspace_kind;

/*! \brief Distributed multi-rank provider; v0.1.0 only accepts NONE. */
typedef enum {
    ROCDENSITYMAT_DISTRIBUTED_PROVIDER_NONE = 0,
    ROCDENSITYMAT_DISTRIBUTED_PROVIDER_MPI  = 1,
    ROCDENSITYMAT_DISTRIBUTED_PROVIDER_NCCL = 2,
    ROCDENSITYMAT_DISTRIBUTED_PROVIDER_RCCL = 3
} rocdensitymat_distributed_provider;

/*! \brief Convenience kinds for `rocdensitymat_create_elementary_operator`. */
typedef enum {
    ROCDENSITYMAT_ELEMENTARY_IDENTITY = 0,
    ROCDENSITYMAT_ELEMENTARY_PAULI_X  = 1,
    ROCDENSITYMAT_ELEMENTARY_PAULI_Y  = 2,
    ROCDENSITYMAT_ELEMENTARY_PAULI_Z  = 3,
    ROCDENSITYMAT_ELEMENTARY_DENSE    = 4,
    ROCDENSITYMAT_ELEMENTARY_DIAGONAL = 5
} rocdensitymat_elementary_kind;

/*! \brief Master-equation integrator selector. v0.1.0 only ships RK4. */
typedef enum {
    ROCDENSITYMAT_SOLVER_RK4 = 0
} rocdensitymat_solver_kind;

/*! \brief Mode-action duality flag for an OperatorTerm factor.
 *
 *  ``KET = 0`` applies the elementary operator on the left (ket / row)
 *  index of a mixed state, ``BRA = 1`` on the right (bra / column) index.
 *  Pure states only see ``KET``.
 */
typedef enum {
    ROCDENSITYMAT_DUALITY_KET = 0,
    ROCDENSITYMAT_DUALITY_BRA = 1
} rocdensitymat_duality;

/*! \brief Complex scalar pair used for host-side coefficients.
 *
 *  Layout-compatible with `hipDoubleComplex` so a `reinterpret_cast`
 *  is guaranteed to be valid.
 */
typedef struct _rocdensitymat_complex_double {
    double x;
    double y;
} rocdensitymat_complex_double;

/*! \brief 32-bit complex scalar pair. Layout-compatible with `hipFloatComplex`. */
typedef struct _rocdensitymat_complex_float {
    float x;
    float y;
} rocdensitymat_complex_float;

/*! \brief Time-dependent scalar callback for `OperatorTermAppendElementaryProduct`.
 *
 *  Returned scalar is multiplied into the operator-term coefficient at the
 *  current simulation time. ``params`` is a host pointer to ``num_params``
 *  doubles passed unchanged from the caller. The callback runs on the host
 *  inside the operator-action prepare/compute path.
 */
typedef rocdensitymat_complex_double (*rocdensitymat_scalar_callback_t)(
    double t, int32_t num_params, const double* params);

/*! \brief User-supplied device-memory handler.
 *
 *  Mirrors `cudensitymatDeviceMemHandler_t`. Functions are async-on-stream;
 *  rocDENSITYMAT stores the handler on the handle and uses it whenever a
 *  workspace allocation would otherwise be required.
 */
typedef struct _rocdensitymat_device_mem_handler {
    void* ctx;
    int (*device_alloc)(void* ctx, void** ptr, size_t size, hipStream_t stream);
    int (*device_free)(void* ctx, void* ptr, size_t size, hipStream_t stream);
    char name[64];
} rocdensitymat_device_mem_handler_t;

/*! \brief Logger callback signature mirroring `cudensitymatLoggerCallback_t`. */
typedef void (*rocdensitymat_logger_callback_t)(int32_t level,
                                                const char* function_name,
                                                const char* message);

/*! \brief Logger callback signature with a user-data pointer. */
typedef void (*rocdensitymat_logger_callback_data_t)(int32_t level,
                                                     const char* function_name,
                                                     const char* message,
                                                     void* user_data);

#endif /* ROCDENSITYMAT_TYPES_H */
