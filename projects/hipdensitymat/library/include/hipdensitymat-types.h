/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Public C type definitions for hipDENSITYMAT.
 *
 * Function-name shape and enum numeric values mirror NVIDIA's
 * cuDensityMat (cuQuantum 24.11 / cuDensityMat 0.1.0) so that consumer
 * code written against `<cudensitymat.h>` ports to `<hipdensitymat.h>`
 * with a mechanical `s/cu/hip/` rename. The library is built and runs
 * only against AMD ROCm via the rocDENSITYMAT backend; it does not link
 * to any NVIDIA runtime.
 * ************************************************************************ */

#ifndef HIPDENSITYMAT_TYPES_H
#define HIPDENSITYMAT_TYPES_H

#include <stddef.h>
#include <stdint.h>

/// \cond DO_NOT_DOCUMENT
typedef struct ihipStream_t* hipStream_t;
/// \endcond

typedef struct _hipdensitymatHandle*                  hipdensitymatHandle_t;
typedef struct _hipdensitymatWorkspaceDescriptor*     hipdensitymatWorkspaceDescriptor_t;
typedef struct _hipdensitymatState*                   hipdensitymatState_t;
typedef struct _hipdensitymatElementaryOperator*      hipdensitymatElementaryOperator_t;
typedef struct _hipdensitymatOperatorTerm*            hipdensitymatOperatorTerm_t;
typedef struct _hipdensitymatOperator*                hipdensitymatOperator_t;
typedef struct _hipdensitymatMasterEquationSolver*    hipdensitymatMasterEquationSolver_t;

/*! \brief Status codes returned by all hipDENSITYMAT API functions. */
typedef enum {
    HIPDENSITYMAT_STATUS_SUCCESS                  = 0,
    HIPDENSITYMAT_STATUS_NOT_INITIALIZED          = 1,
    HIPDENSITYMAT_STATUS_ALLOC_FAILED             = 2,
    HIPDENSITYMAT_STATUS_INVALID_VALUE            = 3,
    HIPDENSITYMAT_STATUS_ARCH_MISMATCH            = 4,
    HIPDENSITYMAT_STATUS_EXECUTION_FAILED         = 5,
    HIPDENSITYMAT_STATUS_INTERNAL_ERROR           = 6,
    HIPDENSITYMAT_STATUS_NOT_SUPPORTED            = 7,
    HIPDENSITYMAT_STATUS_CALLBACK_ERROR           = 8,
    HIPDENSITYMAT_STATUS_BLAS_ERROR               = 9,
    HIPDENSITYMAT_STATUS_HIP_ERROR                = 10,
    HIPDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE   = 11,
    HIPDENSITYMAT_STATUS_INSUFFICIENT_DRIVER      = 12,
    HIPDENSITYMAT_STATUS_IO_ERROR                 = 13,
    HIPDENSITYMAT_STATUS_NO_DEVICE_ALLOCATOR      = 15,
    HIPDENSITYMAT_STATUS_DEVICE_ALLOCATOR_ERROR   = 19,
    HIPDENSITYMAT_STATUS_DISTRIBUTED_FAILURE      = 20,
    HIPDENSITYMAT_STATUS_INTERRUPTED              = 21,
    HIPDENSITYMAT_STATUS_DRIVER_VERSION_MISMATCH  = 23,
    HIPDENSITYMAT_STATUS_MAX_VALUE                = 24
} hipdensitymatStatus_t;

typedef enum {
    HIPDENSITYMAT_COMPUTE_64F     = (1U << 4),
    HIPDENSITYMAT_COMPUTE_32F     = (1U << 2),
    HIPDENSITYMAT_COMPUTE_DEFAULT = 0
} hipdensitymatComputeType_t;

typedef enum {
    HIPDENSITYMAT_R_32F = 0,
    HIPDENSITYMAT_R_64F = 1,
    HIPDENSITYMAT_C_32F = 4,
    HIPDENSITYMAT_C_64F = 5
} hipdensitymatDataType_t;

typedef enum {
    HIPDENSITYMAT_PROPERTY_MAJOR_VERSION = 0,
    HIPDENSITYMAT_PROPERTY_MINOR_VERSION = 1,
    HIPDENSITYMAT_PROPERTY_PATCH_LEVEL   = 2
} hipdensitymatLibraryPropertyType_t;

typedef enum {
    HIPDENSITYMAT_STATE_PURITY_PURE  = 0,
    HIPDENSITYMAT_STATE_PURITY_MIXED = 1,
    HIPDENSITYMAT_STATE_PURITY_MPS   = 2
} hipdensitymatStatePurity_t;

typedef enum {
    HIPDENSITYMAT_MEMSPACE_DEVICE = 0,
    HIPDENSITYMAT_MEMSPACE_HOST   = 1
} hipdensitymatMemspace_t;

typedef enum {
    HIPDENSITYMAT_WORKSPACE_SCRATCH = 0,
    HIPDENSITYMAT_WORKSPACE_CACHE   = 1
} hipdensitymatWorkspaceKind_t;

typedef enum {
    HIPDENSITYMAT_DISTRIBUTED_PROVIDER_NONE = 0,
    HIPDENSITYMAT_DISTRIBUTED_PROVIDER_MPI  = 1,
    HIPDENSITYMAT_DISTRIBUTED_PROVIDER_NCCL = 2,
    HIPDENSITYMAT_DISTRIBUTED_PROVIDER_RCCL = 3
} hipdensitymatDistributedProvider_t;

typedef enum {
    HIPDENSITYMAT_ELEMENTARY_IDENTITY = 0,
    HIPDENSITYMAT_ELEMENTARY_PAULI_X  = 1,
    HIPDENSITYMAT_ELEMENTARY_PAULI_Y  = 2,
    HIPDENSITYMAT_ELEMENTARY_PAULI_Z  = 3,
    HIPDENSITYMAT_ELEMENTARY_DENSE    = 4,
    HIPDENSITYMAT_ELEMENTARY_DIAGONAL = 5
} hipdensitymatElementaryKind_t;

typedef enum {
    HIPDENSITYMAT_SOLVER_RK4 = 0
} hipdensitymatSolverKind_t;

typedef enum {
    HIPDENSITYMAT_DUALITY_KET = 0,
    HIPDENSITYMAT_DUALITY_BRA = 1
} hipdensitymatDuality_t;

typedef struct _hipdensitymatComplexDouble { double x; double y; } hipdensitymatComplexDouble_t;
typedef struct _hipdensitymatComplexFloat  { float  x; float  y; } hipdensitymatComplexFloat_t;

typedef hipdensitymatComplexDouble_t (*hipdensitymatScalarCallback_t)(
    double t, int32_t numParams, const double* params);

typedef struct _hipdensitymatDeviceMemHandler {
    void* ctx;
    int (*device_alloc)(void* ctx, void** ptr, size_t size, hipStream_t stream);
    int (*device_free)(void* ctx, void* ptr, size_t size, hipStream_t stream);
    char name[64];
} hipdensitymatDeviceMemHandler_t;

typedef void (*hipdensitymatLoggerCallback_t)(int32_t level,
                                              const char* function_name,
                                              const char* message);
typedef void (*hipdensitymatLoggerCallbackData_t)(int32_t level,
                                                  const char* function_name,
                                                  const char* message,
                                                  void* user_data);

#endif /* HIPDENSITYMAT_TYPES_H */
