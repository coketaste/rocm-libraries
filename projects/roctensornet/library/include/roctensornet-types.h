/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Public type definitions and enums for rocTENSORNET.
 *
 * The naming and meaning of every token here follow the cuTensorNet 2.x
 * public API (cuQuantum 24.11). This header is consumable from both C
 * and C++ code.
 * ************************************************************************ */

#ifndef ROCTENSORNET_TYPES_H
#define ROCTENSORNET_TYPES_H

#include <hip/hip_runtime.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Status / error codes ---- */
typedef enum {
    ROCTENSORNET_STATUS_SUCCESS            = 0,
    ROCTENSORNET_STATUS_NOT_INITIALIZED    = 1,
    ROCTENSORNET_STATUS_ALLOC_FAILED       = 2,
    ROCTENSORNET_STATUS_INVALID_VALUE      = 3,
    ROCTENSORNET_STATUS_ARCH_MISMATCH      = 4,
    ROCTENSORNET_STATUS_EXECUTION_FAILED   = 5,
    ROCTENSORNET_STATUS_INTERNAL_ERROR     = 6,
    ROCTENSORNET_STATUS_NOT_SUPPORTED      = 7,
    ROCTENSORNET_STATUS_LICENSE_ERROR      = 8,
    ROCTENSORNET_STATUS_DEVICE_ALLOCATOR_ERROR = 9,
    ROCTENSORNET_STATUS_IO_ERROR           = 10,
    ROCTENSORNET_STATUS_INSUFFICIENT_WORKSPACE = 11,
    ROCTENSORNET_STATUS_INSUFFICIENT_DRIVER    = 12,
    ROCTENSORNET_STATUS_INTERRUPTED        = 13,
} roctensornet_status;

/* ---- Common scalar data types (mirror cudaDataType_t shape) ---- */
typedef enum {
    ROCTENSORNET_R_32F = 0, /* float */
    ROCTENSORNET_R_64F = 1, /* double */
    ROCTENSORNET_C_32F = 2, /* complex<float> */
    ROCTENSORNET_C_64F = 3, /* complex<double> */
    ROCTENSORNET_R_16F = 4, /* fp16 */
    ROCTENSORNET_R_16BF = 5, /* bf16 */
} roctensornet_data_type;

/* ---- Compute type for contractions and SVDs ---- */
typedef enum {
    ROCTENSORNET_COMPUTE_DEFAULT = 0,
    ROCTENSORNET_COMPUTE_32F     = 1,
    ROCTENSORNET_COMPUTE_64F     = 2,
    ROCTENSORNET_COMPUTE_16F     = 3,
    ROCTENSORNET_COMPUTE_16BF    = 4,
    ROCTENSORNET_COMPUTE_TF32    = 5,
} roctensornet_compute_type;

/* ---- Opaque handles ---- */
typedef struct roctensornet_handle_st*                       roctensornet_handle;
typedef struct roctensornet_network_descriptor_st*           roctensornet_network_descriptor;
typedef struct roctensornet_tensor_descriptor_st*            roctensornet_tensor_descriptor;
typedef struct roctensornet_contraction_optimizer_config_st* roctensornet_contraction_optimizer_config;
typedef struct roctensornet_contraction_optimizer_info_st*   roctensornet_contraction_optimizer_info;
typedef struct roctensornet_workspace_descriptor_st*         roctensornet_workspace_descriptor;
typedef struct roctensornet_contraction_plan_st*             roctensornet_contraction_plan;
typedef struct roctensornet_contraction_autotune_preference_st*
        roctensornet_contraction_autotune_preference;
typedef struct roctensornet_slice_group_st*                  roctensornet_slice_group;
typedef struct roctensornet_tensor_svd_config_st*            roctensornet_tensor_svd_config;
typedef struct roctensornet_tensor_svd_info_st*              roctensornet_tensor_svd_info;
typedef struct roctensornet_network_operator_st*             roctensornet_network_operator;
typedef struct roctensornet_state_st*                        roctensornet_state;
typedef struct roctensornet_state_marginal_st*               roctensornet_state_marginal;
typedef struct roctensornet_state_sampler_st*                roctensornet_state_sampler;
typedef struct roctensornet_state_expectation_st*            roctensornet_state_expectation;
typedef struct roctensornet_state_accessor_st*               roctensornet_state_accessor;

/* ---- 64-bit index type used throughout the API ---- */
typedef int64_t roctensornet_index_t;

/* ---- Memory space enum (for workspace descriptor) ---- */
typedef enum {
    ROCTENSORNET_MEMSPACE_DEVICE = 0,
    ROCTENSORNET_MEMSPACE_HOST   = 1,
} roctensornet_memspace;

/* ---- Workspace kind enum ---- */
typedef enum {
    ROCTENSORNET_WORKSPACE_SCRATCH = 0,
    ROCTENSORNET_WORKSPACE_CACHE   = 1,
} roctensornet_workspace_kind;

/* ---- Workspace size preference ---- */
typedef enum {
    ROCTENSORNET_WORKSIZE_PREF_MIN         = 0,
    ROCTENSORNET_WORKSIZE_PREF_RECOMMENDED = 1,
    ROCTENSORNET_WORKSIZE_PREF_MAX         = 2,
} roctensornet_worksize_pref;

/* ---- Network attribute enum (subset; extend as needed) ---- */
typedef enum {
    ROCTENSORNET_NETWORK_INPUT_TENSORS_DATA_TYPES  = 0,
    ROCTENSORNET_NETWORK_OUTPUT_TENSOR_DATA_TYPE   = 1,
    ROCTENSORNET_NETWORK_COMPUTE_TYPE              = 2,
    ROCTENSORNET_NETWORK_INPUT_TENSORS_NUM_MODES   = 3,
    ROCTENSORNET_NETWORK_OUTPUT_TENSOR_NUM_MODES   = 4,
    ROCTENSORNET_NETWORK_INPUT_TENSORS_NUM         = 5,
} roctensornet_network_attribute;

/* ---- Optimizer config attribute enum ---- */
typedef enum {
    ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_NUM_GRAPH_ITERATIONS        = 0,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_NUM_GRAPH_CUTS              = 1,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM             = 2,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS     = 3,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_LEAVES         = 4,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING      = 5,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL         = 6,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_FACTOR        = 7,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED                        = 8,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE     = 9,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_CACHE_REUSE_NRUNS           = 10,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION                = 11,
} roctensornet_contraction_optimizer_config_attribute;

/* ---- Optimizer info attribute enum ---- */
typedef enum {
    ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES         = 0,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES   = 1,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE        = 2,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT      = 3,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH               = 4,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT         = 5,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR     = 6,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_EFFECTIVE_FLOPS    = 7,
    ROCTENSORNET_CONTRACTION_OPTIMIZER_INFO_INTERMEDIATE_MODES = 8,
} roctensornet_contraction_optimizer_info_attribute;

/* ---- Autotune preference attribute enum ---- */
typedef enum {
    ROCTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS              = 0,
    ROCTENSORNET_CONTRACTION_AUTOTUNE_INTERMEDIATE_MODES          = 1,
    ROCTENSORNET_CONTRACTION_AUTOTUNE_GEMM_ALGORITHM              = 2,
} roctensornet_contraction_autotune_preference_attribute;

/* ---- SVD config attribute enum ---- */
typedef enum {
    ROCTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF                 = 0,
    ROCTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF                 = 1,
    ROCTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION            = 2,
    ROCTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION                = 3,
    ROCTENSORNET_TENSOR_SVD_CONFIG_ALGO                       = 4,
    ROCTENSORNET_TENSOR_SVD_CONFIG_MAX_EXTENT                 = 5,
    ROCTENSORNET_TENSOR_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF    = 6,
} roctensornet_tensor_svd_config_attribute;

typedef enum {
    ROCTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT                  = 0,
    ROCTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT               = 1,
    ROCTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT             = 2,
    ROCTENSORNET_TENSOR_SVD_INFO_ALGO                         = 3,
    ROCTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS                  = 4,
} roctensornet_tensor_svd_info_attribute;

typedef enum {
    ROCTENSORNET_TENSOR_SVD_NORMALIZATION_NONE = 0,
    ROCTENSORNET_TENSOR_SVD_NORMALIZATION_L1   = 1,
    ROCTENSORNET_TENSOR_SVD_NORMALIZATION_L2   = 2,
    ROCTENSORNET_TENSOR_SVD_NORMALIZATION_LINF = 3,
} roctensornet_tensor_svd_normalization;

typedef enum {
    ROCTENSORNET_TENSOR_SVD_PARTITION_NONE = 0,
    ROCTENSORNET_TENSOR_SVD_PARTITION_US   = 1,
    ROCTENSORNET_TENSOR_SVD_PARTITION_SV   = 2,
    ROCTENSORNET_TENSOR_SVD_PARTITION_UV   = 3,
} roctensornet_tensor_svd_partition;

typedef enum {
    ROCTENSORNET_TENSOR_SVD_ALGO_GESVD = 0,
    ROCTENSORNET_TENSOR_SVD_ALGO_GESVDJ = 1,
    ROCTENSORNET_TENSOR_SVD_ALGO_GESVDP = 2,
    ROCTENSORNET_TENSOR_SVD_ALGO_GESVDR = 3,
} roctensornet_tensor_svd_algo;

/* ---- Network state purity ---- */
typedef enum {
    ROCTENSORNET_STATE_PURITY_PURE  = 0,
    ROCTENSORNET_STATE_PURITY_MIXED = 1,
} roctensornet_state_purity;

/* ---- Network state attribute enum ---- */
typedef enum {
    ROCTENSORNET_STATE_CONFIG_NUM_HYPER_SAMPLES  = 0,
    ROCTENSORNET_STATE_CONFIG_MPS_CANONICAL_CENTER = 1,
    ROCTENSORNET_STATE_CONFIG_MPS_SVD_CONFIG_ABS_CUTOFF = 2,
    ROCTENSORNET_STATE_CONFIG_MPS_SVD_CONFIG_REL_CUTOFF = 3,
    ROCTENSORNET_STATE_CONFIG_MPS_SVD_CONFIG_S_NORMALIZATION = 4,
    ROCTENSORNET_STATE_CONFIG_MPS_SVD_CONFIG_S_PARTITION = 5,
    ROCTENSORNET_STATE_CONFIG_MPS_SVD_CONFIG_ALGO = 6,
    ROCTENSORNET_STATE_CONFIG_MPS_MPO_APPLICATION = 7,
    ROCTENSORNET_STATE_CONFIG_MPS_GAUGE_OPTION    = 8,
    ROCTENSORNET_STATE_INFO_FLOPS                = 9,
} roctensornet_state_attribute;

/* ---- Marginal/Sampler/Expectation/Accessor attribute enums ---- */
typedef enum {
    ROCTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES = 0,
    ROCTENSORNET_MARGINAL_INFO_FLOPS               = 1,
} roctensornet_marginal_attribute;

typedef enum {
    ROCTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES = 0,
    ROCTENSORNET_SAMPLER_CONFIG_DETERMINISTIC     = 1,
    ROCTENSORNET_SAMPLER_INFO_FLOPS               = 2,
} roctensornet_sampler_attribute;

typedef enum {
    ROCTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES = 0,
    ROCTENSORNET_EXPECTATION_INFO_FLOPS               = 1,
} roctensornet_expectation_attribute;

typedef enum {
    ROCTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES = 0,
    ROCTENSORNET_ACCESSOR_INFO_FLOPS               = 1,
} roctensornet_accessor_attribute;

/* ---- Device memory handler ---- */
typedef struct roctensornet_device_mem_handler {
    void* ctx;
    int (*device_alloc)(void* ctx, void** ptr, size_t size, hipStream_t stream);
    int (*device_free) (void* ctx, void* ptr, size_t size, hipStream_t stream);
    char name[64];
} roctensornet_device_mem_handler_t;

/* ---- Logger types ---- */
typedef void (*roctensornet_logger_callback_t)(int32_t logLevel, const char* functionName, const char* message);
typedef void (*roctensornet_logger_callback_data_t)(int32_t logLevel, const char* functionName, const char* message, void* userData);

#ifdef __cplusplus
}
#endif

#endif /* ROCTENSORNET_TYPES_H */
