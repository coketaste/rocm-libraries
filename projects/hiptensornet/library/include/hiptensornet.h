/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * hipTENSORNET umbrella header.
 *
 * Function-name correspondence with cuTensorNet 2.x is bijective and
 * lexical, with `cu` replaced by `hip`. This header is self-contained
 * (does not transitively include `<cutensornet.h>`). The library is
 * built and runs only against AMD ROCm via the rocTENSORNET backend.
 * ************************************************************************ */

#ifndef HIPTENSORNET_H
#define HIPTENSORNET_H

#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Status ---- */
typedef enum {
    HIPTENSORNET_STATUS_SUCCESS                = 0,
    HIPTENSORNET_STATUS_NOT_INITIALIZED        = 1,
    HIPTENSORNET_STATUS_ALLOC_FAILED           = 2,
    HIPTENSORNET_STATUS_INVALID_VALUE          = 3,
    HIPTENSORNET_STATUS_ARCH_MISMATCH          = 4,
    HIPTENSORNET_STATUS_EXECUTION_FAILED       = 5,
    HIPTENSORNET_STATUS_INTERNAL_ERROR         = 6,
    HIPTENSORNET_STATUS_NOT_SUPPORTED          = 7,
    HIPTENSORNET_STATUS_LICENSE_ERROR          = 8,
    HIPTENSORNET_STATUS_DEVICE_ALLOCATOR_ERROR = 9,
    HIPTENSORNET_STATUS_IO_ERROR               = 10,
    HIPTENSORNET_STATUS_INSUFFICIENT_WORKSPACE = 11,
    HIPTENSORNET_STATUS_INSUFFICIENT_DRIVER    = 12,
    HIPTENSORNET_STATUS_INTERRUPTED            = 13,
} hiptensornet_status;

typedef enum { HIPTENSORNET_R_32F = 0, HIPTENSORNET_R_64F = 1,
               HIPTENSORNET_C_32F = 2, HIPTENSORNET_C_64F = 3,
               HIPTENSORNET_R_16F = 4, HIPTENSORNET_R_16BF = 5 } hiptensornet_data_type;
typedef enum { HIPTENSORNET_COMPUTE_DEFAULT = 0,
               HIPTENSORNET_COMPUTE_32F = 1, HIPTENSORNET_COMPUTE_64F = 2,
               HIPTENSORNET_COMPUTE_16F = 3, HIPTENSORNET_COMPUTE_16BF = 4,
               HIPTENSORNET_COMPUTE_TF32 = 5 } hiptensornet_compute_type;
typedef enum { HIPTENSORNET_MEMSPACE_DEVICE = 0, HIPTENSORNET_MEMSPACE_HOST = 1 } hiptensornet_memspace;
typedef enum { HIPTENSORNET_WORKSPACE_SCRATCH = 0, HIPTENSORNET_WORKSPACE_CACHE = 1 } hiptensornet_workspace_kind;
typedef enum { HIPTENSORNET_WORKSIZE_PREF_MIN = 0,
               HIPTENSORNET_WORKSIZE_PREF_RECOMMENDED = 1,
               HIPTENSORNET_WORKSIZE_PREF_MAX = 2 } hiptensornet_worksize_pref;
typedef enum { HIPTENSORNET_STATE_PURITY_PURE = 0, HIPTENSORNET_STATE_PURITY_MIXED = 1 } hiptensornet_state_purity;

typedef int64_t hiptensornet_index_t;

/* ---- Opaque handles ---- */
typedef struct hiptensornet_handle_st*                       hiptensornet_handle;
typedef struct hiptensornet_network_descriptor_st*           hiptensornet_network_descriptor;
typedef struct hiptensornet_tensor_descriptor_st*            hiptensornet_tensor_descriptor;
typedef struct hiptensornet_contraction_optimizer_config_st* hiptensornet_contraction_optimizer_config;
typedef struct hiptensornet_contraction_optimizer_info_st*   hiptensornet_contraction_optimizer_info;
typedef struct hiptensornet_workspace_descriptor_st*         hiptensornet_workspace_descriptor;
typedef struct hiptensornet_contraction_plan_st*             hiptensornet_contraction_plan;
typedef struct hiptensornet_contraction_autotune_preference_st*
        hiptensornet_contraction_autotune_preference;
typedef struct hiptensornet_slice_group_st*                  hiptensornet_slice_group;
typedef struct hiptensornet_tensor_svd_config_st*            hiptensornet_tensor_svd_config;
typedef struct hiptensornet_tensor_svd_info_st*              hiptensornet_tensor_svd_info;
typedef struct hiptensornet_network_operator_st*             hiptensornet_network_operator;
typedef struct hiptensornet_state_st*                        hiptensornet_state;
typedef struct hiptensornet_state_marginal_st*               hiptensornet_state_marginal;
typedef struct hiptensornet_state_sampler_st*                hiptensornet_state_sampler;
typedef struct hiptensornet_state_expectation_st*            hiptensornet_state_expectation;
typedef struct hiptensornet_state_accessor_st*               hiptensornet_state_accessor;

typedef enum { HIPTENSORNET_NETWORK_INPUT_TENSORS_DATA_TYPES  = 0,
               HIPTENSORNET_NETWORK_OUTPUT_TENSOR_DATA_TYPE   = 1,
               HIPTENSORNET_NETWORK_COMPUTE_TYPE              = 2,
               HIPTENSORNET_NETWORK_INPUT_TENSORS_NUM_MODES   = 3,
               HIPTENSORNET_NETWORK_OUTPUT_TENSOR_NUM_MODES   = 4,
               HIPTENSORNET_NETWORK_INPUT_TENSORS_NUM         = 5
             } hiptensornet_network_attribute;

typedef enum {
    HIPTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_NUM_GRAPH_ITERATIONS        = 0,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_NUM_GRAPH_CUTS              = 1,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM             = 2,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS     = 3,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_LEAVES         = 4,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING      = 5,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL         = 6,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_FACTOR        = 7,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED                        = 8,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE     = 9,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_CACHE_REUSE_NRUNS           = 10,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION                = 11,
} hiptensornet_contraction_optimizer_config_attribute;

typedef enum {
    HIPTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES         = 0,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES   = 1,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE        = 2,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT      = 3,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH               = 4,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT         = 5,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR     = 6,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_INFO_EFFECTIVE_FLOPS    = 7,
    HIPTENSORNET_CONTRACTION_OPTIMIZER_INFO_INTERMEDIATE_MODES = 8,
} hiptensornet_contraction_optimizer_info_attribute;

typedef enum {
    HIPTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS     = 0,
    HIPTENSORNET_CONTRACTION_AUTOTUNE_INTERMEDIATE_MODES = 1,
    HIPTENSORNET_CONTRACTION_AUTOTUNE_GEMM_ALGORITHM     = 2,
} hiptensornet_contraction_autotune_preference_attribute;

typedef enum {
    HIPTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF              = 0,
    HIPTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF              = 1,
    HIPTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION         = 2,
    HIPTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION             = 3,
    HIPTENSORNET_TENSOR_SVD_CONFIG_ALGO                    = 4,
    HIPTENSORNET_TENSOR_SVD_CONFIG_MAX_EXTENT              = 5,
    HIPTENSORNET_TENSOR_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF = 6,
} hiptensornet_tensor_svd_config_attribute;

typedef enum {
    HIPTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT      = 0,
    HIPTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT   = 1,
    HIPTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT = 2,
    HIPTENSORNET_TENSOR_SVD_INFO_ALGO             = 3,
    HIPTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS      = 4,
} hiptensornet_tensor_svd_info_attribute;

typedef enum {
    HIPTENSORNET_STATE_CONFIG_NUM_HYPER_SAMPLES              = 0,
    HIPTENSORNET_STATE_CONFIG_MPS_CANONICAL_CENTER           = 1,
    HIPTENSORNET_STATE_CONFIG_MPS_SVD_CONFIG_ABS_CUTOFF      = 2,
    HIPTENSORNET_STATE_CONFIG_MPS_SVD_CONFIG_REL_CUTOFF      = 3,
    HIPTENSORNET_STATE_CONFIG_MPS_SVD_CONFIG_S_NORMALIZATION = 4,
    HIPTENSORNET_STATE_CONFIG_MPS_SVD_CONFIG_S_PARTITION     = 5,
    HIPTENSORNET_STATE_CONFIG_MPS_SVD_CONFIG_ALGO            = 6,
    HIPTENSORNET_STATE_CONFIG_MPS_MPO_APPLICATION            = 7,
    HIPTENSORNET_STATE_CONFIG_MPS_GAUGE_OPTION               = 8,
    HIPTENSORNET_STATE_INFO_FLOPS                            = 9,
} hiptensornet_state_attribute;

typedef enum {
    HIPTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES = 0,
    HIPTENSORNET_MARGINAL_INFO_FLOPS               = 1,
} hiptensornet_marginal_attribute;

typedef enum {
    HIPTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES = 0,
    HIPTENSORNET_SAMPLER_CONFIG_DETERMINISTIC     = 1,
    HIPTENSORNET_SAMPLER_INFO_FLOPS               = 2,
} hiptensornet_sampler_attribute;

typedef enum {
    HIPTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES = 0,
    HIPTENSORNET_EXPECTATION_INFO_FLOPS               = 1,
} hiptensornet_expectation_attribute;

typedef enum {
    HIPTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES = 0,
    HIPTENSORNET_ACCESSOR_INFO_FLOPS               = 1,
} hiptensornet_accessor_attribute;

typedef struct hiptensornet_device_mem_handler {
    void* ctx;
    int (*device_alloc)(void* ctx, void** ptr, size_t size, hipStream_t stream);
    int (*device_free) (void* ctx, void* ptr, size_t size, hipStream_t stream);
    char name[64];
} hiptensornet_device_mem_handler_t;

typedef void (*hiptensornet_logger_callback_t)(int32_t logLevel, const char* fn, const char* msg);
typedef void (*hiptensornet_logger_callback_data_t)(int32_t logLevel, const char* fn, const char* msg, void* user);

/* ============================== HANDLE / LIBRARY ============================== */
hiptensornet_status hiptensornet_create                     (hiptensornet_handle* handle);
hiptensornet_status hiptensornet_destroy                    (hiptensornet_handle  handle);
hiptensornet_status hiptensornet_get_version                (int* version);
hiptensornet_status hiptensornet_get_hip_runtime_version    (int* version);
const char*         hiptensornet_get_error_string           (hiptensornet_status status);
hiptensornet_status hiptensornet_set_stream                 (hiptensornet_handle h, hipStream_t s);
hiptensornet_status hiptensornet_get_stream                 (hiptensornet_handle h, hipStream_t* s);
hiptensornet_status hiptensornet_set_device_mem_handler     (hiptensornet_handle h, const hiptensornet_device_mem_handler_t* m);
hiptensornet_status hiptensornet_get_device_mem_handler     (hiptensornet_handle h, hiptensornet_device_mem_handler_t* out);
hiptensornet_status hiptensornet_logger_set_callback        (hiptensornet_logger_callback_t cb);
hiptensornet_status hiptensornet_logger_set_callback_data   (hiptensornet_logger_callback_data_t cb, void* user);
hiptensornet_status hiptensornet_logger_set_file            (void* file);
hiptensornet_status hiptensornet_logger_open_file           (const char* file_name);
hiptensornet_status hiptensornet_logger_set_level           (int32_t level);
hiptensornet_status hiptensornet_logger_set_mask            (int32_t mask);
hiptensornet_status hiptensornet_logger_force_disable       (void);

/* ============================== NETWORK / TENSOR DESCRIPTOR ============================== */
hiptensornet_status
hiptensornet_create_network_descriptor(hiptensornet_handle h, int32_t num_inputs,
                                       const int32_t* num_modes_in,
                                       const hiptensornet_index_t* const* extents_in,
                                       const hiptensornet_index_t* const* strides_in,
                                       const int32_t* const* modes_in,
                                       const uint32_t* alignments_in,
                                       int32_t num_modes_out,
                                       const hiptensornet_index_t* extents_out,
                                       const hiptensornet_index_t* strides_out,
                                       const int32_t* modes_out,
                                       uint32_t alignments_out,
                                       hiptensornet_data_type dt,
                                       hiptensornet_compute_type ct,
                                       hiptensornet_network_descriptor* out);
hiptensornet_status
hiptensornet_destroy_network_descriptor(hiptensornet_network_descriptor desc);
hiptensornet_status
hiptensornet_get_output_tensor_descriptor(hiptensornet_handle h,
                                          hiptensornet_network_descriptor desc,
                                          hiptensornet_tensor_descriptor* out);
hiptensornet_status
hiptensornet_create_tensor_descriptor(hiptensornet_handle h,
                                      int32_t num_modes,
                                      const hiptensornet_index_t* extents,
                                      const hiptensornet_index_t* strides,
                                      const int32_t* modes,
                                      hiptensornet_data_type dt,
                                      hiptensornet_tensor_descriptor* out);
hiptensornet_status hiptensornet_destroy_tensor_descriptor(hiptensornet_tensor_descriptor d);

hiptensornet_status
hiptensornet_get_tensor_details(hiptensornet_handle h, hiptensornet_tensor_descriptor d,
                                int32_t* num_modes, size_t* data_size_bytes,
                                hiptensornet_data_type* dt,
                                int32_t* modes, hiptensornet_index_t* extents,
                                hiptensornet_index_t* strides);
hiptensornet_status
hiptensornet_network_get_attribute(hiptensornet_handle h, hiptensornet_network_descriptor d,
                                   hiptensornet_network_attribute a, void* v, size_t s);
hiptensornet_status
hiptensornet_network_set_attribute(hiptensornet_handle h, hiptensornet_network_descriptor d,
                                   hiptensornet_network_attribute a, const void* v, size_t s);

/* ============================== OPTIMIZER ============================== */
hiptensornet_status hiptensornet_create_contraction_optimizer_config (hiptensornet_handle h, hiptensornet_contraction_optimizer_config* o);
hiptensornet_status hiptensornet_destroy_contraction_optimizer_config(hiptensornet_contraction_optimizer_config c);
hiptensornet_status hiptensornet_contraction_optimizer_config_get_attribute(hiptensornet_handle h, hiptensornet_contraction_optimizer_config c, hiptensornet_contraction_optimizer_config_attribute a, void* v, size_t s);
hiptensornet_status hiptensornet_contraction_optimizer_config_set_attribute(hiptensornet_handle h, hiptensornet_contraction_optimizer_config c, hiptensornet_contraction_optimizer_config_attribute a, const void* v, size_t s);
hiptensornet_status hiptensornet_create_contraction_optimizer_info   (hiptensornet_handle h, hiptensornet_network_descriptor d, hiptensornet_contraction_optimizer_info* o);
hiptensornet_status hiptensornet_destroy_contraction_optimizer_info  (hiptensornet_contraction_optimizer_info i);
hiptensornet_status hiptensornet_contraction_optimizer_info_get_attribute(hiptensornet_handle h, hiptensornet_contraction_optimizer_info i, hiptensornet_contraction_optimizer_info_attribute a, void* v, size_t s);
hiptensornet_status hiptensornet_contraction_optimizer_info_set_attribute(hiptensornet_handle h, hiptensornet_contraction_optimizer_info i, hiptensornet_contraction_optimizer_info_attribute a, const void* v, size_t s);
hiptensornet_status hiptensornet_contraction_optimizer_info_get_packed_size(hiptensornet_handle h, hiptensornet_contraction_optimizer_info i, size_t* out_size);
hiptensornet_status hiptensornet_contraction_optimizer_info_pack_data(hiptensornet_handle h, hiptensornet_contraction_optimizer_info i, void* buf, size_t size);
hiptensornet_status hiptensornet_create_contraction_optimizer_info_from_packed_data(hiptensornet_handle h, hiptensornet_network_descriptor d, const void* buf, size_t size, hiptensornet_contraction_optimizer_info* out);
hiptensornet_status hiptensornet_contraction_optimize                (hiptensornet_handle h, hiptensornet_network_descriptor d, hiptensornet_contraction_optimizer_config c, uint64_t ws, hiptensornet_contraction_optimizer_info i);

/* ============================== WORKSPACE ============================== */
hiptensornet_status hiptensornet_create_workspace_descriptor             (hiptensornet_handle h, hiptensornet_workspace_descriptor* o);
hiptensornet_status hiptensornet_destroy_workspace_descriptor            (hiptensornet_workspace_descriptor d);
hiptensornet_status hiptensornet_workspace_compute_contraction_sizes     (hiptensornet_handle h, hiptensornet_network_descriptor d, hiptensornet_contraction_optimizer_info i, hiptensornet_workspace_descriptor w);
hiptensornet_status hiptensornet_workspace_compute_svd_sizes             (hiptensornet_handle h, hiptensornet_tensor_descriptor din, hiptensornet_tensor_descriptor du, hiptensornet_tensor_descriptor dv, hiptensornet_tensor_svd_config c, hiptensornet_workspace_descriptor w);
hiptensornet_status hiptensornet_workspace_compute_qr_sizes              (hiptensornet_handle h, hiptensornet_tensor_descriptor din, hiptensornet_tensor_descriptor dq, hiptensornet_tensor_descriptor dr, hiptensornet_workspace_descriptor w);
hiptensornet_status hiptensornet_workspace_get_memory_size               (hiptensornet_handle h, hiptensornet_workspace_descriptor d, hiptensornet_worksize_pref p, hiptensornet_memspace m, hiptensornet_workspace_kind k, int64_t* s);
hiptensornet_status hiptensornet_workspace_set_memory                    (hiptensornet_handle h, hiptensornet_workspace_descriptor d, hiptensornet_memspace m, hiptensornet_workspace_kind k, void* buf, int64_t size);
hiptensornet_status hiptensornet_workspace_get_memory                    (hiptensornet_handle h, hiptensornet_workspace_descriptor d, hiptensornet_memspace m, hiptensornet_workspace_kind k, void** buf, int64_t* size);
hiptensornet_status hiptensornet_workspace_purge_cache                   (hiptensornet_handle h, hiptensornet_workspace_descriptor d, hiptensornet_memspace m);

/* ============================== CONTRACTION ============================== */
hiptensornet_status hiptensornet_create_contraction_plan      (hiptensornet_handle h, hiptensornet_network_descriptor d, hiptensornet_contraction_optimizer_info i, hiptensornet_workspace_descriptor w, hiptensornet_contraction_plan* p);
hiptensornet_status hiptensornet_destroy_contraction_plan     (hiptensornet_contraction_plan p);
hiptensornet_status hiptensornet_contraction                  (hiptensornet_handle h, hiptensornet_contraction_plan p, const void* const* in, void* out, hiptensornet_workspace_descriptor w, int64_t slice, hipStream_t s);
hiptensornet_status hiptensornet_contract_slices              (hiptensornet_handle h, hiptensornet_contraction_plan p, const void* const* in, void* out, int32_t acc, hiptensornet_workspace_descriptor w, hiptensornet_slice_group g, hipStream_t s);
hiptensornet_status hiptensornet_create_contraction_autotune_preference (hiptensornet_handle h, hiptensornet_contraction_autotune_preference* o);
hiptensornet_status hiptensornet_destroy_contraction_autotune_preference(hiptensornet_contraction_autotune_preference o);
hiptensornet_status hiptensornet_contraction_autotune_preference_get_attribute(hiptensornet_handle h, hiptensornet_contraction_autotune_preference p, hiptensornet_contraction_autotune_preference_attribute a, void* v, size_t s);
hiptensornet_status hiptensornet_contraction_autotune_preference_set_attribute(hiptensornet_handle h, hiptensornet_contraction_autotune_preference p, hiptensornet_contraction_autotune_preference_attribute a, const void* v, size_t s);
hiptensornet_status hiptensornet_contraction_autotune                   (hiptensornet_handle h, hiptensornet_contraction_plan p, const void* const* in, void* out, hiptensornet_workspace_descriptor w, hiptensornet_contraction_autotune_preference pref, hipStream_t s);

/* ============================== SLICE GROUP ============================== */
hiptensornet_status hiptensornet_create_slice_group_from_id_range(hiptensornet_handle h, int64_t start, int64_t stop, int64_t step, hiptensornet_slice_group* o);
hiptensornet_status hiptensornet_create_slice_group_from_ids     (hiptensornet_handle h, const int64_t* ids, int32_t n, hiptensornet_slice_group* o);
hiptensornet_status hiptensornet_destroy_slice_group             (hiptensornet_slice_group g);

/* ============================== TENSOR SVD / QR ============================== */
hiptensornet_status hiptensornet_create_tensor_svd_config (hiptensornet_handle h, hiptensornet_tensor_svd_config* o);
hiptensornet_status hiptensornet_destroy_tensor_svd_config(hiptensornet_tensor_svd_config c);
hiptensornet_status hiptensornet_tensor_svd_config_get_attribute(hiptensornet_handle h, hiptensornet_tensor_svd_config c, hiptensornet_tensor_svd_config_attribute a, void* v, size_t s);
hiptensornet_status hiptensornet_tensor_svd_config_set_attribute(hiptensornet_handle h, hiptensornet_tensor_svd_config c, hiptensornet_tensor_svd_config_attribute a, const void* v, size_t s);
hiptensornet_status hiptensornet_create_tensor_svd_info   (hiptensornet_handle h, hiptensornet_tensor_svd_info* o);
hiptensornet_status hiptensornet_destroy_tensor_svd_info  (hiptensornet_tensor_svd_info i);
hiptensornet_status hiptensornet_tensor_svd_info_get_attribute(hiptensornet_handle h, hiptensornet_tensor_svd_info i, hiptensornet_tensor_svd_info_attribute a, void* v, size_t s);
hiptensornet_status hiptensornet_tensor_svd               (hiptensornet_handle h, hiptensornet_tensor_descriptor din, const void* in, hiptensornet_tensor_descriptor du, void* u, void* s, hiptensornet_tensor_descriptor dv, void* v, hiptensornet_tensor_svd_config c, hiptensornet_tensor_svd_info i, hiptensornet_workspace_descriptor w, hipStream_t st);
hiptensornet_status hiptensornet_tensor_qr                (hiptensornet_handle h, hiptensornet_tensor_descriptor din, const void* in, hiptensornet_tensor_descriptor dq, void* q, hiptensornet_tensor_descriptor dr, void* r, hiptensornet_workspace_descriptor w, hipStream_t st);

/* ============================== GRADIENT ============================== */
hiptensornet_status hiptensornet_compute_gradients_backward(hiptensornet_handle h, hiptensornet_contraction_plan p, const void* const* in, const void* dout, void* const* dins, int32_t acc, hiptensornet_workspace_descriptor w, hipStream_t s);

/* ============================== NETWORK OPERATOR ============================== */
hiptensornet_status hiptensornet_create_network_operator     (hiptensornet_handle h, int32_t nsm, const hiptensornet_index_t* ext, hiptensornet_data_type dt, hiptensornet_network_operator* o);
hiptensornet_status hiptensornet_destroy_network_operator    (hiptensornet_network_operator op);
hiptensornet_status hiptensornet_network_operator_append_product(hiptensornet_handle h, hiptensornet_network_operator op, const void* coeff, int32_t nt, const int32_t* nm, const int32_t* const* modes, const hiptensornet_index_t* const* strides, const void* const* data, int64_t* cid);
hiptensornet_status hiptensornet_network_operator_append_mpo(hiptensornet_handle h, hiptensornet_network_operator op, const void* coefficient, int32_t num_state_modes, const int32_t* state_modes, const int32_t* tensor_mode_extents, const int64_t* tensor_mode_strides, const void* const* tensor_data, int32_t boundary_condition, int64_t* component_id);

/* ============================== NETWORK STATE ============================== */
hiptensornet_status hiptensornet_create_state                (hiptensornet_handle h, hiptensornet_state_purity p, int32_t nsm, const hiptensornet_index_t* ext, hiptensornet_data_type dt, hiptensornet_state* o);
hiptensornet_status hiptensornet_destroy_state               (hiptensornet_state s);
hiptensornet_status hiptensornet_state_apply_tensor_operator (hiptensornet_handle h, hiptensornet_state s, int32_t nsm, const int32_t* sm, void* td, const int64_t* ts, int32_t im, int32_t adj, int32_t u, int64_t* tid);
hiptensornet_status hiptensornet_state_apply_controlled_tensor_operator(hiptensornet_handle h, hiptensornet_state s, int32_t num_control_modes, const int32_t* control_modes, const int64_t* control_values, int32_t num_target_modes, const int32_t* target_modes, void* tensor_data, const int64_t* tensor_mode_strides, int32_t immutable, int32_t adjoint, int32_t unitary, int64_t* tensor_id);
hiptensornet_status hiptensornet_state_apply_unitary_channel (hiptensornet_handle h, hiptensornet_state s, int32_t num_state_modes, const int32_t* state_modes, int32_t num_tensors, void* const* tensor_data, const int64_t* const* tensor_mode_strides, const double* probabilities, int64_t* channel_id);
hiptensornet_status hiptensornet_state_apply_general_channel (hiptensornet_handle h, hiptensornet_state s, int32_t num_state_modes, const int32_t* state_modes, int32_t num_tensors, void* const* tensor_data, const int64_t* const* tensor_mode_strides, int64_t* channel_id);
hiptensornet_status hiptensornet_state_update_tensor_operator(hiptensornet_handle h, hiptensornet_state s, int64_t tensor_id, void* tensor_data, int32_t unitary);
hiptensornet_status hiptensornet_state_configure             (hiptensornet_handle h, hiptensornet_state s, hiptensornet_state_attribute a, const void* v, size_t sz);
hiptensornet_status hiptensornet_state_get_info              (hiptensornet_handle h, hiptensornet_state s, hiptensornet_state_attribute a, void* v, size_t sz);
hiptensornet_status hiptensornet_state_prepare               (hiptensornet_handle h, hiptensornet_state s, size_t max_workspace_size_device, hiptensornet_workspace_descriptor workspace, hipStream_t stream);
hiptensornet_status hiptensornet_state_compute               (hiptensornet_handle h, hiptensornet_state s, hiptensornet_workspace_descriptor workspace, void* const* state_tensors_out, hipStream_t stream);

/* ============================== DERIVED ============================== */
hiptensornet_status hiptensornet_create_marginal             (hiptensornet_handle h, hiptensornet_state s, int32_t nm, const int32_t* mm, int32_t npm, const int32_t* pm, const int64_t* mts, hiptensornet_state_marginal* o);
hiptensornet_status hiptensornet_destroy_marginal            (hiptensornet_state_marginal m);
hiptensornet_status hiptensornet_marginal_configure          (hiptensornet_handle h, hiptensornet_state_marginal m, hiptensornet_marginal_attribute a, const void* v, size_t sz);
hiptensornet_status hiptensornet_marginal_get_info           (hiptensornet_handle h, hiptensornet_state_marginal m, hiptensornet_marginal_attribute a, void* v, size_t sz);
hiptensornet_status hiptensornet_marginal_prepare            (hiptensornet_handle h, hiptensornet_state_marginal m, size_t max_workspace_size_device, hiptensornet_workspace_descriptor workspace, hipStream_t stream);
hiptensornet_status hiptensornet_marginal_compute            (hiptensornet_handle h, hiptensornet_state_marginal m, const int64_t* projected_mode_values, hiptensornet_workspace_descriptor workspace, void* marginal_tensor, hipStream_t stream);
hiptensornet_status hiptensornet_create_sampler              (hiptensornet_handle h, hiptensornet_state s, int32_t n, const int32_t* mts, hiptensornet_state_sampler* o);
hiptensornet_status hiptensornet_destroy_sampler             (hiptensornet_state_sampler s);
hiptensornet_status hiptensornet_sampler_configure           (hiptensornet_handle h, hiptensornet_state_sampler s, hiptensornet_sampler_attribute a, const void* v, size_t sz);
hiptensornet_status hiptensornet_sampler_get_info            (hiptensornet_handle h, hiptensornet_state_sampler s, hiptensornet_sampler_attribute a, void* v, size_t sz);
hiptensornet_status hiptensornet_sampler_prepare             (hiptensornet_handle h, hiptensornet_state_sampler s, size_t max_workspace_size_device, hiptensornet_workspace_descriptor workspace, hipStream_t stream);
hiptensornet_status hiptensornet_sampler_sample              (hiptensornet_handle h, hiptensornet_state_sampler s, int64_t num_shots, hiptensornet_workspace_descriptor workspace, int64_t* samples, hipStream_t stream);
hiptensornet_status hiptensornet_create_expectation          (hiptensornet_handle h, hiptensornet_state s, hiptensornet_network_operator op, hiptensornet_state_expectation* o);
hiptensornet_status hiptensornet_destroy_expectation         (hiptensornet_state_expectation e);
hiptensornet_status hiptensornet_expectation_configure       (hiptensornet_handle h, hiptensornet_state_expectation e, hiptensornet_expectation_attribute a, const void* v, size_t sz);
hiptensornet_status hiptensornet_expectation_get_info        (hiptensornet_handle h, hiptensornet_state_expectation e, hiptensornet_expectation_attribute a, void* v, size_t sz);
hiptensornet_status hiptensornet_expectation_prepare         (hiptensornet_handle h, hiptensornet_state_expectation e, size_t max_workspace_size_device, hiptensornet_workspace_descriptor workspace, hipStream_t stream);
hiptensornet_status hiptensornet_expectation_compute         (hiptensornet_handle h, hiptensornet_state_expectation e, hiptensornet_workspace_descriptor workspace, void* expectation_value, void* state_norm, hipStream_t stream);
hiptensornet_status hiptensornet_create_accessor             (hiptensornet_handle h, hiptensornet_state s, int32_t npm, const int32_t* pm, const int64_t* ats, hiptensornet_state_accessor* o);
hiptensornet_status hiptensornet_destroy_accessor            (hiptensornet_state_accessor a);
hiptensornet_status hiptensornet_accessor_configure          (hiptensornet_handle h, hiptensornet_state_accessor a, hiptensornet_accessor_attribute attr, const void* v, size_t sz);
hiptensornet_status hiptensornet_accessor_get_info           (hiptensornet_handle h, hiptensornet_state_accessor a, hiptensornet_accessor_attribute attr, void* v, size_t sz);
hiptensornet_status hiptensornet_accessor_prepare            (hiptensornet_handle h, hiptensornet_state_accessor a, size_t max_workspace_size_device, hiptensornet_workspace_descriptor workspace, hipStream_t stream);
hiptensornet_status hiptensornet_accessor_compute            (hiptensornet_handle h, hiptensornet_state_accessor a, const int64_t* projected_mode_values, hiptensornet_workspace_descriptor workspace, void* amplitudes_tensor, void* state_norm, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* HIPTENSORNET_H */
