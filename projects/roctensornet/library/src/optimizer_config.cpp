/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Contraction optimizer config: lifecycle + attribute round-trip.
 *
 * The attribute store is a plain C++ struct (`contraction_optimizer_config_st`)
 * defined in `roctensornet_descriptors.hpp`; setters/getters validate
 * sizes and copy in/out of named fields.
 * ************************************************************************ */

#include "roctensornet_descriptors.hpp"

using namespace roctensornet;

extern "C" {

roctensornet_status
roctensornet_create_contraction_optimizer_config(
    roctensornet_handle h,
    roctensornet_contraction_optimizer_config* out)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    ROCTENSORNET_CHECK_PTR(out);
    auto* c = new (std::nothrow) contraction_optimizer_config_st();
    if(c == nullptr) return ROCTENSORNET_STATUS_ALLOC_FAILED;
    *out = reinterpret_cast<roctensornet_contraction_optimizer_config>(c);
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_destroy_contraction_optimizer_config(
    roctensornet_contraction_optimizer_config c)
{
    delete cast<contraction_optimizer_config_st>(c);
    return ROCTENSORNET_STATUS_SUCCESS;
}

#define READ_INT32(field)                                                    \
    do {                                                                     \
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE; \
        *static_cast<int32_t*>(value) = cc->field;                           \
        return ROCTENSORNET_STATUS_SUCCESS;                                  \
    } while(0)

#define WRITE_INT32(field)                                                   \
    do {                                                                     \
        if(size < sizeof(int32_t)) return ROCTENSORNET_STATUS_INVALID_VALUE; \
        cc->field = *static_cast<const int32_t*>(value);                     \
        return ROCTENSORNET_STATUS_SUCCESS;                                  \
    } while(0)

roctensornet_status
roctensornet_contraction_optimizer_config_get_attribute(
    roctensornet_handle                                  h,
    roctensornet_contraction_optimizer_config            c,
    roctensornet_contraction_optimizer_config_attribute  attr,
    void*                                                value,
    size_t                                               size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* cc = cast<contraction_optimizer_config_st>(c);
    if(cc == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    switch(attr)
    {
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_NUM_GRAPH_ITERATIONS:    READ_INT32(num_graph_iterations);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_NUM_GRAPH_CUTS:          READ_INT32(num_graph_cuts);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM:         READ_INT32(graph_algorithm);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS: READ_INT32(reconfig_num_iterations);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_LEAVES:     READ_INT32(reconfig_num_leaves);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING:  READ_INT32(slicer_disable_slicing);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL:     READ_INT32(slicer_memory_model);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_FACTOR:
        if(size < sizeof(double)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        *static_cast<double*>(value) = cc->slicer_memory_factor;
        return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED:                    READ_INT32(seed);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE: READ_INT32(cost_function_objective);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_CACHE_REUSE_NRUNS:       READ_INT32(cache_reuse_nruns);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION:            READ_INT32(smart_option);
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

roctensornet_status
roctensornet_contraction_optimizer_config_set_attribute(
    roctensornet_handle                                  h,
    roctensornet_contraction_optimizer_config            c,
    roctensornet_contraction_optimizer_config_attribute  attr,
    const void*                                          value,
    size_t                                               size)
{
    ROCTENSORNET_CHECK_HANDLE(h);
    auto* cc = cast<contraction_optimizer_config_st>(c);
    if(cc == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;
    ROCTENSORNET_CHECK_PTR(value);
    switch(attr)
    {
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_NUM_GRAPH_ITERATIONS:    WRITE_INT32(num_graph_iterations);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_NUM_GRAPH_CUTS:          WRITE_INT32(num_graph_cuts);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM:         WRITE_INT32(graph_algorithm);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS: WRITE_INT32(reconfig_num_iterations);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_LEAVES:     WRITE_INT32(reconfig_num_leaves);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING:  WRITE_INT32(slicer_disable_slicing);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL:     WRITE_INT32(slicer_memory_model);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_FACTOR:
        if(size < sizeof(double)) return ROCTENSORNET_STATUS_INVALID_VALUE;
        cc->slicer_memory_factor = *static_cast<const double*>(value);
        return ROCTENSORNET_STATUS_SUCCESS;
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED:                    WRITE_INT32(seed);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE: WRITE_INT32(cost_function_objective);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_CACHE_REUSE_NRUNS:       WRITE_INT32(cache_reuse_nruns);
    case ROCTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION:            WRITE_INT32(smart_option);
    }
    return ROCTENSORNET_STATUS_NOT_SUPPORTED;
}

#undef READ_INT32
#undef WRITE_INT32

} // extern "C"
