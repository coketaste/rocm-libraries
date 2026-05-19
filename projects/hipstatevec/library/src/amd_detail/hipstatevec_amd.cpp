/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * AMD backend for hipSTATEVEC: pure pass-through to rocSTATEVEC.
 *
 * Every function in this file casts opaque hipSTATEVEC handles to their
 * rocSTATEVEC equivalents (which are layout-identical because they are
 * the same opaque struct in different headers) and forwards. Status
 * codes use the same numeric values; index pairs are layout-compatible.
 * ************************************************************************ */

#include <hipstatevec.h>
#include <rocstatevec.h>

#include <cstring>

namespace
{
inline rocstatevec_handle to_roc(hipstatevecHandle_t h)
{
    return reinterpret_cast<rocstatevec_handle>(h);
}
inline rocstatevec_sampler_descriptor to_roc_s(hipstatevecSamplerDescriptor_t s)
{
    return reinterpret_cast<rocstatevec_sampler_descriptor>(s);
}
inline rocstatevec_accessor_descriptor to_roc_a(hipstatevecAccessorDescriptor_t a)
{
    return reinterpret_cast<rocstatevec_accessor_descriptor>(a);
}
inline hipstatevecStatus_t hipify(rocstatevec_status s)
{
    return static_cast<hipstatevecStatus_t>(s);
}
} // namespace

extern "C" {

/* ----- handle / version / property ----- */

hipstatevecStatus_t hipstatevecCreate(hipstatevecHandle_t* handle)
{
    rocstatevec_handle h = nullptr;
    auto rc = rocstatevec_create_handle(&h);
    *handle = reinterpret_cast<hipstatevecHandle_t>(h);
    return hipify(rc);
}

hipstatevecStatus_t hipstatevecDestroy(hipstatevecHandle_t handle)
{
    return hipify(rocstatevec_destroy_handle(to_roc(handle)));
}

hipstatevecStatus_t hipstatevecGetVersion(hipstatevecHandle_t handle, int* version)
{
    return hipify(rocstatevec_get_version(to_roc(handle), version));
}

hipstatevecStatus_t hipstatevecGetProperty(hipstatevecLibraryPropertyType_t type, int* value)
{
    return hipify(rocstatevec_get_property(static_cast<rocstatevec_library_property_type>(type), value));
}

hipstatevecStatus_t hipstatevecSetStream(hipstatevecHandle_t handle, hipStream_t stream)
{
    return hipify(rocstatevec_set_stream(to_roc(handle), stream));
}

hipstatevecStatus_t hipstatevecGetStream(hipstatevecHandle_t handle, hipStream_t* stream)
{
    return hipify(rocstatevec_get_stream(to_roc(handle), stream));
}

const char* hipstatevecGetErrorName(hipstatevecStatus_t status)
{
    return rocstatevec_get_error_name(static_cast<rocstatevec_status>(status));
}

const char* hipstatevecGetErrorString(hipstatevecStatus_t status)
{
    return rocstatevec_get_error_string(static_cast<rocstatevec_status>(status));
}

hipstatevecStatus_t hipstatevecSetDeviceMemHandler(
    hipstatevecHandle_t handle, const hipstatevecDeviceMemHandler_t* mem_handler)
{
    if(mem_handler == nullptr)
        return hipify(rocstatevec_set_device_mem_handler(to_roc(handle), nullptr));

    rocstatevec_device_mem_handler_t roc{};
    roc.ctx          = mem_handler->ctx;
    roc.device_alloc = mem_handler->device_alloc;
    roc.device_free  = mem_handler->device_free;
    std::memcpy(roc.name, mem_handler->name, sizeof(roc.name));
    return hipify(rocstatevec_set_device_mem_handler(to_roc(handle), &roc));
}

hipstatevecStatus_t hipstatevecGetDeviceMemHandler(
    hipstatevecHandle_t handle, hipstatevecDeviceMemHandler_t* mem_handler)
{
    rocstatevec_device_mem_handler_t roc{};
    auto rc = rocstatevec_get_device_mem_handler(to_roc(handle), &roc);
    if(rc != ROCSTATEVEC_STATUS_SUCCESS) return hipify(rc);
    if(mem_handler == nullptr) return HIPSTATEVEC_STATUS_INVALID_VALUE;
    mem_handler->ctx          = roc.ctx;
    mem_handler->device_alloc = roc.device_alloc;
    mem_handler->device_free  = roc.device_free;
    std::memcpy(mem_handler->name, roc.name, sizeof(mem_handler->name));
    return HIPSTATEVEC_STATUS_SUCCESS;
}

/* ----- logger ----- */

hipstatevecStatus_t hipstatevecLoggerSetCallback(hipstatevecLoggerCallback_t cb)
{
    return hipify(rocstatevec_logger_set_callback(reinterpret_cast<rocstatevec_logger_callback_t>(cb)));
}
hipstatevecStatus_t hipstatevecLoggerSetCallbackData(hipstatevecLoggerCallbackData_t cb, void* user_data)
{
    return hipify(rocstatevec_logger_set_callback_data(
        reinterpret_cast<rocstatevec_logger_callback_data_t>(cb), user_data));
}
hipstatevecStatus_t hipstatevecLoggerSetFile(void* file)         { return hipify(rocstatevec_logger_set_file(file)); }
hipstatevecStatus_t hipstatevecLoggerOpenFile(const char* path)  { return hipify(rocstatevec_logger_open_file(path)); }
hipstatevecStatus_t hipstatevecLoggerSetLevel(int32_t level)     { return hipify(rocstatevec_logger_set_level(level)); }
hipstatevecStatus_t hipstatevecLoggerSetMask(int32_t mask)       { return hipify(rocstatevec_logger_set_mask(mask)); }
hipstatevecStatus_t hipstatevecLoggerForceDisable(void)          { return hipify(rocstatevec_logger_force_disable()); }

/* ----- init ----- */

hipstatevecStatus_t hipstatevecInitializeStateVector(
    hipstatevecHandle_t handle, void* sv, hipstatevecDataType_t dtype,
    uint32_t n, hipstatevecStateVectorType_t kind)
{
    return hipify(rocstatevec_initialize_state_vector(
        to_roc(handle), sv,
        static_cast<rocstatevec_data_type>(dtype), n,
        static_cast<rocstatevec_state_vector_type>(kind)));
}

/* ----- apply ----- */

hipstatevecStatus_t hipstatevecApplyMatrixGetWorkspaceSize(
    hipstatevecHandle_t handle, hipstatevecDataType_t sv_dt, uint32_t n,
    const void* matrix, hipstatevecDataType_t m_dt, hipstatevecMatrixLayout_t layout,
    int32_t adjoint, uint32_t n_targets, uint32_t n_controls,
    hipstatevecComputeType_t ct, size_t* ws)
{
    return hipify(rocstatevec_apply_matrix_get_workspace_size(
        to_roc(handle), static_cast<rocstatevec_data_type>(sv_dt), n,
        matrix, static_cast<rocstatevec_data_type>(m_dt),
        static_cast<rocstatevec_matrix_layout>(layout),
        adjoint, n_targets, n_controls,
        static_cast<rocstatevec_compute_type>(ct), ws));
}

hipstatevecStatus_t hipstatevecApplyMatrix(
    hipstatevecHandle_t handle, void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    const void* matrix, hipstatevecDataType_t m_dt, hipstatevecMatrixLayout_t layout,
    int32_t adjoint, const int32_t* targets, uint32_t n_targets,
    const int32_t* controls, const int32_t* control_bit_values, uint32_t n_controls,
    hipstatevecComputeType_t ct, void* ws, size_t ws_bytes)
{
    return hipify(rocstatevec_apply_matrix(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        matrix, static_cast<rocstatevec_data_type>(m_dt),
        static_cast<rocstatevec_matrix_layout>(layout),
        adjoint, targets, n_targets, controls, control_bit_values, n_controls,
        static_cast<rocstatevec_compute_type>(ct), ws, ws_bytes));
}

hipstatevecStatus_t hipstatevecApplyPauliRotation(
    hipstatevecHandle_t handle, void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    double theta, const hipstatevecPauli_t* paulis, const int32_t* targets, uint32_t n_targets,
    const int32_t* controls, const int32_t* control_bit_values, uint32_t n_controls)
{
    static_assert(sizeof(hipstatevecPauli_t) == sizeof(rocstatevec_pauli),
                  "pauli enum sizes must match for reinterpret_cast");
    return hipify(rocstatevec_apply_pauli_rotation(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        theta, reinterpret_cast<const rocstatevec_pauli*>(paulis),
        targets, n_targets, controls, control_bit_values, n_controls));
}

hipstatevecStatus_t hipstatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(
    hipstatevecHandle_t handle, hipstatevecDataType_t sv_dt, uint32_t n,
    const hipstatevecIndex_t* permutation, const void* diag, hipstatevecDataType_t d_dt,
    const int32_t* targets, uint32_t n_targets, uint32_t n_controls, size_t* ws)
{
    return hipify(rocstatevec_apply_generalized_permutation_matrix_get_workspace_size(
        to_roc(handle), static_cast<rocstatevec_data_type>(sv_dt), n,
        reinterpret_cast<const rocstatevec_index_t*>(permutation),
        diag, static_cast<rocstatevec_data_type>(d_dt),
        targets, n_targets, n_controls, ws));
}

hipstatevecStatus_t hipstatevecApplyGeneralizedPermutationMatrix(
    hipstatevecHandle_t handle, void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    const hipstatevecIndex_t* permutation, const void* diag, hipstatevecDataType_t d_dt,
    int32_t adjoint, const int32_t* targets, uint32_t n_targets,
    const int32_t* controls, const int32_t* control_bit_values, uint32_t n_controls,
    void* ws, size_t ws_bytes)
{
    return hipify(rocstatevec_apply_generalized_permutation_matrix(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        reinterpret_cast<const rocstatevec_index_t*>(permutation),
        diag, static_cast<rocstatevec_data_type>(d_dt),
        adjoint, targets, n_targets, controls, control_bit_values, n_controls,
        ws, ws_bytes));
}

hipstatevecStatus_t hipstatevecAbsorbDiagonalMatrix(
    hipstatevecHandle_t handle, void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    const void* diag, hipstatevecDataType_t d_dt, int32_t adjoint,
    const int32_t* targets, uint32_t n_targets,
    const int32_t* controls, const int32_t* control_bit_values, uint32_t n_controls)
{
    return hipify(rocstatevec_absorb_diagonal_matrix(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        diag, static_cast<rocstatevec_data_type>(d_dt),
        adjoint, targets, n_targets, controls, control_bit_values, n_controls));
}

/* ----- batched apply ----- */

hipstatevecStatus_t hipstatevecApplyMatrixBatchedGetWorkspaceSize(
    hipstatevecHandle_t handle, hipstatevecDataType_t sv_dt, uint32_t n,
    uint32_t n_state_vectors, hipstatevecIndex_t state_vector_size,
    hipstatevecMatrixMapType_t map_type, const int32_t* matrix_indices,
    const void* matrices, hipstatevecDataType_t m_dt, hipstatevecMatrixLayout_t layout,
    int32_t adjoint, uint32_t n_matrices, uint32_t n_targets, uint32_t n_controls,
    hipstatevecComputeType_t ct, size_t* ws)
{
    return hipify(rocstatevec_apply_matrix_batched_get_workspace_size(
        to_roc(handle), static_cast<rocstatevec_data_type>(sv_dt), n,
        n_state_vectors, state_vector_size,
        static_cast<rocstatevec_matrix_map_type>(map_type), matrix_indices,
        matrices, static_cast<rocstatevec_data_type>(m_dt),
        static_cast<rocstatevec_matrix_layout>(layout),
        adjoint, n_matrices, n_targets, n_controls,
        static_cast<rocstatevec_compute_type>(ct), ws));
}

hipstatevecStatus_t hipstatevecApplyMatrixBatched(
    hipstatevecHandle_t handle, void* batched_sv, hipstatevecDataType_t sv_dt, uint32_t n,
    uint32_t n_state_vectors, hipstatevecIndex_t state_vector_size,
    hipstatevecMatrixMapType_t map_type, const int32_t* matrix_indices,
    const void* matrices, hipstatevecDataType_t m_dt, hipstatevecMatrixLayout_t layout,
    int32_t adjoint, uint32_t n_matrices, const int32_t* targets, uint32_t n_targets,
    const int32_t* controls, const int32_t* control_bit_values, uint32_t n_controls,
    hipstatevecComputeType_t ct, void* ws, size_t ws_bytes)
{
    return hipify(rocstatevec_apply_matrix_batched(
        to_roc(handle), batched_sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        n_state_vectors, state_vector_size,
        static_cast<rocstatevec_matrix_map_type>(map_type), matrix_indices,
        matrices, static_cast<rocstatevec_data_type>(m_dt),
        static_cast<rocstatevec_matrix_layout>(layout),
        adjoint, n_matrices, targets, n_targets, controls, control_bit_values, n_controls,
        static_cast<rocstatevec_compute_type>(ct), ws, ws_bytes));
}

/* ----- measure / abs2 / collapse / batch_measure ----- */

hipstatevecStatus_t hipstatevecAbs2SumArray(
    hipstatevecHandle_t handle, const void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    double* abs2_sum, const int32_t* bit_ordering, uint32_t bo_len,
    const int32_t* mask_bit_string, const int32_t* mask_ordering, uint32_t mask_len)
{
    return hipify(rocstatevec_abs2_sum_array(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        abs2_sum, bit_ordering, bo_len, mask_bit_string, mask_ordering, mask_len));
}

hipstatevecStatus_t hipstatevecAbs2SumOnZBasis(
    hipstatevecHandle_t handle, const void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    double* s0, double* s1, const int32_t* basis_bits, uint32_t n_bits)
{
    return hipify(rocstatevec_abs2_sum_on_z_basis(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        s0, s1, basis_bits, n_bits));
}

hipstatevecStatus_t hipstatevecMeasureOnZBasis(
    hipstatevecHandle_t handle, void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    int32_t* parity, const int32_t* basis_bits, uint32_t n_bits,
    double randnum, hipstatevecCollapseOp_t collapse)
{
    return hipify(rocstatevec_measure_on_z_basis(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        parity, basis_bits, n_bits, randnum,
        static_cast<rocstatevec_collapse_op>(collapse)));
}

hipstatevecStatus_t hipstatevecBatchMeasure(
    hipstatevecHandle_t handle, void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    int32_t* bit_string, const int32_t* bit_ordering, uint32_t bs_len,
    double randnum, hipstatevecCollapseOp_t collapse)
{
    return hipify(rocstatevec_batch_measure(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        bit_string, bit_ordering, bs_len, randnum,
        static_cast<rocstatevec_collapse_op>(collapse)));
}

hipstatevecStatus_t hipstatevecBatchMeasureWithOffset(
    hipstatevecHandle_t handle, void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    int32_t* bit_string, const int32_t* bit_ordering, uint32_t bs_len,
    double randnum, hipstatevecCollapseOp_t collapse, double offset, double abs2_sum)
{
    return hipify(rocstatevec_batch_measure_with_offset(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        bit_string, bit_ordering, bs_len, randnum,
        static_cast<rocstatevec_collapse_op>(collapse), offset, abs2_sum));
}

hipstatevecStatus_t hipstatevecCollapseOnZBasis(
    hipstatevecHandle_t handle, void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    int32_t parity, const int32_t* basis_bits, uint32_t n_bits, double norm)
{
    return hipify(rocstatevec_collapse_on_z_basis(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        parity, basis_bits, n_bits, norm));
}

hipstatevecStatus_t hipstatevecCollapseByBitString(
    hipstatevecHandle_t handle, void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    const int32_t* bit_string, const int32_t* bit_ordering, uint32_t bs_len, double norm)
{
    return hipify(rocstatevec_collapse_by_bit_string(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        bit_string, bit_ordering, bs_len, norm));
}

/* ----- sampler ----- */

hipstatevecStatus_t hipstatevecSamplerCreate(
    hipstatevecHandle_t handle, const void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    hipstatevecSamplerDescriptor_t* sampler, uint32_t n_max_shots, size_t* ws)
{
    rocstatevec_sampler_descriptor s = nullptr;
    auto rc = rocstatevec_sampler_create(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        &s, n_max_shots, ws);
    *sampler = reinterpret_cast<hipstatevecSamplerDescriptor_t>(s);
    return hipify(rc);
}

hipstatevecStatus_t hipstatevecSamplerDestroy(hipstatevecSamplerDescriptor_t sampler)
{
    return hipify(rocstatevec_sampler_destroy(to_roc_s(sampler)));
}

hipstatevecStatus_t hipstatevecSamplerPreprocess(
    hipstatevecHandle_t handle, hipstatevecSamplerDescriptor_t sampler, void* ws, size_t ws_bytes)
{
    return hipify(rocstatevec_sampler_preprocess(to_roc(handle), to_roc_s(sampler), ws, ws_bytes));
}

hipstatevecStatus_t hipstatevecSamplerGetSquaredNorm(
    hipstatevecHandle_t handle, hipstatevecSamplerDescriptor_t sampler, double* norm)
{
    return hipify(rocstatevec_sampler_get_squared_norm(to_roc(handle), to_roc_s(sampler), norm));
}

hipstatevecStatus_t hipstatevecSamplerApplySubSVOffset(
    hipstatevecHandle_t handle, hipstatevecSamplerDescriptor_t sampler,
    int32_t sub_sv_index, uint32_t n_sub_svs, double offset, double norm)
{
    return hipify(rocstatevec_sampler_apply_sub_sv_offset(
        to_roc(handle), to_roc_s(sampler), sub_sv_index, n_sub_svs, offset, norm));
}

hipstatevecStatus_t hipstatevecSamplerSample(
    hipstatevecHandle_t handle, hipstatevecSamplerDescriptor_t sampler,
    hipstatevecIndex_t* bit_strings, const int32_t* bit_ordering, uint32_t bs_len,
    const double* randnums, uint32_t n_shots, hipstatevecSamplerOutput_t output)
{
    return hipify(rocstatevec_sampler_sample(
        to_roc(handle), to_roc_s(sampler),
        reinterpret_cast<rocstatevec_index_t*>(bit_strings),
        bit_ordering, bs_len, randnums, n_shots,
        static_cast<rocstatevec_sampler_output>(output)));
}

/* ----- expectation ----- */

hipstatevecStatus_t hipstatevecComputeExpectationGetWorkspaceSize(
    hipstatevecHandle_t handle, hipstatevecDataType_t sv_dt, uint32_t n,
    const void* matrix, hipstatevecDataType_t m_dt, hipstatevecMatrixLayout_t layout,
    uint32_t n_basis_bits, hipstatevecComputeType_t ct, size_t* ws)
{
    return hipify(rocstatevec_compute_expectation_get_workspace_size(
        to_roc(handle), static_cast<rocstatevec_data_type>(sv_dt), n,
        matrix, static_cast<rocstatevec_data_type>(m_dt),
        static_cast<rocstatevec_matrix_layout>(layout),
        n_basis_bits, static_cast<rocstatevec_compute_type>(ct), ws));
}

hipstatevecStatus_t hipstatevecComputeExpectation(
    hipstatevecHandle_t handle, const void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    void* expectation_value, hipstatevecDataType_t e_dt, double* residual_norm,
    const void* matrix, hipstatevecDataType_t m_dt, hipstatevecMatrixLayout_t layout,
    const int32_t* basis_bits, uint32_t n_basis_bits, hipstatevecComputeType_t ct,
    void* ws, size_t ws_bytes)
{
    return hipify(rocstatevec_compute_expectation(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        expectation_value, static_cast<rocstatevec_data_type>(e_dt), residual_norm,
        matrix, static_cast<rocstatevec_data_type>(m_dt),
        static_cast<rocstatevec_matrix_layout>(layout),
        basis_bits, n_basis_bits,
        static_cast<rocstatevec_compute_type>(ct), ws, ws_bytes));
}

hipstatevecStatus_t hipstatevecComputeExpectationsOnPauliBasis(
    hipstatevecHandle_t handle, const void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    double* expectation_values, const hipstatevecPauli_t* const* pauli_arr,
    uint32_t n_pauli_arrs, const int32_t* const* basis_arr, const uint32_t* basis_lens)
{
    return hipify(rocstatevec_compute_expectations_on_pauli_basis(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        expectation_values,
        reinterpret_cast<const rocstatevec_pauli* const*>(pauli_arr),
        n_pauli_arrs, basis_arr, basis_lens));
}

/* ----- accessor ----- */

hipstatevecStatus_t hipstatevecAccessorCreate(
    hipstatevecHandle_t handle, void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    hipstatevecAccessorDescriptor_t* accessor, const int32_t* bit_ordering, uint32_t bo_len,
    const int32_t* mask_bit_string, const int32_t* mask_ordering, uint32_t mask_len, size_t* ws)
{
    rocstatevec_accessor_descriptor a = nullptr;
    auto rc = rocstatevec_accessor_create(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        &a, bit_ordering, bo_len, mask_bit_string, mask_ordering, mask_len, ws);
    *accessor = reinterpret_cast<hipstatevecAccessorDescriptor_t>(a);
    return hipify(rc);
}

hipstatevecStatus_t hipstatevecAccessorCreateView(
    hipstatevecHandle_t handle, const void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    hipstatevecAccessorDescriptor_t* accessor, const int32_t* bit_ordering, uint32_t bo_len,
    const int32_t* mask_bit_string, const int32_t* mask_ordering, uint32_t mask_len, size_t* ws)
{
    rocstatevec_accessor_descriptor a = nullptr;
    auto rc = rocstatevec_accessor_create_view(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        &a, bit_ordering, bo_len, mask_bit_string, mask_ordering, mask_len, ws);
    *accessor = reinterpret_cast<hipstatevecAccessorDescriptor_t>(a);
    return hipify(rc);
}

hipstatevecStatus_t hipstatevecAccessorDestroy(hipstatevecAccessorDescriptor_t accessor)
{
    return hipify(rocstatevec_accessor_destroy(to_roc_a(accessor)));
}

hipstatevecStatus_t hipstatevecAccessorSetExtraWorkspace(
    hipstatevecHandle_t handle, hipstatevecAccessorDescriptor_t accessor,
    void* ws, size_t ws_bytes)
{
    return hipify(rocstatevec_accessor_set_extra_workspace(
        to_roc(handle), to_roc_a(accessor), ws, ws_bytes));
}

hipstatevecStatus_t hipstatevecAccessorGet(
    hipstatevecHandle_t handle, hipstatevecAccessorDescriptor_t accessor, void* buf,
    hipstatevecIndex_t begin, hipstatevecIndex_t end)
{
    return hipify(rocstatevec_accessor_get(to_roc(handle), to_roc_a(accessor), buf, begin, end));
}

hipstatevecStatus_t hipstatevecAccessorSet(
    hipstatevecHandle_t handle, hipstatevecAccessorDescriptor_t accessor, const void* buf,
    hipstatevecIndex_t begin, hipstatevecIndex_t end)
{
    return hipify(rocstatevec_accessor_set(to_roc(handle), to_roc_a(accessor), buf, begin, end));
}

/* ----- permute ----- */

hipstatevecStatus_t hipstatevecSwapIndexBits(
    hipstatevecHandle_t handle, void* sv, hipstatevecDataType_t sv_dt, uint32_t n,
    const hipstatevecIndexPair_t* bit_swaps, uint32_t n_bit_swaps,
    const int32_t* mask_bit_string, const int32_t* mask_ordering, uint32_t mask_len)
{
    static_assert(sizeof(hipstatevecIndexPair_t) == sizeof(rocstatevec_index_pair_t),
                  "index pair sizes must match for reinterpret_cast");
    return hipify(rocstatevec_swap_index_bits(
        to_roc(handle), sv, static_cast<rocstatevec_data_type>(sv_dt), n,
        reinterpret_cast<const rocstatevec_index_pair_t*>(bit_swaps), n_bit_swaps,
        mask_bit_string, mask_ordering, mask_len));
}

/* ----- test_matrix_type ----- */

hipstatevecStatus_t hipstatevecTestMatrixTypeGetWorkspaceSize(
    hipstatevecHandle_t handle, hipstatevecMatrixType_t mt, const void* matrix,
    hipstatevecDataType_t m_dt, hipstatevecMatrixLayout_t layout, uint32_t n_targets,
    int32_t adjoint, hipstatevecComputeType_t ct, size_t* ws)
{
    return hipify(rocstatevec_test_matrix_type_get_workspace_size(
        to_roc(handle), static_cast<rocstatevec_matrix_type>(mt), matrix,
        static_cast<rocstatevec_data_type>(m_dt),
        static_cast<rocstatevec_matrix_layout>(layout),
        n_targets, adjoint, static_cast<rocstatevec_compute_type>(ct), ws));
}

hipstatevecStatus_t hipstatevecTestMatrixType(
    hipstatevecHandle_t handle, double* residual_norm, hipstatevecMatrixType_t mt,
    const void* matrix, hipstatevecDataType_t m_dt, hipstatevecMatrixLayout_t layout,
    uint32_t n_targets, int32_t adjoint, hipstatevecComputeType_t ct,
    void* ws, size_t ws_bytes)
{
    return hipify(rocstatevec_test_matrix_type(
        to_roc(handle), residual_norm, static_cast<rocstatevec_matrix_type>(mt), matrix,
        static_cast<rocstatevec_data_type>(m_dt),
        static_cast<rocstatevec_matrix_layout>(layout),
        n_targets, adjoint, static_cast<rocstatevec_compute_type>(ct), ws, ws_bytes));
}

} // extern "C"
