/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * AMD backend for hipDENSITYMAT: pure pass-through to rocDENSITYMAT.
 *
 * Every function in this file casts opaque hipDENSITYMAT handles to their
 * rocDENSITYMAT equivalents (the underlying structs are the same; only
 * the typedef shape differs between the two public headers) and
 * forwards. Status codes and enum values share numeric identity with
 * cuDensityMat so static_cast is the right tool. The scalar callback
 * struct uses the same `{double x; double y;}` layout in both libraries,
 * so callback pointers are forwarded by reinterpret_cast.
 * ************************************************************************ */

#include <hipdensitymat.h>
#include <rocdensitymat.h>

#include <cstring>

namespace
{
inline rocdensitymat_handle to_roc(hipdensitymatHandle_t h)
{
    return reinterpret_cast<rocdensitymat_handle>(h);
}
inline rocdensitymat_workspace_descriptor to_roc(hipdensitymatWorkspaceDescriptor_t w)
{
    return reinterpret_cast<rocdensitymat_workspace_descriptor>(w);
}
inline rocdensitymat_state to_roc(hipdensitymatState_t s)
{
    return reinterpret_cast<rocdensitymat_state>(s);
}
inline rocdensitymat_elementary_operator to_roc(hipdensitymatElementaryOperator_t e)
{
    return reinterpret_cast<rocdensitymat_elementary_operator>(e);
}
inline rocdensitymat_operator_term to_roc(hipdensitymatOperatorTerm_t t)
{
    return reinterpret_cast<rocdensitymat_operator_term>(t);
}
inline rocdensitymat_operator to_roc(hipdensitymatOperator_t o)
{
    return reinterpret_cast<rocdensitymat_operator>(o);
}
inline rocdensitymat_master_equation_solver to_roc(hipdensitymatMasterEquationSolver_t s)
{
    return reinterpret_cast<rocdensitymat_master_equation_solver>(s);
}
inline hipdensitymatStatus_t hipify(rocdensitymat_status s)
{
    return static_cast<hipdensitymatStatus_t>(s);
}
inline rocdensitymat_complex_double to_roc_c(hipdensitymatComplexDouble_t c)
{
    rocdensitymat_complex_double r{c.x, c.y};
    return r;
}
inline hipdensitymatComplexDouble_t from_roc_c(rocdensitymat_complex_double c)
{
    hipdensitymatComplexDouble_t r{c.x, c.y};
    return r;
}
inline rocdensitymat_scalar_callback_t to_roc_cb(hipdensitymatScalarCallback_t cb)
{
    return reinterpret_cast<rocdensitymat_scalar_callback_t>(cb);
}
} // namespace

extern "C" {

/* ===== handle / version / property / error / memhandler / logger ===== */

hipdensitymatStatus_t hipdensitymatCreate(hipdensitymatHandle_t* handle)
{
    rocdensitymat_handle h = nullptr;
    auto rc = rocdensitymat_create(&h);
    *handle = reinterpret_cast<hipdensitymatHandle_t>(h);
    return hipify(rc);
}

hipdensitymatStatus_t hipdensitymatDestroy(hipdensitymatHandle_t handle)
{
    return hipify(rocdensitymat_destroy(to_roc(handle)));
}

hipdensitymatStatus_t hipdensitymatResetRandomSeed(hipdensitymatHandle_t handle, uint32_t seed)
{
    return hipify(rocdensitymat_reset_random_seed(to_roc(handle), seed));
}

hipdensitymatStatus_t hipdensitymatSetStream(hipdensitymatHandle_t handle, hipStream_t stream)
{
    return hipify(rocdensitymat_set_stream(to_roc(handle), stream));
}

hipdensitymatStatus_t hipdensitymatGetStream(hipdensitymatHandle_t handle, hipStream_t* stream)
{
    return hipify(rocdensitymat_get_stream(to_roc(handle), stream));
}

size_t hipdensitymatGetVersion(void)
{
    return rocdensitymat_get_version();
}

hipdensitymatStatus_t hipdensitymatGetProperty(
    hipdensitymatLibraryPropertyType_t type, int32_t* value)
{
    return hipify(rocdensitymat_get_property(
        static_cast<rocdensitymat_library_property_type>(type), value));
}

const char* hipdensitymatGetErrorName(hipdensitymatStatus_t status)
{
    return rocdensitymat_get_error_name(static_cast<rocdensitymat_status>(status));
}

const char* hipdensitymatGetErrorString(hipdensitymatStatus_t status)
{
    return rocdensitymat_get_error_string(static_cast<rocdensitymat_status>(status));
}

hipdensitymatStatus_t hipdensitymatSetDeviceMemHandler(
    hipdensitymatHandle_t handle, const hipdensitymatDeviceMemHandler_t* memHandler)
{
    if(memHandler == nullptr)
        return hipify(rocdensitymat_set_device_mem_handler(to_roc(handle), nullptr));
    rocdensitymat_device_mem_handler_t roc{};
    roc.ctx          = memHandler->ctx;
    roc.device_alloc = memHandler->device_alloc;
    roc.device_free  = memHandler->device_free;
    std::memcpy(roc.name, memHandler->name, sizeof(roc.name));
    return hipify(rocdensitymat_set_device_mem_handler(to_roc(handle), &roc));
}

hipdensitymatStatus_t hipdensitymatGetDeviceMemHandler(
    hipdensitymatHandle_t handle, hipdensitymatDeviceMemHandler_t* memHandler)
{
    rocdensitymat_device_mem_handler_t roc{};
    auto rc = rocdensitymat_get_device_mem_handler(to_roc(handle), &roc);
    if(rc != ROCDENSITYMAT_STATUS_SUCCESS) return hipify(rc);
    if(memHandler == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    memHandler->ctx          = roc.ctx;
    memHandler->device_alloc = roc.device_alloc;
    memHandler->device_free  = roc.device_free;
    std::memcpy(memHandler->name, roc.name, sizeof(memHandler->name));
    return HIPDENSITYMAT_STATUS_SUCCESS;
}

hipdensitymatStatus_t hipdensitymatLoggerSetCallback(hipdensitymatLoggerCallback_t cb)
{
    return hipify(rocdensitymat_logger_set_callback(
        reinterpret_cast<rocdensitymat_logger_callback_t>(cb)));
}

hipdensitymatStatus_t hipdensitymatLoggerSetCallbackData(
    hipdensitymatLoggerCallbackData_t cb, void* userData)
{
    return hipify(rocdensitymat_logger_set_callback_data(
        reinterpret_cast<rocdensitymat_logger_callback_data_t>(cb), userData));
}

hipdensitymatStatus_t hipdensitymatLoggerSetFile(void* file)
{
    return hipify(rocdensitymat_logger_set_file(file));
}

hipdensitymatStatus_t hipdensitymatLoggerOpenFile(const char* path)
{
    return hipify(rocdensitymat_logger_open_file(path));
}

hipdensitymatStatus_t hipdensitymatLoggerSetLevel(int32_t level)
{
    return hipify(rocdensitymat_logger_set_level(level));
}

hipdensitymatStatus_t hipdensitymatLoggerSetMask(int32_t mask)
{
    return hipify(rocdensitymat_logger_set_mask(mask));
}

hipdensitymatStatus_t hipdensitymatLoggerForceDisable(void)
{
    return hipify(rocdensitymat_logger_force_disable());
}

/* ===== workspace ===== */

hipdensitymatStatus_t hipdensitymatCreateWorkspace(
    hipdensitymatHandle_t handle, hipdensitymatWorkspaceDescriptor_t* workspace)
{
    rocdensitymat_workspace_descriptor w = nullptr;
    auto rc = rocdensitymat_create_workspace(to_roc(handle), &w);
    *workspace = reinterpret_cast<hipdensitymatWorkspaceDescriptor_t>(w);
    return hipify(rc);
}

hipdensitymatStatus_t hipdensitymatDestroyWorkspace(hipdensitymatWorkspaceDescriptor_t workspace)
{
    return hipify(rocdensitymat_destroy_workspace(to_roc(workspace)));
}

hipdensitymatStatus_t hipdensitymatWorkspaceGetMemorySize(
    hipdensitymatHandle_t handle, hipdensitymatWorkspaceDescriptor_t workspace,
    hipdensitymatMemspace_t memSpace, hipdensitymatWorkspaceKind_t kind,
    size_t* memorySizeBytes)
{
    return hipify(rocdensitymat_workspace_get_memory_size(
        to_roc(handle), to_roc(workspace),
        static_cast<rocdensitymat_memspace>(memSpace),
        static_cast<rocdensitymat_workspace_kind>(kind),
        memorySizeBytes));
}

hipdensitymatStatus_t hipdensitymatWorkspaceSetMemory(
    hipdensitymatHandle_t handle, hipdensitymatWorkspaceDescriptor_t workspace,
    hipdensitymatMemspace_t memSpace, hipdensitymatWorkspaceKind_t kind,
    void* memoryPtr, size_t memorySizeBytes)
{
    return hipify(rocdensitymat_workspace_set_memory(
        to_roc(handle), to_roc(workspace),
        static_cast<rocdensitymat_memspace>(memSpace),
        static_cast<rocdensitymat_workspace_kind>(kind),
        memoryPtr, memorySizeBytes));
}

hipdensitymatStatus_t hipdensitymatWorkspaceGetMemory(
    hipdensitymatHandle_t handle, hipdensitymatWorkspaceDescriptor_t workspace,
    hipdensitymatMemspace_t memSpace, hipdensitymatWorkspaceKind_t kind,
    void** memoryPtr, size_t* memorySizeBytes)
{
    return hipify(rocdensitymat_workspace_get_memory(
        to_roc(handle), to_roc(workspace),
        static_cast<rocdensitymat_memspace>(memSpace),
        static_cast<rocdensitymat_workspace_kind>(kind),
        memoryPtr, memorySizeBytes));
}

/* ===== state ===== */

hipdensitymatStatus_t hipdensitymatCreateState(
    hipdensitymatHandle_t handle, hipdensitymatStatePurity_t purity,
    int32_t numSpaceModes, const int64_t* spaceShape,
    int64_t batchSize, hipdensitymatDataType_t dataType,
    hipdensitymatState_t* state)
{
    rocdensitymat_state s = nullptr;
    auto rc = rocdensitymat_create_state(
        to_roc(handle), static_cast<rocdensitymat_state_purity>(purity),
        numSpaceModes, spaceShape, batchSize,
        static_cast<rocdensitymat_data_type>(dataType), &s);
    *state = reinterpret_cast<hipdensitymatState_t>(s);
    return hipify(rc);
}

hipdensitymatStatus_t hipdensitymatDestroyState(hipdensitymatState_t state)
{
    return hipify(rocdensitymat_destroy_state(to_roc(state)));
}

hipdensitymatStatus_t hipdensitymatStateGetNumComponents(
    hipdensitymatHandle_t handle, hipdensitymatState_t state, int32_t* numComponents)
{
    return hipify(rocdensitymat_state_get_num_components(
        to_roc(handle), to_roc(state), numComponents));
}

hipdensitymatStatus_t hipdensitymatStateGetComponentInfo(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    int32_t componentId, int32_t* numModes, int64_t* modeExtents,
    size_t* componentSizeBytes)
{
    return hipify(rocdensitymat_state_get_component_info(
        to_roc(handle), to_roc(state), componentId,
        numModes, modeExtents, componentSizeBytes));
}

hipdensitymatStatus_t hipdensitymatStateAttachComponentBuffer(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    int32_t componentId, void* componentBuffer, size_t componentBufferSize)
{
    return hipify(rocdensitymat_state_attach_component_buffer(
        to_roc(handle), to_roc(state), componentId,
        componentBuffer, componentBufferSize));
}

hipdensitymatStatus_t hipdensitymatStateInitializeZero(
    hipdensitymatHandle_t handle, hipdensitymatState_t state)
{
    return hipify(rocdensitymat_state_initialize_zero(to_roc(handle), to_roc(state)));
}

hipdensitymatStatus_t hipdensitymatStateInitializeUniform(
    hipdensitymatHandle_t handle, hipdensitymatState_t state)
{
    return hipify(rocdensitymat_state_initialize_uniform(to_roc(handle), to_roc(state)));
}

hipdensitymatStatus_t hipdensitymatStateInitializeBasis(
    hipdensitymatHandle_t handle, hipdensitymatState_t state, const int64_t* basisIndices)
{
    return hipify(rocdensitymat_state_initialize_basis(
        to_roc(handle), to_roc(state), basisIndices));
}

hipdensitymatStatus_t hipdensitymatStateComputeNorm(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    hipdensitymatWorkspaceDescriptor_t workspace, double* norm)
{
    return hipify(rocdensitymat_state_compute_norm(
        to_roc(handle), to_roc(state), to_roc(workspace), norm));
}

hipdensitymatStatus_t hipdensitymatStateComputeTrace(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    hipdensitymatWorkspaceDescriptor_t workspace, hipdensitymatComplexDouble_t* trace)
{
    rocdensitymat_complex_double t{0.0, 0.0};
    auto rc = rocdensitymat_state_compute_trace(
        to_roc(handle), to_roc(state), to_roc(workspace), &t);
    if(trace != nullptr) *trace = from_roc_c(t);
    return hipify(rc);
}

hipdensitymatStatus_t hipdensitymatStateComputeOverlap(
    hipdensitymatHandle_t handle, hipdensitymatState_t lhs, hipdensitymatState_t rhs,
    hipdensitymatWorkspaceDescriptor_t workspace, hipdensitymatComplexDouble_t* overlap)
{
    rocdensitymat_complex_double o{0.0, 0.0};
    auto rc = rocdensitymat_state_compute_overlap(
        to_roc(handle), to_roc(lhs), to_roc(rhs), to_roc(workspace), &o);
    if(overlap != nullptr) *overlap = from_roc_c(o);
    return hipify(rc);
}

/* ===== operator ===== */

hipdensitymatStatus_t hipdensitymatCreateElementaryOperator(
    hipdensitymatHandle_t handle, int32_t numModes, const int64_t* modeExtents,
    hipdensitymatElementaryKind_t kind, hipdensitymatDataType_t dataType,
    const void* data, hipdensitymatScalarCallback_t tdepCallback,
    hipdensitymatElementaryOperator_t* op)
{
    rocdensitymat_elementary_operator e = nullptr;
    auto rc = rocdensitymat_create_elementary_operator(
        to_roc(handle), numModes, modeExtents,
        static_cast<rocdensitymat_elementary_kind>(kind),
        static_cast<rocdensitymat_data_type>(dataType),
        data, to_roc_cb(tdepCallback), &e);
    *op = reinterpret_cast<hipdensitymatElementaryOperator_t>(e);
    return hipify(rc);
}

hipdensitymatStatus_t hipdensitymatDestroyElementaryOperator(hipdensitymatElementaryOperator_t op)
{
    return hipify(rocdensitymat_destroy_elementary_operator(to_roc(op)));
}

hipdensitymatStatus_t hipdensitymatCreateOperatorTerm(
    hipdensitymatHandle_t handle, int32_t numSpaceModes,
    const int64_t* spaceShape, hipdensitymatOperatorTerm_t* term)
{
    rocdensitymat_operator_term t = nullptr;
    auto rc = rocdensitymat_create_operator_term(
        to_roc(handle), numSpaceModes, spaceShape, &t);
    *term = reinterpret_cast<hipdensitymatOperatorTerm_t>(t);
    return hipify(rc);
}

hipdensitymatStatus_t hipdensitymatDestroyOperatorTerm(hipdensitymatOperatorTerm_t term)
{
    return hipify(rocdensitymat_destroy_operator_term(to_roc(term)));
}

hipdensitymatStatus_t hipdensitymatOperatorTermAppendElementaryProduct(
    hipdensitymatHandle_t handle, hipdensitymatOperatorTerm_t term,
    int32_t numFactors,
    const hipdensitymatElementaryOperator_t* operators,
    const int32_t* stateModes, const int32_t* modeActionDuality,
    hipdensitymatComplexDouble_t coefficient,
    hipdensitymatScalarCallback_t tdepCallback)
{
    // hipdensitymatElementaryOperator_t and rocdensitymat_elementary_operator
    // are both opaque pointer typedefs over the same underlying struct, so
    // the array of handles is layout-identical and we can reinterpret_cast.
    auto roc_ops = reinterpret_cast<const rocdensitymat_elementary_operator*>(operators);
    return hipify(rocdensitymat_operator_term_append_elementary_product(
        to_roc(handle), to_roc(term), numFactors,
        roc_ops, stateModes, modeActionDuality,
        to_roc_c(coefficient), to_roc_cb(tdepCallback)));
}

hipdensitymatStatus_t hipdensitymatCreateOperator(
    hipdensitymatHandle_t handle, int32_t numSpaceModes,
    const int64_t* spaceShape, hipdensitymatOperator_t* op)
{
    rocdensitymat_operator o = nullptr;
    auto rc = rocdensitymat_create_operator(
        to_roc(handle), numSpaceModes, spaceShape, &o);
    *op = reinterpret_cast<hipdensitymatOperator_t>(o);
    return hipify(rc);
}

hipdensitymatStatus_t hipdensitymatDestroyOperator(hipdensitymatOperator_t op)
{
    return hipify(rocdensitymat_destroy_operator(to_roc(op)));
}

hipdensitymatStatus_t hipdensitymatOperatorAppendTerm(
    hipdensitymatHandle_t handle, hipdensitymatOperator_t op,
    hipdensitymatOperatorTerm_t term, int32_t dualityOffset,
    hipdensitymatComplexDouble_t coefficient,
    hipdensitymatScalarCallback_t tdepCallback)
{
    return hipify(rocdensitymat_operator_append_term(
        to_roc(handle), to_roc(op), to_roc(term),
        dualityOffset, to_roc_c(coefficient), to_roc_cb(tdepCallback)));
}

/* ===== action / expectation ===== */

hipdensitymatStatus_t hipdensitymatOperatorPrepareAction(
    hipdensitymatHandle_t handle, hipdensitymatOperator_t op,
    hipdensitymatState_t stateIn, hipdensitymatState_t stateOut,
    hipdensitymatComputeType_t computeType, size_t workspaceSizeLimit,
    hipdensitymatWorkspaceDescriptor_t workspace)
{
    return hipify(rocdensitymat_operator_prepare_action(
        to_roc(handle), to_roc(op),
        to_roc(stateIn), to_roc(stateOut),
        static_cast<rocdensitymat_compute_type>(computeType),
        workspaceSizeLimit, to_roc(workspace)));
}

hipdensitymatStatus_t hipdensitymatOperatorComputeAction(
    hipdensitymatHandle_t handle, hipdensitymatOperator_t op,
    double t, int32_t numParams, const double* params,
    hipdensitymatState_t stateIn, hipdensitymatState_t stateOut,
    hipdensitymatWorkspaceDescriptor_t workspace)
{
    return hipify(rocdensitymat_operator_compute_action(
        to_roc(handle), to_roc(op), t, numParams, params,
        to_roc(stateIn), to_roc(stateOut), to_roc(workspace)));
}

hipdensitymatStatus_t hipdensitymatOperatorComputeExpectation(
    hipdensitymatHandle_t handle, hipdensitymatOperator_t op,
    double t, int32_t numParams, const double* params,
    hipdensitymatState_t state,
    hipdensitymatWorkspaceDescriptor_t workspace,
    hipdensitymatComplexDouble_t* expectation)
{
    rocdensitymat_complex_double e{0.0, 0.0};
    auto rc = rocdensitymat_operator_compute_expectation(
        to_roc(handle), to_roc(op), t, numParams, params,
        to_roc(state), to_roc(workspace), &e);
    if(expectation != nullptr) *expectation = from_roc_c(e);
    return hipify(rc);
}

/* ===== ode ===== */

hipdensitymatStatus_t hipdensitymatCreateMasterEquationSolver(
    hipdensitymatHandle_t handle, hipdensitymatOperator_t liouvillian,
    hipdensitymatSolverKind_t kind,
    hipdensitymatMasterEquationSolver_t* solver)
{
    rocdensitymat_master_equation_solver s = nullptr;
    auto rc = rocdensitymat_create_master_equation_solver(
        to_roc(handle), to_roc(liouvillian),
        static_cast<rocdensitymat_solver_kind>(kind), &s);
    *solver = reinterpret_cast<hipdensitymatMasterEquationSolver_t>(s);
    return hipify(rc);
}

hipdensitymatStatus_t hipdensitymatDestroyMasterEquationSolver(
    hipdensitymatMasterEquationSolver_t solver)
{
    return hipify(rocdensitymat_destroy_master_equation_solver(to_roc(solver)));
}

hipdensitymatStatus_t hipdensitymatMasterEquationSolverPrepare(
    hipdensitymatHandle_t handle, hipdensitymatMasterEquationSolver_t solver,
    hipdensitymatState_t state, hipdensitymatComputeType_t computeType,
    size_t workspaceSizeLimit, hipdensitymatWorkspaceDescriptor_t workspace)
{
    return hipify(rocdensitymat_master_equation_solver_prepare(
        to_roc(handle), to_roc(solver), to_roc(state),
        static_cast<rocdensitymat_compute_type>(computeType),
        workspaceSizeLimit, to_roc(workspace)));
}

hipdensitymatStatus_t hipdensitymatMasterEquationStep(
    hipdensitymatHandle_t handle, hipdensitymatMasterEquationSolver_t solver,
    double t0, double dt, int32_t numParams, const double* params,
    hipdensitymatState_t state, hipdensitymatWorkspaceDescriptor_t workspace)
{
    return hipify(rocdensitymat_master_equation_step(
        to_roc(handle), to_roc(solver), t0, dt,
        numParams, params, to_roc(state), to_roc(workspace)));
}

hipdensitymatStatus_t hipdensitymatMasterEquationStepN(
    hipdensitymatHandle_t handle, hipdensitymatMasterEquationSolver_t solver,
    double t0, double dt, int64_t numSteps,
    int32_t numParams, const double* params,
    hipdensitymatState_t state, hipdensitymatWorkspaceDescriptor_t workspace)
{
    return hipify(rocdensitymat_master_equation_step_n(
        to_roc(handle), to_roc(solver), t0, dt, numSteps,
        numParams, params, to_roc(state), to_roc(workspace)));
}

/* ===== properties ===== */

hipdensitymatStatus_t hipdensitymatStateGetDataType(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    hipdensitymatDataType_t* dataType)
{
    rocdensitymat_data_type d{};
    auto rc = rocdensitymat_state_get_data_type(to_roc(handle), to_roc(state), &d);
    if(dataType != nullptr) *dataType = static_cast<hipdensitymatDataType_t>(d);
    return hipify(rc);
}

hipdensitymatStatus_t hipdensitymatStateGetPurity(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    hipdensitymatStatePurity_t* purity)
{
    rocdensitymat_state_purity p{};
    auto rc = rocdensitymat_state_get_purity(to_roc(handle), to_roc(state), &p);
    if(purity != nullptr) *purity = static_cast<hipdensitymatStatePurity_t>(p);
    return hipify(rc);
}

hipdensitymatStatus_t hipdensitymatStateGetSpaceShape(
    hipdensitymatHandle_t handle, hipdensitymatState_t state,
    int32_t* numSpaceModes, int64_t* spaceShape)
{
    return hipify(rocdensitymat_state_get_space_shape(
        to_roc(handle), to_roc(state), numSpaceModes, spaceShape));
}

/* ===== distributed (NOT_SUPPORTED) ===== */

hipdensitymatStatus_t hipdensitymatResetDistributedConfiguration(
    hipdensitymatHandle_t handle, hipdensitymatDistributedProvider_t provider,
    const void* commPtr, size_t commSize)
{
    return hipify(rocdensitymat_reset_distributed_configuration(
        to_roc(handle),
        static_cast<rocdensitymat_distributed_provider>(provider),
        commPtr, commSize));
}

hipdensitymatStatus_t hipdensitymatGetNumRanks(hipdensitymatHandle_t handle, int32_t* numRanks)
{
    return hipify(rocdensitymat_get_num_ranks(to_roc(handle), numRanks));
}

hipdensitymatStatus_t hipdensitymatGetProcRank(hipdensitymatHandle_t handle, int32_t* procRank)
{
    return hipify(rocdensitymat_get_proc_rank(to_roc(handle), procRank));
}

/* ===== eigen (NOT_SUPPORTED) ===== */

hipdensitymatStatus_t hipdensitymatOperatorPrepareEigenspectrum(
    hipdensitymatHandle_t handle, hipdensitymatOperator_t op,
    hipdensitymatState_t state, int32_t numEigenpairs,
    hipdensitymatComputeType_t computeType, size_t workspaceSizeLimit,
    hipdensitymatWorkspaceDescriptor_t workspace)
{
    return hipify(rocdensitymat_operator_prepare_eigenspectrum(
        to_roc(handle), to_roc(op), to_roc(state),
        numEigenpairs,
        static_cast<rocdensitymat_compute_type>(computeType),
        workspaceSizeLimit, to_roc(workspace)));
}

hipdensitymatStatus_t hipdensitymatOperatorComputeEigenspectrum(
    hipdensitymatHandle_t handle, hipdensitymatOperator_t op,
    int32_t numEigenpairs, hipdensitymatComplexDouble_t* eigenvalues,
    hipdensitymatState_t* eigenvectors,
    hipdensitymatWorkspaceDescriptor_t workspace)
{
    auto roc_eigvecs = reinterpret_cast<rocdensitymat_state*>(eigenvectors);
    auto roc_eigvals = reinterpret_cast<rocdensitymat_complex_double*>(eigenvalues);
    return hipify(rocdensitymat_operator_compute_eigenspectrum(
        to_roc(handle), to_roc(op),
        numEigenpairs, roc_eigvals, roc_eigvecs, to_roc(workspace)));
}

/* ===== backward (NOT_SUPPORTED) ===== */

hipdensitymatStatus_t hipdensitymatOperatorPrepareActionBackwardDiff(
    hipdensitymatHandle_t handle, hipdensitymatOperator_t op,
    hipdensitymatState_t stateIn, hipdensitymatState_t stateOut,
    hipdensitymatComputeType_t computeType, size_t workspaceSizeLimit,
    hipdensitymatWorkspaceDescriptor_t workspace)
{
    return hipify(rocdensitymat_operator_prepare_action_backward_diff(
        to_roc(handle), to_roc(op), to_roc(stateIn), to_roc(stateOut),
        static_cast<rocdensitymat_compute_type>(computeType),
        workspaceSizeLimit, to_roc(workspace)));
}

hipdensitymatStatus_t hipdensitymatOperatorComputeActionBackwardDiff(
    hipdensitymatHandle_t handle, hipdensitymatOperator_t op,
    double t, int32_t numParams, const double* params,
    hipdensitymatState_t stateIn,
    hipdensitymatState_t stateOutGrad, hipdensitymatState_t stateInGrad,
    double* paramsGrad, hipdensitymatWorkspaceDescriptor_t workspace)
{
    return hipify(rocdensitymat_operator_compute_action_backward_diff(
        to_roc(handle), to_roc(op), t, numParams, params,
        to_roc(stateIn), to_roc(stateOutGrad), to_roc(stateInGrad),
        paramsGrad, to_roc(workspace)));
}

} // extern "C"
