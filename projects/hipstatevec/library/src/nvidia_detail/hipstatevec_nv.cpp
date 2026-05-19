/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * NVIDIA backend for hipSTATEVEC: pass-through to cuStateVec.
 *
 * Built when `HIPSTATEVEC_BACKEND_NVIDIA` is defined and the cuQuantum SDK
 * headers/library are available at compile/link time. Each entry point
 * below maps the hipSTATEVEC argument list to the matching cuStateVec
 * call. Because every hipSTATEVEC enum and integer typedef was chosen
 * with numeric values identical to cuStateVec, the mapping is a small
 * number of straight-line `static_cast`s — there is no per-call
 * translation logic.
 *
 * If the SDK is unavailable, the symbols below are defined as
 * `HIPSTATEVEC_STATUS_LOADING_LIBRARY_FAILED` so consumers get an obvious
 * runtime error instead of a link error.
 * ************************************************************************ */

#include <hipstatevec.h>

#include <cstring>
#include <new>

#if defined(HIPSTATEVEC_HAS_CUSTATEVEC)

#include <cuda_runtime.h>
#include <custatevec.h>

namespace
{

inline custatevecHandle_t cu(hipstatevecHandle_t h)
{
    return reinterpret_cast<custatevecHandle_t>(h);
}
inline hipstatevecHandle_t hp(custatevecHandle_t h)
{
    return reinterpret_cast<hipstatevecHandle_t>(h);
}
inline custatevecSamplerDescriptor_t sd(hipstatevecSamplerDescriptor_t s)
{
    return reinterpret_cast<custatevecSamplerDescriptor_t>(s);
}
inline custatevecAccessorDescriptor_t ad(hipstatevecAccessorDescriptor_t s)
{
    return reinterpret_cast<custatevecAccessorDescriptor_t>(s);
}
inline cudaDataType_t                cd(hipstatevecDataType_t       t) { return static_cast<cudaDataType_t>(int(t)); }
inline custatevecComputeType_t       ct(hipstatevecComputeType_t    t) { return static_cast<custatevecComputeType_t>(uint32_t(t)); }
inline custatevecMatrixLayout_t      ml(hipstatevecMatrixLayout_t   t) { return static_cast<custatevecMatrixLayout_t>(int(t)); }
inline custatevecMatrixType_t        mt(hipstatevecMatrixType_t     t) { return static_cast<custatevecMatrixType_t>(int(t)); }
inline custatevecMatrixMapType_t     mm(hipstatevecMatrixMapType_t  t) { return static_cast<custatevecMatrixMapType_t>(int(t)); }
inline custatevecCollapseOp_t        co(hipstatevecCollapseOp_t     t) { return static_cast<custatevecCollapseOp_t>(int(t)); }
inline custatevecSamplerOutput_t     so(hipstatevecSamplerOutput_t  t) { return static_cast<custatevecSamplerOutput_t>(int(t)); }
inline custatevecStateVectorType_t   sv(hipstatevecStateVectorType_t t) { return static_cast<custatevecStateVectorType_t>(int(t)); }
inline custatevecPauli_t             pa(hipstatevecPauli_t          t) { return static_cast<custatevecPauli_t>(int(t)); }
inline libraryPropertyType           lp(hipstatevecLibraryPropertyType_t t)
{
    switch(t)
    {
    case HIPSTATEVEC_PROPERTY_MAJOR_VERSION: return MAJOR_VERSION;
    case HIPSTATEVEC_PROPERTY_MINOR_VERSION: return MINOR_VERSION;
    case HIPSTATEVEC_PROPERTY_PATCH_LEVEL:   return PATCH_LEVEL;
    }
    return MAJOR_VERSION;
}

inline hipstatevecStatus_t st(custatevecStatus_t s)
{
    return static_cast<hipstatevecStatus_t>(int(s));
}

inline const custatevecPauli_t* pa_arr(const hipstatevecPauli_t* p)
{
    static_assert(sizeof(custatevecPauli_t) == sizeof(hipstatevecPauli_t),
                  "enum widths must match");
    return reinterpret_cast<const custatevecPauli_t*>(p);
}

} // namespace

extern "C" {

#define HSV_OK return HIPSTATEVEC_STATUS_SUCCESS

hipstatevecStatus_t hipstatevecCreate(hipstatevecHandle_t* h)
{
    custatevecHandle_t cuh = nullptr;
    auto s = custatevecCreate(&cuh);
    if(h) *h = hp(cuh);
    return st(s);
}
hipstatevecStatus_t hipstatevecDestroy(hipstatevecHandle_t h)
{
    return st(custatevecDestroy(cu(h)));
}
hipstatevecStatus_t hipstatevecGetVersion(hipstatevecHandle_t, int* v)
{
    if(!v) return HIPSTATEVEC_STATUS_INVALID_VALUE;
    *v = CUSTATEVEC_VER_MAJOR * 1000 + CUSTATEVEC_VER_MINOR * 100 + CUSTATEVEC_VER_PATCH;
    HSV_OK;
}
hipstatevecStatus_t hipstatevecGetProperty(hipstatevecLibraryPropertyType_t t, int* v)
{
    return st(custatevecGetProperty(lp(t), v));
}
hipstatevecStatus_t hipstatevecSetStream(hipstatevecHandle_t h, hipStream_t s)
{
    return st(custatevecSetStream(cu(h), reinterpret_cast<cudaStream_t>(s)));
}
hipstatevecStatus_t hipstatevecGetStream(hipstatevecHandle_t h, hipStream_t* s)
{
    cudaStream_t cs = nullptr;
    auto status = custatevecGetStream(cu(h), &cs);
    if(s) *s = reinterpret_cast<hipStream_t>(cs);
    return st(status);
}
const char* hipstatevecGetErrorName(hipstatevecStatus_t s)
{
    return custatevecGetErrorName(static_cast<custatevecStatus_t>(int(s)));
}
const char* hipstatevecGetErrorString(hipstatevecStatus_t s)
{
    return custatevecGetErrorString(static_cast<custatevecStatus_t>(int(s)));
}
hipstatevecStatus_t hipstatevecSetDeviceMemHandler(hipstatevecHandle_t h,
                                                    const hipstatevecDeviceMemHandler_t* m)
{
    return st(custatevecSetDeviceMemHandler(
        cu(h), reinterpret_cast<const custatevecDeviceMemHandler_t*>(m)));
}
hipstatevecStatus_t hipstatevecGetDeviceMemHandler(hipstatevecHandle_t h,
                                                    hipstatevecDeviceMemHandler_t* m)
{
    return st(custatevecGetDeviceMemHandler(
        cu(h), reinterpret_cast<custatevecDeviceMemHandler_t*>(m)));
}
hipstatevecStatus_t hipstatevecLoggerSetCallback(hipstatevecLoggerCallback_t cb)
{
    return st(custatevecLoggerSetCallback(reinterpret_cast<custatevecLoggerCallback_t>(cb)));
}
hipstatevecStatus_t hipstatevecLoggerSetCallbackData(hipstatevecLoggerCallbackData_t cb, void* u)
{
    return st(custatevecLoggerSetCallbackData(
        reinterpret_cast<custatevecLoggerCallbackData_t>(cb), u));
}
hipstatevecStatus_t hipstatevecLoggerSetFile(void* f)
{
    return st(custatevecLoggerSetFile(reinterpret_cast<FILE*>(f)));
}
hipstatevecStatus_t hipstatevecLoggerOpenFile(const char* p)
{
    return st(custatevecLoggerOpenFile(p));
}
hipstatevecStatus_t hipstatevecLoggerSetLevel(int32_t l) { return st(custatevecLoggerSetLevel(l)); }
hipstatevecStatus_t hipstatevecLoggerSetMask(int32_t m)  { return st(custatevecLoggerSetMask(m)); }
hipstatevecStatus_t hipstatevecLoggerForceDisable(void)  { return st(custatevecLoggerForceDisable()); }

hipstatevecStatus_t hipstatevecInitializeStateVector(
    hipstatevecHandle_t h, void* p, hipstatevecDataType_t dt, uint32_t n, hipstatevecStateVectorType_t t)
{
    return st(custatevecInitializeStateVector(cu(h), p, cd(dt), n, sv(t)));
}
hipstatevecStatus_t hipstatevecApplyMatrixGetWorkspaceSize(
    hipstatevecHandle_t h, hipstatevecDataType_t dt, uint32_t n, const void* m,
    hipstatevecDataType_t mdt, hipstatevecMatrixLayout_t l, int32_t adj,
    uint32_t nt, uint32_t nc, hipstatevecComputeType_t c, size_t* ws)
{
    return st(custatevecApplyMatrixGetWorkspaceSize(
        cu(h), cd(dt), n, m, cd(mdt), ml(l), adj, nt, nc, ct(c), ws));
}
hipstatevecStatus_t hipstatevecApplyMatrix(
    hipstatevecHandle_t h, void* p, hipstatevecDataType_t dt, uint32_t n,
    const void* m, hipstatevecDataType_t mdt, hipstatevecMatrixLayout_t l, int32_t adj,
    const int32_t* tg, uint32_t nt, const int32_t* ct_qb, const int32_t* cv, uint32_t nc,
    hipstatevecComputeType_t c, void* ws, size_t ws_b)
{
    return st(custatevecApplyMatrix(
        cu(h), p, cd(dt), n, m, cd(mdt), ml(l), adj,
        tg, nt, ct_qb, cv, nc, ct(c), ws, ws_b));
}
hipstatevecStatus_t hipstatevecApplyPauliRotation(
    hipstatevecHandle_t h, void* p, hipstatevecDataType_t dt, uint32_t n, double theta,
    const hipstatevecPauli_t* pp, const int32_t* tg, uint32_t nt,
    const int32_t* ct_qb, const int32_t* cv, uint32_t nc)
{
    return st(custatevecApplyPauliRotation(
        cu(h), p, cd(dt), n, theta, pa_arr(pp), tg, nt, ct_qb, cv, nc));
}
hipstatevecStatus_t hipstatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(
    hipstatevecHandle_t h, hipstatevecDataType_t dt, uint32_t n,
    const hipstatevecIndex_t* perm, const void* dg, hipstatevecDataType_t ddt,
    const int32_t* tg, uint32_t nt, uint32_t nc, size_t* ws)
{
    return st(custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(
        cu(h), cd(dt), n, perm, dg, cd(ddt), tg, nt, nc, ws));
}
hipstatevecStatus_t hipstatevecApplyGeneralizedPermutationMatrix(
    hipstatevecHandle_t h, void* p, hipstatevecDataType_t dt, uint32_t n,
    const hipstatevecIndex_t* perm, const void* dg, hipstatevecDataType_t ddt, int32_t adj,
    const int32_t* tg, uint32_t nt, const int32_t* ct_qb, const int32_t* cv, uint32_t nc,
    void* ws, size_t ws_b)
{
    return st(custatevecApplyGeneralizedPermutationMatrix(
        cu(h), p, cd(dt), n, perm, dg, cd(ddt), adj,
        tg, nt, ct_qb, cv, nc, ws, ws_b));
}
hipstatevecStatus_t hipstatevecAbsorbDiagonalMatrix(
    hipstatevecHandle_t h, void* p, hipstatevecDataType_t dt, uint32_t n,
    const void* dg, hipstatevecDataType_t ddt, int32_t adj,
    const int32_t* tg, uint32_t nt, const int32_t* ct_qb, const int32_t* cv, uint32_t nc)
{
    return st(custatevecApplyGeneralizedPermutationMatrix(
        cu(h), p, cd(dt), n, /*permutation=*/nullptr, dg, cd(ddt), adj,
        tg, nt, ct_qb, cv, nc, /*workspace=*/nullptr, /*workspace_bytes=*/0));
}
hipstatevecStatus_t hipstatevecApplyMatrixBatchedGetWorkspaceSize(
    hipstatevecHandle_t h, hipstatevecDataType_t dt, uint32_t nbits,
    uint32_t n_sv, hipstatevecIndex_t sv_size, hipstatevecMatrixMapType_t map_t,
    const int32_t* mi, const void* mats, hipstatevecDataType_t mdt,
    hipstatevecMatrixLayout_t l, int32_t adj, uint32_t n_mats,
    uint32_t nt, uint32_t nc, hipstatevecComputeType_t c, size_t* ws)
{
    return st(custatevecApplyMatrixBatchedGetWorkspaceSize(
        cu(h), cd(dt), nbits, n_sv, sv_size, mm(map_t), mi, mats, cd(mdt),
        ml(l), adj, n_mats, nt, nc, ct(c), ws));
}
hipstatevecStatus_t hipstatevecApplyMatrixBatched(
    hipstatevecHandle_t h, void* sv_buf, hipstatevecDataType_t dt, uint32_t nbits,
    uint32_t n_sv, hipstatevecIndex_t sv_size, hipstatevecMatrixMapType_t map_t,
    const int32_t* mi, const void* mats, hipstatevecDataType_t mdt,
    hipstatevecMatrixLayout_t l, int32_t adj, uint32_t n_mats,
    const int32_t* tg, uint32_t nt, const int32_t* ct_qb, const int32_t* cv, uint32_t nc,
    hipstatevecComputeType_t c, void* ws, size_t ws_b)
{
    return st(custatevecApplyMatrixBatched(
        cu(h), sv_buf, cd(dt), nbits, n_sv, sv_size, mm(map_t), mi, mats, cd(mdt),
        ml(l), adj, n_mats, tg, nt, ct_qb, cv, nc, ct(c), ws, ws_b));
}
hipstatevecStatus_t hipstatevecAbs2SumArray(
    hipstatevecHandle_t h, const void* p, hipstatevecDataType_t dt, uint32_t n,
    double* out, const int32_t* bo, uint32_t bo_len,
    const int32_t* mb, const int32_t* mo, uint32_t ml_len)
{
    return st(custatevecAbs2SumArray(
        cu(h), p, cd(dt), n, out, bo, bo_len, mb, mo, ml_len));
}
hipstatevecStatus_t hipstatevecAbs2SumOnZBasis(
    hipstatevecHandle_t h, const void* p, hipstatevecDataType_t dt, uint32_t n,
    double* a0, double* a1, const int32_t* base, uint32_t base_len)
{
    return st(custatevecAbs2SumOnZBasis(cu(h), p, cd(dt), n, a0, a1, base, base_len));
}
hipstatevecStatus_t hipstatevecMeasureOnZBasis(
    hipstatevecHandle_t h, void* p, hipstatevecDataType_t dt, uint32_t n,
    int32_t* parity, const int32_t* base, uint32_t base_len, double rng,
    hipstatevecCollapseOp_t collapse)
{
    return st(custatevecMeasureOnZBasis(
        cu(h), p, cd(dt), n, parity, base, base_len, rng, co(collapse)));
}
hipstatevecStatus_t hipstatevecBatchMeasure(
    hipstatevecHandle_t h, void* p, hipstatevecDataType_t dt, uint32_t n,
    int32_t* bs, const int32_t* bo, uint32_t bo_len, double rng,
    hipstatevecCollapseOp_t collapse)
{
    return st(custatevecBatchMeasure(
        cu(h), p, cd(dt), n, bs, bo, bo_len, rng, co(collapse)));
}
hipstatevecStatus_t hipstatevecBatchMeasureWithOffset(
    hipstatevecHandle_t h, void* p, hipstatevecDataType_t dt, uint32_t n,
    int32_t* bs, const int32_t* bo, uint32_t bo_len, double rng,
    hipstatevecCollapseOp_t collapse, double offset, double norm)
{
    return st(custatevecBatchMeasureWithOffset(
        cu(h), p, cd(dt), n, bs, bo, bo_len, rng, co(collapse), offset, norm));
}
hipstatevecStatus_t hipstatevecCollapseOnZBasis(
    hipstatevecHandle_t h, void* p, hipstatevecDataType_t dt, uint32_t n,
    int32_t parity, const int32_t* base, uint32_t base_len, double norm)
{
    return st(custatevecCollapseOnZBasis(
        cu(h), p, cd(dt), n, parity, base, base_len, norm));
}
hipstatevecStatus_t hipstatevecCollapseByBitString(
    hipstatevecHandle_t h, void* p, hipstatevecDataType_t dt, uint32_t n,
    const int32_t* bs, const int32_t* bo, uint32_t bo_len, double norm)
{
    return st(custatevecCollapseByBitString(
        cu(h), p, cd(dt), n, bs, bo, bo_len, norm));
}
hipstatevecStatus_t hipstatevecSamplerCreate(
    hipstatevecHandle_t h, const void* p, hipstatevecDataType_t dt, uint32_t n,
    hipstatevecSamplerDescriptor_t* out, uint32_t shots, size_t* ws)
{
    custatevecSamplerDescriptor_t s = nullptr;
    auto status = custatevecSamplerCreate(cu(h), p, cd(dt), n, &s, shots, ws);
    if(out) *out = reinterpret_cast<hipstatevecSamplerDescriptor_t>(s);
    return st(status);
}
hipstatevecStatus_t hipstatevecSamplerDestroy(hipstatevecSamplerDescriptor_t s)
{
    return st(custatevecSamplerDestroy(sd(s)));
}
hipstatevecStatus_t hipstatevecSamplerPreprocess(
    hipstatevecHandle_t h, hipstatevecSamplerDescriptor_t s, void* ws, size_t ws_b)
{
    return st(custatevecSamplerPreprocess(cu(h), sd(s), ws, ws_b));
}
hipstatevecStatus_t hipstatevecSamplerGetSquaredNorm(
    hipstatevecHandle_t h, hipstatevecSamplerDescriptor_t s, double* out)
{
    return st(custatevecSamplerGetSquaredNorm(cu(h), sd(s), out));
}
hipstatevecStatus_t hipstatevecSamplerApplySubSVOffset(
    hipstatevecHandle_t h, hipstatevecSamplerDescriptor_t s, int32_t idx,
    uint32_t n_sub, double offset, double norm)
{
    return st(custatevecSamplerApplySubSVOffset(cu(h), sd(s), idx, n_sub, offset, norm));
}
hipstatevecStatus_t hipstatevecSamplerSample(
    hipstatevecHandle_t h, hipstatevecSamplerDescriptor_t s, hipstatevecIndex_t* out,
    const int32_t* bo, uint32_t bo_len, const double* rng, uint32_t shots,
    hipstatevecSamplerOutput_t order)
{
    return st(custatevecSamplerSample(
        cu(h), sd(s), out, bo, bo_len, rng, shots, so(order)));
}
hipstatevecStatus_t hipstatevecComputeExpectationGetWorkspaceSize(
    hipstatevecHandle_t h, hipstatevecDataType_t dt, uint32_t n,
    const void* m, hipstatevecDataType_t mdt, hipstatevecMatrixLayout_t l,
    uint32_t nt, hipstatevecComputeType_t c, size_t* ws)
{
    return st(custatevecComputeExpectationGetWorkspaceSize(
        cu(h), cd(dt), n, m, cd(mdt), ml(l), nt, ct(c), ws));
}
hipstatevecStatus_t hipstatevecComputeExpectation(
    hipstatevecHandle_t h, const void* p, hipstatevecDataType_t dt, uint32_t n,
    void* exp_v, hipstatevecDataType_t exp_dt, double* residual,
    const void* m, hipstatevecDataType_t mdt, hipstatevecMatrixLayout_t l,
    const int32_t* tg, uint32_t nt, hipstatevecComputeType_t c,
    void* ws, size_t ws_b)
{
    return st(custatevecComputeExpectation(
        cu(h), p, cd(dt), n, exp_v, cd(exp_dt), residual, m, cd(mdt),
        ml(l), tg, nt, ct(c), ws, ws_b));
}
hipstatevecStatus_t hipstatevecComputeExpectationsOnPauliBasis(
    hipstatevecHandle_t h, const void* p, hipstatevecDataType_t dt, uint32_t n,
    double* out, const hipstatevecPauli_t* const* paulis, uint32_t n_strings,
    const int32_t* const* qubits, const uint32_t* qubit_counts)
{
    return st(custatevecComputeExpectationsOnPauliBasis(
        cu(h), p, cd(dt), n, out,
        reinterpret_cast<const custatevecPauli_t* const*>(paulis),
        n_strings, qubits, qubit_counts));
}
hipstatevecStatus_t hipstatevecAccessorCreate(
    hipstatevecHandle_t h, void* p, hipstatevecDataType_t dt, uint32_t n,
    hipstatevecAccessorDescriptor_t* out, const int32_t* bo, uint32_t bo_len,
    const int32_t* mb, const int32_t* mo, uint32_t ml_len, size_t* ws)
{
    custatevecAccessorDescriptor_t a = nullptr;
    auto status = custatevecAccessorCreate(
        cu(h), p, cd(dt), n, &a, bo, bo_len, mb, mo, ml_len, ws);
    if(out) *out = reinterpret_cast<hipstatevecAccessorDescriptor_t>(a);
    return st(status);
}
hipstatevecStatus_t hipstatevecAccessorCreateView(
    hipstatevecHandle_t h, const void* p, hipstatevecDataType_t dt, uint32_t n,
    hipstatevecAccessorDescriptor_t* out, const int32_t* bo, uint32_t bo_len,
    const int32_t* mb, const int32_t* mo, uint32_t ml_len, size_t* ws)
{
    custatevecAccessorDescriptor_t a = nullptr;
    auto status = custatevecAccessorCreateView(
        cu(h), p, cd(dt), n, &a, bo, bo_len, mb, mo, ml_len, ws);
    if(out) *out = reinterpret_cast<hipstatevecAccessorDescriptor_t>(a);
    return st(status);
}
hipstatevecStatus_t hipstatevecAccessorDestroy(hipstatevecAccessorDescriptor_t a)
{
    return st(custatevecAccessorDestroy(ad(a)));
}
hipstatevecStatus_t hipstatevecAccessorSetExtraWorkspace(
    hipstatevecHandle_t h, hipstatevecAccessorDescriptor_t a, void* ws, size_t ws_b)
{
    return st(custatevecAccessorSetExtraWorkspace(cu(h), ad(a), ws, ws_b));
}
hipstatevecStatus_t hipstatevecAccessorGet(
    hipstatevecHandle_t h, hipstatevecAccessorDescriptor_t a, void* buf,
    hipstatevecIndex_t b, hipstatevecIndex_t e)
{
    return st(custatevecAccessorGet(cu(h), ad(a), buf, b, e));
}
hipstatevecStatus_t hipstatevecAccessorSet(
    hipstatevecHandle_t h, hipstatevecAccessorDescriptor_t a, const void* buf,
    hipstatevecIndex_t b, hipstatevecIndex_t e)
{
    return st(custatevecAccessorSet(cu(h), ad(a), buf, b, e));
}
hipstatevecStatus_t hipstatevecSwapIndexBits(
    hipstatevecHandle_t h, void* p, hipstatevecDataType_t dt, uint32_t n,
    const hipstatevecIndexPair_t* pairs, uint32_t n_pairs,
    const int32_t* mb, const int32_t* mo, uint32_t ml_len)
{
    return st(custatevecSwapIndexBits(
        cu(h), p, cd(dt), n,
        reinterpret_cast<const int2*>(pairs), n_pairs, mb, mo, ml_len));
}
hipstatevecStatus_t hipstatevecTestMatrixTypeGetWorkspaceSize(
    hipstatevecHandle_t h, hipstatevecMatrixType_t kind, const void* m,
    hipstatevecDataType_t dt, hipstatevecMatrixLayout_t l, uint32_t nt,
    int32_t adj, hipstatevecComputeType_t c, size_t* ws)
{
    return st(custatevecTestMatrixTypeGetWorkspaceSize(
        cu(h), mt(kind), m, cd(dt), ml(l), nt, adj, ct(c), ws));
}
hipstatevecStatus_t hipstatevecTestMatrixType(
    hipstatevecHandle_t h, double* residual, hipstatevecMatrixType_t kind,
    const void* m, hipstatevecDataType_t dt, hipstatevecMatrixLayout_t l,
    uint32_t nt, int32_t adj, hipstatevecComputeType_t c, void* ws, size_t ws_b)
{
    return st(custatevecTestMatrixType(
        cu(h), residual, mt(kind), m, cd(dt), ml(l), nt, adj, ct(c), ws, ws_b));
}

#undef HSV_OK

} // extern "C"

#else /* HIPSTATEVEC_HAS_CUSTATEVEC */

namespace
{
struct hip_handle_stub
{
    void* stream = nullptr;
};
} // namespace

extern "C" {

#define HSV_NL return HIPSTATEVEC_STATUS_LOADING_LIBRARY_FAILED

hipstatevecStatus_t hipstatevecCreate(hipstatevecHandle_t* handle)
{
    if(handle == nullptr) return HIPSTATEVEC_STATUS_INVALID_VALUE;
    *handle = reinterpret_cast<hipstatevecHandle_t>(new(std::nothrow) hip_handle_stub{});
    return HIPSTATEVEC_STATUS_SUCCESS;
}
hipstatevecStatus_t hipstatevecDestroy(hipstatevecHandle_t handle)
{
    delete reinterpret_cast<hip_handle_stub*>(handle);
    return HIPSTATEVEC_STATUS_SUCCESS;
}
hipstatevecStatus_t hipstatevecGetVersion(hipstatevecHandle_t, int* v)
{
    if(v == nullptr) return HIPSTATEVEC_STATUS_INVALID_VALUE;
    *v = HIPSTATEVEC_VERSION_MAJOR * 1000 + HIPSTATEVEC_VERSION_MINOR * 100 + HIPSTATEVEC_VERSION_PATCH;
    return HIPSTATEVEC_STATUS_SUCCESS;
}
hipstatevecStatus_t hipstatevecGetProperty(hipstatevecLibraryPropertyType_t t, int* v)
{
    if(v == nullptr) return HIPSTATEVEC_STATUS_INVALID_VALUE;
    switch(t) {
    case HIPSTATEVEC_PROPERTY_MAJOR_VERSION: *v = HIPSTATEVEC_VERSION_MAJOR; return HIPSTATEVEC_STATUS_SUCCESS;
    case HIPSTATEVEC_PROPERTY_MINOR_VERSION: *v = HIPSTATEVEC_VERSION_MINOR; return HIPSTATEVEC_STATUS_SUCCESS;
    case HIPSTATEVEC_PROPERTY_PATCH_LEVEL:   *v = HIPSTATEVEC_VERSION_PATCH; return HIPSTATEVEC_STATUS_SUCCESS;
    }
    return HIPSTATEVEC_STATUS_INVALID_VALUE;
}
hipstatevecStatus_t hipstatevecSetStream(hipstatevecHandle_t, hipStream_t)         { HSV_NL; }
hipstatevecStatus_t hipstatevecGetStream(hipstatevecHandle_t, hipStream_t*)        { HSV_NL; }
const char* hipstatevecGetErrorName(hipstatevecStatus_t)                            { return "HIPSTATEVEC_NV_BACKEND_NOT_LINKED"; }
const char* hipstatevecGetErrorString(hipstatevecStatus_t)                          { return "rebuild with cuStateVec available."; }
hipstatevecStatus_t hipstatevecSetDeviceMemHandler(hipstatevecHandle_t, const hipstatevecDeviceMemHandler_t*) { HSV_NL; }
hipstatevecStatus_t hipstatevecGetDeviceMemHandler(hipstatevecHandle_t, hipstatevecDeviceMemHandler_t*)       { HSV_NL; }
hipstatevecStatus_t hipstatevecLoggerSetCallback(hipstatevecLoggerCallback_t)       { HSV_NL; }
hipstatevecStatus_t hipstatevecLoggerSetCallbackData(hipstatevecLoggerCallbackData_t, void*) { HSV_NL; }
hipstatevecStatus_t hipstatevecLoggerSetFile(void*)                                  { HSV_NL; }
hipstatevecStatus_t hipstatevecLoggerOpenFile(const char*)                           { HSV_NL; }
hipstatevecStatus_t hipstatevecLoggerSetLevel(int32_t)                               { HSV_NL; }
hipstatevecStatus_t hipstatevecLoggerSetMask(int32_t)                                { HSV_NL; }
hipstatevecStatus_t hipstatevecLoggerForceDisable(void)                              { HSV_NL; }

hipstatevecStatus_t hipstatevecInitializeStateVector(hipstatevecHandle_t, void*, hipstatevecDataType_t, uint32_t, hipstatevecStateVectorType_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecApplyMatrixGetWorkspaceSize(hipstatevecHandle_t, hipstatevecDataType_t, uint32_t, const void*, hipstatevecDataType_t, hipstatevecMatrixLayout_t, int32_t, uint32_t, uint32_t, hipstatevecComputeType_t, size_t*) { HSV_NL; }
hipstatevecStatus_t hipstatevecApplyMatrix(hipstatevecHandle_t, void*, hipstatevecDataType_t, uint32_t, const void*, hipstatevecDataType_t, hipstatevecMatrixLayout_t, int32_t, const int32_t*, uint32_t, const int32_t*, const int32_t*, uint32_t, hipstatevecComputeType_t, void*, size_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecApplyPauliRotation(hipstatevecHandle_t, void*, hipstatevecDataType_t, uint32_t, double, const hipstatevecPauli_t*, const int32_t*, uint32_t, const int32_t*, const int32_t*, uint32_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(hipstatevecHandle_t, hipstatevecDataType_t, uint32_t, const hipstatevecIndex_t*, const void*, hipstatevecDataType_t, const int32_t*, uint32_t, uint32_t, size_t*) { HSV_NL; }
hipstatevecStatus_t hipstatevecApplyGeneralizedPermutationMatrix(hipstatevecHandle_t, void*, hipstatevecDataType_t, uint32_t, const hipstatevecIndex_t*, const void*, hipstatevecDataType_t, int32_t, const int32_t*, uint32_t, const int32_t*, const int32_t*, uint32_t, void*, size_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecAbsorbDiagonalMatrix(hipstatevecHandle_t, void*, hipstatevecDataType_t, uint32_t, const void*, hipstatevecDataType_t, int32_t, const int32_t*, uint32_t, const int32_t*, const int32_t*, uint32_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecApplyMatrixBatchedGetWorkspaceSize(hipstatevecHandle_t, hipstatevecDataType_t, uint32_t, uint32_t, hipstatevecIndex_t, hipstatevecMatrixMapType_t, const int32_t*, const void*, hipstatevecDataType_t, hipstatevecMatrixLayout_t, int32_t, uint32_t, uint32_t, uint32_t, hipstatevecComputeType_t, size_t*) { HSV_NL; }
hipstatevecStatus_t hipstatevecApplyMatrixBatched(hipstatevecHandle_t, void*, hipstatevecDataType_t, uint32_t, uint32_t, hipstatevecIndex_t, hipstatevecMatrixMapType_t, const int32_t*, const void*, hipstatevecDataType_t, hipstatevecMatrixLayout_t, int32_t, uint32_t, const int32_t*, uint32_t, const int32_t*, const int32_t*, uint32_t, hipstatevecComputeType_t, void*, size_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecAbs2SumArray(hipstatevecHandle_t, const void*, hipstatevecDataType_t, uint32_t, double*, const int32_t*, uint32_t, const int32_t*, const int32_t*, uint32_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecAbs2SumOnZBasis(hipstatevecHandle_t, const void*, hipstatevecDataType_t, uint32_t, double*, double*, const int32_t*, uint32_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecMeasureOnZBasis(hipstatevecHandle_t, void*, hipstatevecDataType_t, uint32_t, int32_t*, const int32_t*, uint32_t, double, hipstatevecCollapseOp_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecBatchMeasure(hipstatevecHandle_t, void*, hipstatevecDataType_t, uint32_t, int32_t*, const int32_t*, uint32_t, double, hipstatevecCollapseOp_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecBatchMeasureWithOffset(hipstatevecHandle_t, void*, hipstatevecDataType_t, uint32_t, int32_t*, const int32_t*, uint32_t, double, hipstatevecCollapseOp_t, double, double) { HSV_NL; }
hipstatevecStatus_t hipstatevecCollapseOnZBasis(hipstatevecHandle_t, void*, hipstatevecDataType_t, uint32_t, int32_t, const int32_t*, uint32_t, double) { HSV_NL; }
hipstatevecStatus_t hipstatevecCollapseByBitString(hipstatevecHandle_t, void*, hipstatevecDataType_t, uint32_t, const int32_t*, const int32_t*, uint32_t, double) { HSV_NL; }
hipstatevecStatus_t hipstatevecSamplerCreate(hipstatevecHandle_t, const void*, hipstatevecDataType_t, uint32_t, hipstatevecSamplerDescriptor_t*, uint32_t, size_t*) { HSV_NL; }
hipstatevecStatus_t hipstatevecSamplerDestroy(hipstatevecSamplerDescriptor_t) { return HIPSTATEVEC_STATUS_SUCCESS; }
hipstatevecStatus_t hipstatevecSamplerPreprocess(hipstatevecHandle_t, hipstatevecSamplerDescriptor_t, void*, size_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecSamplerGetSquaredNorm(hipstatevecHandle_t, hipstatevecSamplerDescriptor_t, double*) { HSV_NL; }
hipstatevecStatus_t hipstatevecSamplerApplySubSVOffset(hipstatevecHandle_t, hipstatevecSamplerDescriptor_t, int32_t, uint32_t, double, double) { HSV_NL; }
hipstatevecStatus_t hipstatevecSamplerSample(hipstatevecHandle_t, hipstatevecSamplerDescriptor_t, hipstatevecIndex_t*, const int32_t*, uint32_t, const double*, uint32_t, hipstatevecSamplerOutput_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecComputeExpectationGetWorkspaceSize(hipstatevecHandle_t, hipstatevecDataType_t, uint32_t, const void*, hipstatevecDataType_t, hipstatevecMatrixLayout_t, uint32_t, hipstatevecComputeType_t, size_t*) { HSV_NL; }
hipstatevecStatus_t hipstatevecComputeExpectation(hipstatevecHandle_t, const void*, hipstatevecDataType_t, uint32_t, void*, hipstatevecDataType_t, double*, const void*, hipstatevecDataType_t, hipstatevecMatrixLayout_t, const int32_t*, uint32_t, hipstatevecComputeType_t, void*, size_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecComputeExpectationsOnPauliBasis(hipstatevecHandle_t, const void*, hipstatevecDataType_t, uint32_t, double*, const hipstatevecPauli_t* const*, uint32_t, const int32_t* const*, const uint32_t*) { HSV_NL; }
hipstatevecStatus_t hipstatevecAccessorCreate(hipstatevecHandle_t, void*, hipstatevecDataType_t, uint32_t, hipstatevecAccessorDescriptor_t*, const int32_t*, uint32_t, const int32_t*, const int32_t*, uint32_t, size_t*) { HSV_NL; }
hipstatevecStatus_t hipstatevecAccessorCreateView(hipstatevecHandle_t, const void*, hipstatevecDataType_t, uint32_t, hipstatevecAccessorDescriptor_t*, const int32_t*, uint32_t, const int32_t*, const int32_t*, uint32_t, size_t*) { HSV_NL; }
hipstatevecStatus_t hipstatevecAccessorDestroy(hipstatevecAccessorDescriptor_t) { return HIPSTATEVEC_STATUS_SUCCESS; }
hipstatevecStatus_t hipstatevecAccessorSetExtraWorkspace(hipstatevecHandle_t, hipstatevecAccessorDescriptor_t, void*, size_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecAccessorGet(hipstatevecHandle_t, hipstatevecAccessorDescriptor_t, void*, hipstatevecIndex_t, hipstatevecIndex_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecAccessorSet(hipstatevecHandle_t, hipstatevecAccessorDescriptor_t, const void*, hipstatevecIndex_t, hipstatevecIndex_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecSwapIndexBits(hipstatevecHandle_t, void*, hipstatevecDataType_t, uint32_t, const hipstatevecIndexPair_t*, uint32_t, const int32_t*, const int32_t*, uint32_t) { HSV_NL; }
hipstatevecStatus_t hipstatevecTestMatrixTypeGetWorkspaceSize(hipstatevecHandle_t, hipstatevecMatrixType_t, const void*, hipstatevecDataType_t, hipstatevecMatrixLayout_t, uint32_t, int32_t, hipstatevecComputeType_t, size_t*) { HSV_NL; }
hipstatevecStatus_t hipstatevecTestMatrixType(hipstatevecHandle_t, double*, hipstatevecMatrixType_t, const void*, hipstatevecDataType_t, hipstatevecMatrixLayout_t, uint32_t, int32_t, hipstatevecComputeType_t, void*, size_t) { HSV_NL; }

#undef HSV_NL

} // extern "C"

#endif /* HIPSTATEVEC_HAS_CUSTATEVEC */
