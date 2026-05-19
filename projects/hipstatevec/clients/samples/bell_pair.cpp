/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Bell-pair sample using the hipSTATEVEC camelCase API. Identical
 * structure to the rocSTATEVEC sample; the source is portable across
 * AMD and NVIDIA backends.
 * ************************************************************************ */

#include <hipstatevec.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <complex>
#include <cstdio>

#define HIPCHK(expr) do { hipError_t e = (expr); if(e != hipSuccess) { std::fprintf(stderr, "HIP %d at %s:%d\n", (int)e, __FILE__, __LINE__); return 1; } } while(0)
#define SVCHK(expr)  do { hipstatevecStatus_t s = (expr); if(s != HIPSTATEVEC_STATUS_SUCCESS) { std::fprintf(stderr, "%s: %s\n", hipstatevecGetErrorName(s), hipstatevecGetErrorString(s)); return 1; } } while(0)

int main()
{
    using cd = std::complex<double>;
    constexpr uint32_t n_qubits = 2;
    constexpr size_t   N        = size_t{1} << n_qubits;

    hipstatevecHandle_t h = nullptr;
    SVCHK(hipstatevecCreate(&h));

    cd* dsv = nullptr;
    HIPCHK(hipMalloc(&dsv, N * sizeof(cd)));
    SVCHK(hipstatevecInitializeStateVector(h, dsv, HIPSTATEVEC_C_64F, n_qubits, HIPSTATEVEC_STATE_VECTOR_TYPE_ZERO));

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    cd            H_mat[4] = {{inv_sqrt2, 0}, {inv_sqrt2, 0}, {inv_sqrt2, 0}, {-inv_sqrt2, 0}};
    cd            X_mat[4] = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    int32_t       q0 = 0;
    int32_t       q1 = 1;

    SVCHK(hipstatevecApplyMatrix(h, dsv, HIPSTATEVEC_C_64F, n_qubits, H_mat, HIPSTATEVEC_C_64F,
                                 HIPSTATEVEC_MATRIX_LAYOUT_ROW, 0, &q0, 1, nullptr, nullptr, 0,
                                 HIPSTATEVEC_COMPUTE_64F, nullptr, 0));
    SVCHK(hipstatevecApplyMatrix(h, dsv, HIPSTATEVEC_C_64F, n_qubits, X_mat, HIPSTATEVEC_C_64F,
                                 HIPSTATEVEC_MATRIX_LAYOUT_ROW, 0, &q1, 1, &q0, nullptr, 1,
                                 HIPSTATEVEC_COMPUTE_64F, nullptr, 0));

    HIPCHK(hipFree(dsv));
    SVCHK(hipstatevecDestroy(h));
    std::printf("hipstatevec bell pair scaffold ok\n");
    return 0;
}
