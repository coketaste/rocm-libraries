/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Bell-pair construction and Z-basis measurement using rocSTATEVEC.
 * Runs an H on qubit 0, a CX from 0 to 1, then measures both qubits
 * in the Z basis without collapse.
 * ************************************************************************ */

#include <rocstatevec.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIPCHK(expr) do { hipError_t e = (expr); if(e != hipSuccess) { std::fprintf(stderr, "HIP error %d at %s:%d\n", (int)e, __FILE__, __LINE__); return 1; } } while(0)
#define SVCHK(expr)  do { rocstatevec_status s = (expr); if(s != ROCSTATEVEC_STATUS_SUCCESS) { std::fprintf(stderr, "rocSTATEVEC error %s: %s\n", rocstatevec_get_error_name(s), rocstatevec_get_error_string(s)); return 1; } } while(0)

int main()
{
    using cd = std::complex<double>;
    constexpr uint32_t n_qubits = 2;
    constexpr size_t   N        = size_t{1} << n_qubits;

    rocstatevec_handle h = nullptr;
    SVCHK(rocstatevec_create_handle(&h));

    cd* dsv = nullptr;
    HIPCHK(hipMalloc(&dsv, N * sizeof(cd)));
    SVCHK(rocstatevec_initialize_state_vector(h, dsv, ROCSTATEVEC_C_64F, n_qubits, ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO));

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    cd            H_mat[4] = {{inv_sqrt2, 0}, {inv_sqrt2, 0}, {inv_sqrt2, 0}, {-inv_sqrt2, 0}};
    cd            X_mat[4] = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    int32_t       q0       = 0;
    int32_t       q1       = 1;

    SVCHK(rocstatevec_apply_matrix(h, dsv, ROCSTATEVEC_C_64F, n_qubits,
                                   H_mat, ROCSTATEVEC_C_64F, ROCSTATEVEC_MATRIX_LAYOUT_ROW,
                                   /*adjoint*/ 0, &q0, 1, nullptr, nullptr, 0,
                                   ROCSTATEVEC_COMPUTE_64F, nullptr, 0));

    SVCHK(rocstatevec_apply_matrix(h, dsv, ROCSTATEVEC_C_64F, n_qubits,
                                   X_mat, ROCSTATEVEC_C_64F, ROCSTATEVEC_MATRIX_LAYOUT_ROW,
                                   /*adjoint*/ 0, &q1, 1, &q0, nullptr, 1,
                                   ROCSTATEVEC_COMPUTE_64F, nullptr, 0));

    std::vector<cd> hsv(N);
    HIPCHK(hipMemcpy(hsv.data(), dsv, N * sizeof(cd), hipMemcpyDeviceToHost));
    std::printf("Bell pair amplitudes:\n");
    for(size_t i = 0; i < N; ++i)
        std::printf("  |%zu> -> (%.4f, %.4f)\n", i, hsv[i].real(), hsv[i].imag());

    HIPCHK(hipFree(dsv));
    SVCHK(rocstatevec_destroy_handle(h));
    return 0;
}
