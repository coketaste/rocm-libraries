/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

/*! \file
 *  \brief hipdensitymat.h is the umbrella header for the hipDENSITYMAT wrapper.
 *
 *  Including this header is sufficient to reach the entire public C API.
 *  Function names use camelCase exactly as cuDensityMat does (cuQuantum
 *  24.11 / cuDensityMat 0.1.0), so a `s/cu/hip/g` rewrite of consumer
 *  code compiles unchanged. The library is built and runs only against
 *  AMD ROCm; it forwards every entry point to rocDENSITYMAT.
 */

#ifndef HIPDENSITYMAT_H
#define HIPDENSITYMAT_H

#include "hipdensitymat-version.h"
#include "hipdensitymat-types.h"
#include "hipdensitymat-auxiliary.h"
#include "hipdensitymat-functions.h"

#endif /* HIPDENSITYMAT_H */
