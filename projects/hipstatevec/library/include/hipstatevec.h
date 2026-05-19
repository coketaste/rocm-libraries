/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

/*! \file
 *  \brief hipstatevec.h is the umbrella header for the hipSTATEVEC wrapper.
 *
 *  Including this header is sufficient to reach the entire public C API.
 *  Function names use camelCase exactly as cuStateVec does (cuQuantum
 *  24.11 / cuStateVec 1.7.x), so a `s/cu/hip/g` rewrite of consumer code
 *  compiles unchanged. The library is built and runs only against AMD
 *  ROCm; it forwards every entry point to rocSTATEVEC.
 */

#ifndef HIPSTATEVEC_H
#define HIPSTATEVEC_H

#include "hipstatevec-version.h"
#include "hipstatevec-types.h"
#include "hipstatevec-auxiliary.h"
#include "hipstatevec-functions.h"

#endif /* HIPSTATEVEC_H */
