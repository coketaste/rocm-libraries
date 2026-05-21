/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * The unitary Liouvillian (-i [H, .] for mixed; -i H |psi> for pure) is
 * synthesized on-the-fly by `rocdensitymat_operator_compute_action`.
 * v0.1.0 does not need a separate `Liouvillian` object on the public API
 * surface because cuDensityMat 0.1.0 also folds it into the `Operator`
 * abstraction via `OperatorAppendTerm` with a Hermitian H. Lindblad
 * collapse contributions arrive as terms with non-zero `duality_offset`
 * and are rejected during compute (returns NOT_SUPPORTED).
 *
 * The translation unit is kept in the build because it gives the linker
 * an obvious anchor point for the unitary-only Liouvillian compute path
 * and is the natural home for any future Lindblad helpers in v0.2.
 * ************************************************************************ */

#include "rocdensitymat_descriptors.hpp"

namespace rocdensitymat
{

bool operator_is_unitary_only(rocdensitymat_operator op)
{
    return !op->has_collapse_term();
}

} // namespace rocdensitymat
