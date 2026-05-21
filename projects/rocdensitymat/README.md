# rocDENSITYMAT

rocDENSITYMAT is an AMD-native, ROCm/HIP quantum density-matrix
simulation library for analog quantum-dynamics workloads. v0.1.0
mirrors the public C API of NVIDIA's cuDensityMat v0.1.0 (cuQuantum
24.11) so ports of cuDensityMat consumers can replace `cudensitymat`
calls with the matching `rocdensitymat`/`hipdensitymat` calls without
changing program structure.

## Status

`v0.1.0` is a single-GPU release covering:

- Library context, workspace, logger, memory handler, and version query.
- Pure (`PURITY_PURE`) and mixed (`PURITY_MIXED`) states on dense
  device tensors with norm / trace / overlap reductions.
- Elementary operators (Identity / Pauli / dense / diagonal),
  `OperatorTerm` products, and full `Operator` sums.
- `OperatorComputeAction` (`-i H |psi>` pure, `-i [H, rho]` mixed) and
  `OperatorComputeExpectation`.
- Fixed-step RK4 master-equation stepping
  (`CreateMasterEquationSolver` + `MasterEquationStep[N]`).

The following surface is wired but currently returns
`ROCDENSITYMAT_STATUS_NOT_SUPPORTED`, with a v0.2 implementation plan:

- Lindblad collapse channels inside `OperatorComputeAction`.
- `OperatorComputeEigenspectrum`.
- `OperatorComputeActionBackwardDiff`.
- Multi-GPU multi-node distribution (`ResetDistributedConfiguration`,
  `GetNumRanks`, `GetProcRank`).
- `PURITY_MPS` matrix-product-state operator action.
- Adaptive ODE integrators.

## Building

rocDENSITYMAT depends on `hip::device` and `roc::rocprim`. Build under
the `rocm-libraries` superbuild via `THEROCK_ENABLE_DENSITYMAT=ON`, or
stand-alone:

```sh
cmake -S projects/rocdensitymat -B build/rocdensitymat \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_CLIENTS_TESTS=ON \
  -DBUILD_CLIENTS_SAMPLES=ON
cmake --build build/rocdensitymat
ctest --test-dir build/rocdensitymat --output-on-failure
```

The matching ABI-cast HIP wrapper lives in `projects/hipdensitymat/`.
