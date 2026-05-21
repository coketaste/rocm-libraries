# rocDENSITYMAT changelog

Full documentation for rocDENSITYMAT lives at the rocDENSITYMAT documentation site.

## (Unreleased) rocDENSITYMAT 0.1.0

### Added

- Initial single-GPU release of the AMD-native quantum density-matrix
  simulation library, mirroring the public C API of NVIDIA's cuDensityMat
  v0.1.0 (cuQuantum 24.11) for ABI-compatible porting.
- Library context: `rocdensitymat_create`, `rocdensitymat_destroy`,
  `rocdensitymat_reset_random_seed`, version + property queries,
  user-supplied device-memory handler, and logger callback registration.
- Workspace descriptor with `MEMSPACE_DEVICE` + `WORKSPACE_KIND_SCRATCH`
  scratch-buffer query / set / get plumbed through every prepare path.
- Quantum state abstractions: `PURITY_PURE` (state vector) and
  `PURITY_MIXED` (density matrix) over dense `c64` / `c32` device tensors,
  with `Initialize{Zero,Uniform,Basis}`, `ComputeNorm`, `ComputeTrace`,
  and `ComputeOverlap` reductions.
- Operator algebra: `ElementaryOperator` (Identity / Pauli / dense /
  diagonal), `OperatorTerm` (product of elementary operators on disjoint
  modes plus a host-side time-dependent scalar callback), and `Operator`
  (sum of terms with per-term coefficients).
- Liouvillian assembly with the unitary `-i [H, .]` compute path enabled.
- `OperatorPrepareAction` + `OperatorComputeAction` for both pure and
  mixed states, reusing the rocSTATEVEC tensor-product apply pattern.
- `OperatorComputeExpectation` for `<psi|O|psi>` (pure) and `Tr(O rho)`
  (mixed) layered on rocBLAS reductions.
- `CreateMasterEquationSolver`, `MasterEquationStep`, and
  `MasterEquationStepN` driving a fixed-step RK4 integrator over the
  Liouvillian, with workspace-charged temporary state buffers.
- Three end-to-end smoke samples: single-qubit Rabi oscillator, two-qubit
  Bell-state coherent evolution, and a driven-harmonic-oscillator demo.
- ~13 GoogleTest fixtures discovered through `gtest_discover_tests`.

### Deferred to v0.2 (returns `ROCDENSITYMAT_STATUS_NOT_SUPPORTED`)

- Lindblad collapse channels in `OperatorComputeAction` for mixed states.
- `OperatorComputeEigenspectrum` and the matching prepare path.
- `OperatorComputeActionBackwardDiff`.
- Multi-GPU multi-node configuration: `ResetDistributedConfiguration`,
  `GetNumRanks`, `GetProcRank`, and any compute path requiring a
  multi-rank communicator.
- `PURITY_MPS` matrix-product-state representation and MPS-aware
  operator action.
- Adaptive ODE integrators (Dormand-Prince RK45 etc.).

The v0.2 milestone tracks cuDensityMat v0.3.x feature parity.
