# Changelog for hipDENSITYMAT

## (Unreleased) hipDENSITYMAT 0.1.0

### Added

* First public release of hipDENSITYMAT.
* AMD-only thin C wrapper that mirrors the cuDensityMat 0.1.0 / cuQuantum
  24.11 camelCase API exactly. A `s/cu/hip/g` rename of consumer code
  compiles unchanged against `<hipdensitymat.h>`.
* Every entry point forwards to the in-tree `rocdensitymat` library via
  `reinterpret_cast` / `static_cast`; no NVIDIA runtime is linked.
* Conformance tests pinning the numeric values of every shared
  status code and enum so the ABI cannot drift away from cuDensityMat.

### Deferred to v0.2

The deferral list is identical to rocDENSITYMAT: Lindblad-collapse
compute path, eigenspectrum, backward differentiation, MGMN distributed
configuration, MPS states, and adaptive ODE stepping. The corresponding
symbols are exposed but return `HIPDENSITYMAT_STATUS_NOT_SUPPORTED`.
