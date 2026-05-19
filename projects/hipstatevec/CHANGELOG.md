# Changelog for hipSTATEVEC

Documentation for hipSTATEVEC is available at
[https://rocm.docs.amd.com/projects/hipstatevec/en/latest/](https://rocm.docs.amd.com/projects/hipstatevec/en/latest/).

## (Unreleased) hipSTATEVEC 0.1.0

### Conformance / behavior fixes

* All conformance fixes landed in rocSTATEVEC are reflected through the
  AMD pass-through: workspace-size queries report real byte counts,
  device-resident matrix pointers are auto-detected, and a
  `hipstatevecDeviceMemHandler_t` registered via
  `hipstatevecSetDeviceMemHandler` is now actually invoked for every
  internal allocation.

### Added

* Initial scaffolding mirroring the hipSPARSE/rocSPARSE pair layout.
* Public C API in `hipstatevec/library/include/`. Function correspondence
  with cuStateVec 1.7.x (cuQuantum 24.11) is bijective and lexical:
  `s/cu/hip/` rewrites a consumer compiled against `<custatevec.h>` to a
  consumer of `<hipstatevec.h>`.
* AMD backend: pass-through to rocSTATEVEC, gated on `HIPSTATEVEC_ENABLE_HIP=ON`.
* NVIDIA backend: pass-through to cuStateVec, gated on `HIPSTATEVEC_ENABLE_CUDA=ON`.

### Notes

* Single-node only. MGMN/distributed APIs are deferred to a later major version.
* Headers are independent: including `<hipstatevec.h>` does **not** require
  `<custatevec.h>` to be installed on AMD, and does not require
  rocSTATEVEC headers on NVIDIA.
