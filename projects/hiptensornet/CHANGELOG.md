# Changelog for hipTENSORNET

## (Unreleased) hipTENSORNET 0.1.0

Initial scaffold release. AMD-ROCm-only.

### Added

* Public C API in `library/include/hiptensornet.h`. Function
  correspondence with cuTensorNet 2.x is bijective and lexical
  (`s/cutensornet/hiptensornet_/`, snake_case).
* AMD pass-through to rocTENSORNET (`library/src/amd_detail/`),
  layout-compatible handle types and bijective enum values so
  forwarding is `reinterpret_cast` + `static_cast` only.
* CMake target `roc::hiptensornet` with `VERSION` / `SOVERSION`
  attached so distros can package it.

### Status

* Shipping (parity with rocTENSORNET v0.1.0): handle / library
  lifecycle, network and tensor descriptors, optimizer config / info,
  workspace queries, sliced contraction execute / autotune, host-side
  Jacobi tensor SVD / QR / gate-split, network-operator descriptors,
  accessor / marginal / sampler / expectation lifecycle and
  configuration.
* Stubbed in v0.1.0 (return `HIPTENSORNET_STATUS_NOT_SUPPORTED`):
  `state_prepare`, `state_compute`, `marginal_prepare`,
  `marginal_compute`, `sampler_prepare`, `sampler_sample`,
  `expectation_prepare`, `expectation_compute`, `accessor_prepare`,
  `accessor_compute`, `state_compute_gradients_backward`. These are
  all behind the gate-list-to-network synthesizer that lands in v0.2.

### Notes

* AMD-only: there is no NVIDIA / CUDA / cuTensorNet backend in this
  build. Headers are self-contained and do not require `<cutensornet.h>`
  to be installed.
