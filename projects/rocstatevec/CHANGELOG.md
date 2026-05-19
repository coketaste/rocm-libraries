# Changelog for rocSTATEVEC

Documentation for rocSTATEVEC is available at
[https://rocm.docs.amd.com/projects/rocstatevec/en/latest/](https://rocm.docs.amd.com/projects/rocstatevec/en/latest/).

## (Unreleased) rocSTATEVEC 0.1.0

### Conformance / behavior fixes

* `apply_matrix`, `apply_matrix_batched`, `compute_expectation`,
  `apply_generalized_permutation_matrix`, `absorb_diagonal_matrix`, and
  `test_matrix_type` now accept a matrix pointer that is host-resident
  *or* device-resident; the runtime auto-detects via
  `hipPointerGetAttributes` and skips the redundant H2D copy in the
  device case. Before the fix, a device pointer was silently
  `hipMemcpyHostToDevice`'d and produced garbage.
* `*_get_workspace_size` entry points now return a real, non-zero byte
  count derived from the call's parameters (targets array, optional
  matrix staging slot, scratch buffer). When the caller hands the
  queried buffer back via `extra_workspace`, the kernel uses it
  directly instead of issuing a fresh internal `hipMallocAsync`.
* Every internal device allocation now routes through the
  `rocstatevec_device_mem_handler_t` registered on the handle (when
  one is bound). Previously the handler was stored but never invoked.
  Sampler temporary state (`d_cum`) and per-call scratch are both
  covered.
* `rocstatevec_compute_expectation` now populates `residual_norm` with
  `||M - 0.5 * (M + M^H)||_F` instead of returning 0.

### Added

* Initial scaffolding mirroring the rocSPARSE/hipSPARSE pair layout.
* Public C API in `rocstatevec/library/include/`. Function-name correspondence with
  cuStateVec 1.7.x (cuQuantum 24.11) is bijective and lexical; see the API fidelity
  contract in `README.md`.
* Library/handle: `rocstatevec_create_handle`, `rocstatevec_destroy_handle`,
  `rocstatevec_get_version`, `rocstatevec_get_property`,
  `rocstatevec_set_stream`, `rocstatevec_get_stream`,
  `rocstatevec_get_error_string`, `rocstatevec_get_error_name`,
  `rocstatevec_logger_*`, `rocstatevec_set_device_mem_handler`,
  `rocstatevec_get_device_mem_handler`.
* State-vector ops: `rocstatevec_initialize_state_vector`,
  `rocstatevec_apply_matrix`(+workspace), `rocstatevec_apply_pauli_rotation`,
  `rocstatevec_apply_generalized_permutation_matrix`(+workspace),
  `rocstatevec_absorb_diagonal_matrix`,
  `rocstatevec_apply_matrix_batched`(+workspace).
* Measure / sample: `rocstatevec_measure_on_z_basis`,
  `rocstatevec_batch_measure[/_with_offset]`,
  `rocstatevec_collapse_on_z_basis`, `rocstatevec_collapse_by_bit_string`,
  `rocstatevec_abs2_sum_array`, `rocstatevec_abs2_sum_on_z_basis`,
  `rocstatevec_sampler_*`.
* Expectation: `rocstatevec_compute_expectation`(+workspace),
  `rocstatevec_compute_expectations_on_pauli_basis`.
* Accessor: `rocstatevec_accessor_*`.
* Permutation: `rocstatevec_swap_index_bits`.
* Validation: `rocstatevec_test_matrix_type`(+workspace).

### Notes

* Single-node only. MGMN/distributed APIs (`custatevecCommunicator*`,
  `MultiDeviceSwapIndexBits`, `DistIndexBitSwap*`, `SVSwapWorker*`) are deferred
  to a later major version.
* `rocstatevec` is a clean-room implementation reverse-engineered from the
  cuStateVec public API documentation; no NVIDIA headers are redistributed.
