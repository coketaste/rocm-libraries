# Changelog for rocTENSORNET

Documentation for rocTENSORNET is available at
[https://rocm.docs.amd.com/projects/roctensornet/en/latest/](https://rocm.docs.amd.com/projects/roctensornet/en/latest/).

## (Unreleased) rocTENSORNET 0.1.0

### Added

* Initial scaffolding mirroring the rocSPARSE / rocSTATEVEC pair layout.
* Public C API in `roctensornet/library/include/`. Function-name correspondence
  with cuTensorNet 2.x (cuQuantum 24.11) is bijective and lexical; see the
  API fidelity contract in `README.md`.
* Library / handle: `roctensornet_create`, `roctensornet_destroy`,
  `roctensornet_get_version`, `roctensornet_get_hip_runtime_version`,
  `roctensornet_get_error_string`, `roctensornet_set_stream`,
  `roctensornet_get_stream`, `roctensornet_logger_*`,
  `roctensornet_set_device_mem_handler`, `roctensornet_get_device_mem_handler`.
* Network / tensor descriptor: `roctensornet_create_network_descriptor`,
  `roctensornet_destroy_network_descriptor`,
  `roctensornet_get_output_tensor_descriptor`,
  `roctensornet_create_tensor_descriptor`,
  `roctensornet_destroy_tensor_descriptor`,
  `roctensornet_get_tensor_details`,
  `roctensornet_network_{get,set}_attribute`.
* Contraction optimizer: config + info entry points,
  `roctensornet_contraction_optimize` (greedy path finder), pack/unpack of
  info blobs.
* Workspace descriptor: `roctensornet_create_workspace_descriptor`,
  `_destroy_workspace_descriptor`,
  `_workspace_compute_{contraction,svd,qr}_sizes`,
  `_workspace_get_memory_size`, `_workspace_{set,get}_memory`,
  `_workspace_purge_cache`.
* Contraction: plan, autotune preference, autotune, single-slice contraction
  and `_contract_slices` with optional accumulate.
* Slice group: `_create_slice_group_from_id_range`,
  `_create_slice_group_from_ids`, `_destroy_slice_group`.
* Tensor SVD / QR: config + info entry points, `_tensor_svd`, `_tensor_qr`,
  `_gate_split`.
* Gradient: `_compute_gradients_backward`.
* Network operator: `_create_network_operator`, `_destroy_network_operator`,
  `_network_operator_append_product`, `_network_operator_append_mpo`.
* Network state: `_create_state`, `_destroy_state`,
  `_state_apply_tensor[_operator]`, `_state_apply_controlled_tensor_operator`,
  `_state_apply_unitary_channel`, `_state_apply_general_channel`,
  `_state_update_tensor_operator`, `_state_configure`, `_state_get_info`,
  `_state_compute`.
* Network-state derived computations: Marginal, Sampler, Expectation,
  Accessor (each with `_create`, `_destroy`, `_configure`, `_get_info`,
  `_prepare`, `_compute` or `_sample`).

### Notes

* Single-node only. MGMN / distributed APIs (`*Communicator*`,
  `*ParallelConfig`, distributed slice schedulers) are deferred to a
  later major version.
* rocTENSORNET is a clean-room implementation reverse-engineered from the
  cuTensorNet public API documentation; no NVIDIA headers are
  redistributed.
* The contraction path finder is a greedy heuristic and intended to be
  replaced by a more sophisticated optimizer in a follow-up release.
