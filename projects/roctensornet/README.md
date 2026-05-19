# rocTENSORNET

`rocTENSORNET` is the AMD-native, single-node tensor-network contraction
library that mirrors the cuTensorNet public C API
(cuQuantum 24.11, cuTensorNet 2.x). It is a clean-room implementation
reverse-engineered from the cuTensorNet documentation; no NVIDIA
headers are redistributed.

The library is the AMD half of the
[`hipTENSORNET`](../hiptensornet/) pair, following the same
`rocSPARSE` / `hipSPARSE` and `rocSTATEVEC` / `hipSTATEVEC` pattern used
elsewhere in ROCm.

## API fidelity contract

The function-name correspondence with cuTensorNet 2.x is **bijective and
lexical**: every public entry point `cutensornetFoo` has a `roctensornet_foo`
counterpart with the same parameter list and the same documented
semantics. The translation rules are:

| cuTensorNet                                 | rocTENSORNET                                  |
|---------------------------------------------|------------------------------------------------|
| `cutensornetFoo`                            | `roctensornet_foo` (snake_case)                |
| `cutensornetStatus_t`                       | `roctensornet_status`                          |
| `cutensornetHandle_t`                       | `roctensornet_handle`                          |
| `cutensornetNetworkDescriptor_t`            | `roctensornet_network_descriptor`              |
| `cutensornetTensorDescriptor_t`             | `roctensornet_tensor_descriptor`               |
| `cutensornetContractionOptimizerConfig_t`   | `roctensornet_contraction_optimizer_config`    |
| `cutensornetContractionOptimizerInfo_t`     | `roctensornet_contraction_optimizer_info`      |
| `cutensornetWorkspaceDescriptor_t`          | `roctensornet_workspace_descriptor`            |
| `cutensornetContractionPlan_t`              | `roctensornet_contraction_plan`                |
| `cutensornetSliceGroup_t`                   | `roctensornet_slice_group`                     |
| `cutensornetTensorSVDConfig_t`              | `roctensornet_tensor_svd_config`               |
| `cutensornetTensorSVDInfo_t`                | `roctensornet_tensor_svd_info`                 |
| `cutensornetNetworkOperator_t`              | `roctensornet_network_operator`                |
| `cutensornetState_t`                        | `roctensornet_state`                           |
| `cutensornetStateMarginal_t`                | `roctensornet_state_marginal`                  |
| `cutensornetStateSampler_t`                 | `roctensornet_state_sampler`                   |
| `cutensornetStateExpectation_t`             | `roctensornet_state_expectation`               |
| `cutensornetStateAccessor_t`                | `roctensornet_state_accessor`                  |
| `cutensornetDeviceMemHandler_t`             | `roctensornet_device_mem_handler_t`            |
| `CUTENSORNET_*` enum tokens                 | `ROCTENSORNET_*` (uppercase, snake)            |

The mechanical mapping means an existing cuTensorNet consumer can be
rebuilt on AMD by running `sed 's/cutensornet/roctensornet/g'` over its
sources (and similarly for the type/enum tokens above).

## Conformance contract

* **Tensor data** (`raw_data_in[]`, `raw_data_out`, gradients) is
  treated as device-resident per the cuTensorNet contract. Descriptor
  inputs (modes, extents, strides, attributes) are host-side.
* **Workspace queries** return real non-zero byte counts derived from
  the optimizer info and slice plan. Passing a queried buffer back via
  `roctensornet_workspace_set_memory` causes the call to consume it
  directly without falling back to `hipMallocAsync`.
* **Device memory handler** registered via
  `roctensornet_set_device_mem_handler` is honored for every internal
  allocation (workspace fallback, autotune temporaries, SVD/QR scratch,
  state-builder intermediates).
* **Streams** — all async HIP work uses the stream bound to the handle.

## Status

In progress. Single-node-only v0.1.0. Distributed APIs are deferred.

## Building stand-alone

```bash
cmake -S projects/roctensornet -B build/roctensornet \
      -DBUILD_CLIENTS_TESTS=ON
cmake --build build/roctensornet -j
ctest --test-dir build/roctensornet/clients/tests --output-on-failure
```

## Building in the rocm-libraries superbuild

```bash
cmake -S . -B build -DTHEROCK_ENABLE_TENSORNET=ON
cmake --build build -j
```
