# hipTENSORNET

`hipTENSORNET` is a thin wrapper that exposes a tensor-network API in
the same shape as
[NVIDIA cuTensorNet](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/),
forwarding every call to AMD's [`rocTENSORNET`](../roctensornet/) on a
HIP runtime. **It runs only on AMD ROCm** — there is no CUDA / NVIDIA
backend in this build.

The reason hipTENSORNET exists alongside rocTENSORNET is the same
reason hipBLAS / hipSPARSE / hipFFT / hipSTATEVEC exist alongside their
`roc*` peers: downstream code that originally targeted cuTensorNet can
be hipified by a mechanical `s/cu/hip/g` rename and link against
`roc::hiptensornet` to run unchanged on AMD GPUs.

## v0.1.0 status

This is the initial scaffold release. The API surface is complete; the
runtime behavior tracks rocTENSORNET, which means:

* **Shipping:** handle / library lifecycle, network and tensor
  descriptors, optimizer config / info, workspace queries, sliced
  contraction execute / autotune, host-side Jacobi tensor SVD / QR /
  gate-split, network-operator descriptors, accessor / marginal /
  sampler / expectation lifecycle and configuration.
* **Stubbed (returns `HIPTENSORNET_STATUS_NOT_SUPPORTED`):** state
  prepare / compute, marginal prepare / compute, sampler prepare /
  sample, expectation prepare / compute, accessor prepare / compute,
  gradient compute. These are all the entry points whose
  implementation requires the gate-list-to-network synthesizer that is
  scheduled for v0.2.

The `*_prepare` and `*_compute` calls are wired symmetrically: prepare
returns `NOT_SUPPORTED` whenever compute will. Callers cannot be
silently fooled by a green prepare into invoking a no-op compute.

## Naming convention

`cutensornet*` ↔ `hiptensornet_*` (snake_case, `hip` prefix).

The hipTENSORNET header is **self-contained**: building a consumer does
not require `<cutensornet.h>` to exist on the system.

## Building

```bash
cmake -S projects/hiptensornet -B build/hiptensornet \
      -DBUILD_CLIENTS_TESTS=ON
cmake --build build/hiptensornet -j
ctest --test-dir build/hiptensornet/clients/tests --output-on-failure
```

Under the rocm-libraries superbuild, hiptensornet builds together with
rocTENSORNET when tensornet is opted-in.
