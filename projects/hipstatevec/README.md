# hipSTATEVEC

hipSTATEVEC is the portable thin wrapper that exposes a state-vector
simulation API in the same shape as
[NVIDIA cuStateVec](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/),
dispatching to either of two backends at build time:

* **AMD backend** (`HIPSTATEVEC_ENABLE_HIP=ON`, default) — calls
  [rocSTATEVEC](../rocstatevec).
* **NVIDIA backend** (`HIPSTATEVEC_ENABLE_CUDA=ON`) — calls cuStateVec
  from the cuQuantum SDK directly.

Function-name correspondence is bijective and lexical with cuStateVec
1.7.x: every `custatevecXxx` entry point has exactly one `hipstatevecXxx`
counterpart with the same suffix in camelCase, and every type and enum
value maps 1:1.

## Header independence

Including `<hipstatevec.h>` does **not** require `<custatevec.h>` to be
installed on AMD systems. The header is rebuilt entirely from the public
cuStateVec API documentation; no NVIDIA header is redistributed.

## Building

```bash
cd projects/hipstatevec
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..    # AMD by default
cmake --build . -j
```

To build against cuStateVec on an NVIDIA host:

```bash
cmake -DHIPSTATEVEC_ENABLE_HIP=OFF -DHIPSTATEVEC_ENABLE_CUDA=ON ..
```

## License

Distributed under the [MIT License](LICENSE.md).
