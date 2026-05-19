# hipSTATEVEC

hipSTATEVEC is a thin wrapper that exposes a state-vector simulation API
in the same shape as
[NVIDIA cuStateVec](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/),
forwarding every call to AMD's [rocSTATEVEC](../rocstatevec) on a HIP
runtime. **It runs only on AMD ROCm** — there is no CUDA / NVIDIA
backend in this build.

The reason hipSTATEVEC exists alongside rocSTATEVEC is the same reason
hipBLAS / hipSPARSE / hipFFT exist alongside their `roc*` peers:
downstream code that originally targeted cuStateVec can hipify by a
mechanical `s/cu/hip/g` rename and link against `roc::hipstatevec` to
run unchanged on AMD GPUs.

Function-name correspondence with cuStateVec 1.7.x (cuQuantum 24.11) is
bijective and lexical: every `custatevecXxx` entry point has exactly one
`hipstatevecXxx` counterpart with the same suffix in camelCase, and
every type and enum value maps 1:1.

## Header independence

Including `<hipstatevec.h>` does **not** require `<custatevec.h>` to be
installed. The header is rebuilt entirely from the public cuStateVec API
documentation; no NVIDIA header is redistributed.

## Building

```bash
cd projects/hipstatevec
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

Under the rocm-libraries superbuild, hipstatevec is built together with
rocstatevec when `-DTHEROCK_ENABLE_STATEVEC=ON` is set on the
top-level CMake configure.

## License

Distributed under the [MIT License](LICENSE.md).
