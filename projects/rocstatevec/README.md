# rocSTATEVEC

rocSTATEVEC is an AMD GPU state-vector simulation library for quantum computing.
It provides a C API for allocating, evolving, measuring, and sampling from
quantum state vectors on AMD GPUs via HIP. It is part of the
[ROCm Libraries](https://github.com/ROCm/rocm-libraries) super-repo.

rocSTATEVEC mirrors the public API of NVIDIA's
[cuStateVec](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/) library
shipped with the cuQuantum SDK. Function-name correspondence is bijective and
lexical — every `custatevecXxx` entry point in cuStateVec 1.7.x has exactly one
`rocstatevec_xxx` (snake_case) counterpart, and every type and enum value maps
1:1. The companion [hipSTATEVEC](../hipstatevec) project provides a thin
camelCase wrapper that selects between the AMD (`rocstatevec`) and NVIDIA
(`cuStateVec`) backends at build time.

## Specification source

rocSTATEVEC v0.x and v1.0 target the public API of cuStateVec **1.7.x**
(shipped in cuQuantum 24.11). MGMN/distributed APIs are deferred to a later
major version.

## Documentation

> [!NOTE]
> The published documentation is available at
> [rocSTATEVEC documentation](https://rocm.docs.amd.com/projects/rocstatevec/en/latest/)
> in an organized, easy-to-read format with search and a table of contents.

To build documentation locally, run the following code from the project root:

```bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Requirements

* AMD ROCm 6.0 or later.
* HIP runtime + clang-based compiler (`hipcc`).
* CMake 3.22 or later.
* `rocprim` >= 4.0.0 (sampler scans, batch reductions).
* `hipcub` >= 4.0.0 (one-shot reductions).
* `rocthrust` >= 4.0.0 (complex-number utilities).

## Building

```bash
cd projects/rocstatevec
./install.sh -i              # build + install to /opt/rocm
```

Or with CMake directly:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

## Testing

```bash
cd build
ctest --output-on-failure
```

## License

Distributed under the [MIT License](LICENSE.md).
