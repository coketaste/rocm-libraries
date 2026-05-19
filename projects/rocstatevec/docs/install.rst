.. meta::
  :description: rocSTATEVEC build and install
  :keywords: rocSTATEVEC, install, build, cmake, ROCm

.. _rocstatevec-install:

Building and installing rocSTATEVEC
====================================

Prerequisites
-------------

* ROCm 6.4 or later, including ``hip``, ``rocprim``, ``rocthrust``, ``hipcub``.
* A C++17-capable compiler (``hipcc`` or ``amdclang++``).
* CMake 3.22+.

Configure and build
-------------------

.. code-block:: bash

   cd projects/rocstatevec
   cmake -S . -B build \
     -DCMAKE_BUILD_TYPE=Release \
     -DAMDGPU_TARGETS="gfx90a;gfx942;gfx1100;gfx1200" \
     -DBUILD_CLIENTS_TESTS=ON \
     -DBUILD_CLIENTS_SAMPLES=ON
   cmake --build build -j

Install
-------

.. code-block:: bash

   sudo cmake --install build

Run the test suite
------------------

.. code-block:: bash

   cd build && ctest --output-on-failure

Run the consumer-readiness sample
---------------------------------

The ``cudaq_pattern_demo`` exercises the 12 cuStateVec entry points that
``cuda-quantum`` consumes today, ported one-for-one to rocSTATEVEC. A
successful run is the green-light signal that a future ``rocm-quantum``
project can consume rocSTATEVEC end-to-end.

.. code-block:: bash

   ./build/clients/staging/rocstatevec-cudaq-pattern-demo
