.. meta::
  :description: rocDENSITYMAT build and install
  :keywords: rocDENSITYMAT, install, build, cmake, ROCm

.. _rocdensitymat-install:

Building and installing rocDENSITYMAT
======================================

Prerequisites
-------------

* ROCm 6.4 or later, including ``hip``, ``rocprim``.
* A C++17-capable compiler (``hipcc`` or ``amdclang++``).
* CMake 3.22+.

Configure and build
-------------------

.. code-block:: bash

   cd projects/rocdensitymat
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

Run the smoke samples
---------------------

.. code-block:: bash

   ./build/clients/staging/rocdensitymat-rabi-oscillator
   ./build/clients/staging/rocdensitymat-bell-state-evolution
   ./build/clients/staging/rocdensitymat-harmonic-oscillator
