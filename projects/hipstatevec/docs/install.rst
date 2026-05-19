.. meta::
  :description: hipSTATEVEC build and install
  :keywords: hipSTATEVEC, install, build, cmake, ROCm, CUDA

.. _hipstatevec-install:

Building and installing hipSTATEVEC
====================================

Backend selection
-----------------

hipSTATEVEC requires exactly one backend to be enabled at configure time.

AMD (HIP, default):

.. code-block:: bash

   cmake -S . -B build \
     -DHIPSTATEVEC_ENABLE_HIP=ON \
     -DHIPSTATEVEC_ENABLE_CUDA=OFF \
     -DBUILD_CLIENTS_TESTS=ON
   cmake --build build -j

NVIDIA (CUDA, requires cuQuantum SDK on ``CMAKE_PREFIX_PATH``):

.. code-block:: bash

   cmake -S . -B build \
     -DHIPSTATEVEC_ENABLE_HIP=OFF \
     -DHIPSTATEVEC_ENABLE_CUDA=ON \
     -DCMAKE_PREFIX_PATH=/path/to/cuquantum \
     -DBUILD_CLIENTS_TESTS=ON
   cmake --build build -j

Run the conformance tests
-------------------------

The same ``clients/tests/`` source set runs against either backend; results
must match within the documented FP64/FP32 tolerances.

.. code-block:: bash

   cd build && ctest --output-on-failure
