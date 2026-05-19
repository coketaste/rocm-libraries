.. meta::
  :description: hipSTATEVEC build and install
  :keywords: hipSTATEVEC, install, build, cmake, ROCm

.. _hipstatevec-install:

Building and installing hipSTATEVEC
====================================

hipSTATEVEC is an AMD-ROCm-only library. Its only backend is rocSTATEVEC.

Standalone build
----------------

.. code-block:: bash

   cmake -S . -B build \
     -DBUILD_CLIENTS_TESTS=ON
   cmake --build build -j

Superbuild
----------

Under the rocm-libraries superbuild, hipSTATEVEC builds together with
rocSTATEVEC when statevec is opted-in:

.. code-block:: bash

   cmake -S rocm-libraries -B build \
     -DTHEROCK_ENABLE_STATEVEC=ON
   cmake --build build -j

Run the conformance tests
-------------------------

The ``clients/tests/`` source set runs against the AMD backend; results
must match the documented FP64/FP32 tolerances declared in the
rocSTATEVEC test fixtures.

.. code-block:: bash

   cd build && ctest --output-on-failure
