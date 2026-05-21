.. meta::
  :description: hipDENSITYMAT build and install
  :keywords: hipDENSITYMAT, install, build, cmake, ROCm

.. _hipdensitymat-install:

Building and installing hipDENSITYMAT
======================================

hipDENSITYMAT is an AMD-ROCm-only library. Its only backend is rocDENSITYMAT.

Standalone build
----------------

.. code-block:: bash

   cmake -S . -B build \
     -DBUILD_CLIENTS_TESTS=ON
   cmake --build build -j
   ctest --test-dir build

The build expects an installed rocDENSITYMAT (or, when built from the
ROCm-libraries superbuild, the in-tree ``roc::rocdensitymat`` target).
