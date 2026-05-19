.. meta::
  :description: How to use hipSTATEVEC
  :keywords: hipSTATEVEC, GHZ, sampler, expectation

.. _hipstatevec-howto:

Using hipSTATEVEC
==================

For functionally identical behavior on AMD or NVIDIA, write to
``<hipstatevec.h>``:

.. code-block:: c++

   #include <hipstatevec.h>

   hipstatevecHandle_t h = nullptr;
   hipstatevecCreate(&h);

   // ... build a circuit using hipstatevecApplyMatrix /
   //   hipstatevecApplyPauliRotation / hipstatevecComputeExpectation ...

   hipstatevecDestroy(h);

The hipSTATEVEC API is intentionally a one-for-one rename of cuStateVec's:

.. code-block:: bash

   sed -i 's/cu/hip/g' your_file.cpp

Anything that worked against cuStateVec compiles and runs against
hipSTATEVEC with identical semantics, with the AMD backend executing
on a HIP runtime via rocSTATEVEC, or the NVIDIA backend forwarding to
cuStateVec on CUDA.
