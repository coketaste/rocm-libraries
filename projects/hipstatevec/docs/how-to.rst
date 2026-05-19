.. meta::
  :description: How to use hipSTATEVEC
  :keywords: hipSTATEVEC, GHZ, sampler, expectation

.. _hipstatevec-howto:

Using hipSTATEVEC
==================

Write to ``<hipstatevec.h>``:

.. code-block:: c++

   #include <hipstatevec.h>

   hipstatevecHandle_t h = nullptr;
   hipstatevecCreate(&h);

   // ... build a circuit using hipstatevecApplyMatrix /
   //   hipstatevecApplyPauliRotation / hipstatevecComputeExpectation ...

   hipstatevecDestroy(h);

The hipSTATEVEC API is intentionally a one-for-one rename of
cuStateVec's:

.. code-block:: bash

   sed -i 's/cu/hip/g' your_file.cpp

Anything that worked against cuStateVec compiles against hipSTATEVEC
with identical semantics. The library forwards every entry point to
rocSTATEVEC, which executes on AMD GPUs through the HIP runtime.
