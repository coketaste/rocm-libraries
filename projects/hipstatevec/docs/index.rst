.. meta::
  :description: hipSTATEVEC: cuStateVec-shaped C API for AMD ROCm
  :keywords: hipSTATEVEC, ROCm, AMD, quantum, statevector, cuStateVec, cuQuantum

.. _hipstatevec:

********************************************************************
hipSTATEVEC documentation
********************************************************************

hipSTATEVEC is a thin wrapper that exposes a ``custatevec``-shaped
camelCase C API on top of AMD's :doc:`rocSTATEVEC
<../../rocstatevec/docs/index>` library. It runs only on AMD ROCm /
HIP; it does not link to or require any NVIDIA runtime.

The "porting contract" is the same one followed by hipBLAS / hipSPARSE /
hipFFT: every public ``custatevec*`` entry point in the targeted cuStateVec
release (cuQuantum 24.11 / cuStateVec 1.7.x) has exactly one
``hipstatevec*`` counterpart with identical function signature, identical
numerical-value enums, identical workspace semantics, and identical
async-on-stream semantics — so consumer code written against
``<custatevec.h>`` ports to ``<hipstatevec.h>`` with a mechanical
``s/cu/hip/`` rename and runs unchanged on AMD hardware.

.. note::

   hipSTATEVEC v0.1.0 covers single-node parity. MGMN/distributed APIs are
   excluded in this version (see
   :doc:`rocSTATEVEC docs <../../rocstatevec/docs/index>`).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   how-to
   reference
