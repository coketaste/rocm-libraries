.. meta::
  :description: hipSTATEVEC: portable cuStateVec-shaped API for AMD and NVIDIA
  :keywords: hipSTATEVEC, ROCm, quantum, statevector, cuStateVec, cuQuantum

.. _hipstatevec:

********************************************************************
hipSTATEVEC documentation
********************************************************************

hipSTATEVEC is a portable wrapper that exposes a ``custatevec``-shaped
camelCase C API. It dispatches to either AMD's :doc:`rocSTATEVEC
<../../rocstatevec/docs/index>` library on a HIP runtime, or to NVIDIA's
``cuStateVec`` (cuQuantum 24.11 / 1.7.x) on a CUDA runtime.

The "porting contract" is the same one followed by hipBLAS / hipSPARSE /
hipFFT: every public ``custatevec*`` entry point in the targeted cuStateVec
release has exactly one ``hipstatevec*`` counterpart with identical
function signature, identical numerical-value enums, identical workspace
semantics, and identical async-on-stream semantics — so consumer code
written against ``<custatevec.h>`` ports to ``<hipstatevec.h>`` with a
mechanical ``s/cu/hip/`` rename.

.. note::

   hipSTATEVEC v1.0 covers single-node parity. MGMN/distributed APIs are
   excluded in this version (see
   :doc:`rocSTATEVEC docs <../../rocstatevec/docs/index>`).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   how-to
   reference
