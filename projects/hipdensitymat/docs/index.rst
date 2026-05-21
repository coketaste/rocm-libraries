.. meta::
  :description: hipDENSITYMAT: cuDensityMat-shaped C API for AMD ROCm
  :keywords: hipDENSITYMAT, ROCm, AMD, quantum, density matrix, cuDensityMat, cuQuantum

.. _hipdensitymat:

********************************************************************
hipDENSITYMAT documentation
********************************************************************

hipDENSITYMAT is a thin wrapper that exposes a ``cudensitymat``-shaped
camelCase C API on top of AMD's :doc:`rocDENSITYMAT
<../../rocdensitymat/docs/index>` library. It runs only on AMD ROCm /
HIP; it does not link to or require any NVIDIA runtime.

The porting contract is the same one followed by hipBLAS / hipSPARSE /
hipSTATEVEC: every public ``cudensitymat*`` entry point in the targeted
cuDensityMat release (cuQuantum 24.11 / cuDensityMat 0.1.0) has exactly
one ``hipdensitymat*`` counterpart with identical function signature,
identical numeric-value enums, identical workspace semantics, and
identical async-on-stream semantics — so consumer code written against
``<cudensitymat.h>`` ports to ``<hipdensitymat.h>`` with a mechanical
``s/cu/hip/`` rename.

.. note::

   hipDENSITYMAT v0.1.0 covers single-GPU parity. The Lindblad-collapse,
   eigenspectrum, backward-differentiation, MGMN-distributed, and MPS
   paths are exposed but return ``HIPDENSITYMAT_STATUS_NOT_SUPPORTED``;
   they will land in v0.2 in lock-step with the rocDENSITYMAT roadmap.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   how-to
   reference
