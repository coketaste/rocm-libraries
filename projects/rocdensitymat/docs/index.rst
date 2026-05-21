.. meta::
  :description: rocDENSITYMAT: AMD-native HIP density-matrix simulation
  :keywords: rocDENSITYMAT, ROCm, quantum, density matrix, cuDensityMat, AMD

.. _rocdensitymat:

********************************************************************
rocDENSITYMAT documentation
********************************************************************

rocDENSITYMAT is an AMD-native HIP implementation of single-GPU
analog quantum-dynamics simulation through dense state-vector and
density-matrix representations. v0.1.0 is a clean-room implementation
that mirrors the public C API of NVIDIA's ``cuDensityMat`` library
(cuQuantum 24.11 / cuDensityMat 0.1.0): every public ``cudensitymat*``
entry point in that revision has exactly one ``rocdensitymat_*``
counterpart with matching semantics. A portable
:doc:`hipDENSITYMAT <../../hipdensitymat/docs/index>` wrapper exposes
the same ``cudensitymat``-shaped camelCase API and dispatches into the
AMD-native rocDENSITYMAT core via a thin ``reinterpret_cast`` forwarder.

.. note::

   rocDENSITYMAT v0.1.0 covers the **single-GPU** cuDensityMat v0.1.0
   API surface for unitary dynamics. Multi-GPU multi-node distribution,
   Lindblad collapse channels, MPS states, eigenspectrum, and backward
   differentiation are wired but return ``NOT_SUPPORTED``; full support
   tracks the v0.2 milestone (cuDensityMat v0.3.x parity).

.. note::

   For portability across AMD and NVIDIA, prefer
   ``#include <hipdensitymat.h>`` over a direct dependency on
   rocDENSITYMAT.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   how-to
   reference

API reference
-------------

* :ref:`rocdensitymat-handle`            — context lifetime
* :ref:`rocdensitymat-workspace`         — workspace descriptor
* :ref:`rocdensitymat-state`             — pure / mixed states
* :ref:`rocdensitymat-operator`          — operator algebra
* :ref:`rocdensitymat-action`            — operator action / expectation
* :ref:`rocdensitymat-ode`               — master-equation stepping
* :ref:`rocdensitymat-properties`        — norm / trace / overlap

Provenance
----------

The rocDENSITYMAT API is reverse-engineered exclusively from the public
cuDensityMat API documentation. No NVIDIA header files or source code are
copied or redistributed; rocDENSITYMAT is a clean-room implementation,
MIT licensed.
