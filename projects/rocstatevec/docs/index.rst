.. meta::
  :description: rocSTATEVEC: AMD-native HIP state-vector simulation
  :keywords: rocSTATEVEC, ROCm, quantum, statevector, cuStateVec, AMD

.. _rocstatevec:

********************************************************************
rocSTATEVEC documentation
********************************************************************

rocSTATEVEC is an AMD-native HIP implementation of single-node state-vector
quantum simulation. The public API is a clean-room implementation that mirrors
NVIDIA's `cuStateVec` (cuQuantum 24.11 / cuStateVec 1.7.x) — every public
``custatevec*`` entry point in that revision has exactly one
``rocstatevec_*`` counterpart with matching semantics. A portable
:doc:`hipSTATEVEC <../../hipstatevec/docs/index>` wrapper exposes the same
``custatevec``-shaped camelCase API and dispatches to either rocSTATEVEC on
AMD or cuStateVec on NVIDIA.

.. note::

   rocSTATEVEC v1.0 covers the **single-node** cuStateVec API surface in full.
   Multi-GPU multi-node (MGMN) entry points (``custatevecCommunicator*``,
   ``custatevecMultiDeviceSwapIndexBits``, ``custatevecDistIndexBitSwap*``,
   ``custatevecSVSwapWorker*``) are deferred to a later major version.

.. note::

   For portability across AMD and NVIDIA, prefer ``#include <hipstatevec.h>``
   over a direct dependency on rocSTATEVEC.

The rocSTATEVEC public repository is at
`<https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocstatevec>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   how-to
   reference

API reference
-------------

* :ref:`rocstatevec-handle`            — context lifetime
* :ref:`rocstatevec-init`              — state-vector initialization
* :ref:`rocstatevec-apply`             — gate / matrix application
* :ref:`rocstatevec-measure`           — measurement and collapse
* :ref:`rocstatevec-sample`            — sampler
* :ref:`rocstatevec-expect`            — expectation values
* :ref:`rocstatevec-accessor`          — sub-statevector slices
* :ref:`rocstatevec-permute`           — index-bit swaps
* :ref:`rocstatevec-batched`           — batched apply
* :ref:`rocstatevec-test`              — matrix property test

Provenance
----------

The rocSTATEVEC API is reverse-engineered exclusively from the public
cuStateVec API documentation. No NVIDIA header files or source code are
copied or redistributed; rocSTATEVEC is a clean-room implementation, MIT
licensed.
