.. meta::
  :description: hipDENSITYMAT public C API reference
  :keywords: hipDENSITYMAT, reference, API

.. _hipdensitymat-reference:

Public C API reference
======================

hipDENSITYMAT exposes the cuDensityMat 0.1.0 surface verbatim. See the
:doc:`rocDENSITYMAT reference <../../rocdensitymat/docs/reference>` for
behaviour and workspace semantics; the only difference at this layer is
the function-name shape and the ``hipdensitymat``-prefixed typedefs.

Convention
----------

- camelCase function names: ``hipdensitymatCreate``,
  ``hipdensitymatOperatorComputeAction``, etc.
- Status enum: ``hipdensitymatStatus_t``; numeric values are identical to
  ``cudensitymatStatus_t`` and ``rocdensitymat_status``.
- Handle typedefs: ``hipdensitymatHandle_t``,
  ``hipdensitymatWorkspaceDescriptor_t``, ``hipdensitymatState_t``,
  ``hipdensitymatElementaryOperator_t``, ``hipdensitymatOperatorTerm_t``,
  ``hipdensitymatOperator_t``,
  ``hipdensitymatMasterEquationSolver_t``.

Function index
--------------

See ``library/include/hipdensitymat.h`` and the ``internal/`` headers it
pulls in for the full function list. The function set is in 1:1
correspondence with cuDensityMat 0.1.0 and rocDENSITYMAT v0.1.0.
