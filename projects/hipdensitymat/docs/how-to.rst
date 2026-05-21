.. meta::
  :description: Porting cuDensityMat code to hipDENSITYMAT
  :keywords: hipDENSITYMAT, cuDensityMat, porting

.. _hipdensitymat-how-to:

Porting from cuDensityMat
=========================

In source files written against cuDensityMat, replace every occurrence of
``custatevec`` / ``cudensitymat`` with ``hipstatevec`` / ``hipdensitymat``:

.. code-block:: bash

   sed -i 's/cudensitymat/hipdensitymat/g; s/custatevec/hipstatevec/g' *.cpp *.h

The result builds against ``<hipdensitymat.h>`` and runs on AMD GPUs via
the rocDENSITYMAT backend with no further code changes.

What is supported in v0.1.0
---------------------------

- Library context, workspace, state (pure / mixed), elementary operators,
  operator terms, operator, operator-action compute and expectation, and
  the fixed-step RK4 master-equation stepper.

What is exposed but currently returns ``HIPDENSITYMAT_STATUS_NOT_SUPPORTED``
---------------------------------------------------------------------------

- Lindblad-collapse compute paths on mixed states.
- Operator eigenspectrum (``hipdensitymatOperatorComputeEigenspectrum``).
- Operator-action backward differentiation.
- MGMN / multi-rank distributed configuration.
- ``HIPDENSITYMAT_STATE_PURITY_MPS`` states.

All of those will land in v0.2.
