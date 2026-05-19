.. meta::
  :description: hipSTATEVEC API reference
  :keywords: hipSTATEVEC, API, reference

.. _hipstatevec-reference:

hipSTATEVEC API reference
==========================

The hipSTATEVEC API is a one-for-one ``s/cu/hip/`` rename of the
cuStateVec public API. Numeric values for status codes, ``Pauli_t``,
``MatrixLayout_t``, ``MatrixType_t``, ``MatrixMapType_t``,
``CollapseOp_t``, ``SamplerOutput_t``, and ``StateVectorType_t`` are
identical to their cuStateVec equivalents.

Migration table (subset; the full bijection is generated at install time
from the public headers):

+----------------------------------------------+----------------------------------------------+
| cuStateVec                                   | hipSTATEVEC                                  |
+==============================================+==============================================+
| ``custatevecCreate``                         | ``hipstatevecCreate``                        |
+----------------------------------------------+----------------------------------------------+
| ``custatevecDestroy``                        | ``hipstatevecDestroy``                       |
+----------------------------------------------+----------------------------------------------+
| ``custatevecApplyMatrix``                    | ``hipstatevecApplyMatrix``                   |
+----------------------------------------------+----------------------------------------------+
| ``custatevecApplyPauliRotation``             | ``hipstatevecApplyPauliRotation``            |
+----------------------------------------------+----------------------------------------------+
| ``custatevecMeasureOnZBasis``                | ``hipstatevecMeasureOnZBasis``               |
+----------------------------------------------+----------------------------------------------+
| ``custatevecComputeExpectation``             | ``hipstatevecComputeExpectation``            |
+----------------------------------------------+----------------------------------------------+
| ``custatevecComputeExpectationsOnPauliBasis``| ``hipstatevecComputeExpectationsOnPauliBasis``|
+----------------------------------------------+----------------------------------------------+
| ``custatevecSamplerCreate``                  | ``hipstatevecSamplerCreate``                 |
+----------------------------------------------+----------------------------------------------+
| ``custatevecSamplerPreprocess``              | ``hipstatevecSamplerPreprocess``             |
+----------------------------------------------+----------------------------------------------+
| ``custatevecSamplerSample``                  | ``hipstatevecSamplerSample``                 |
+----------------------------------------------+----------------------------------------------+
| ``custatevecSamplerDestroy``                 | ``hipstatevecSamplerDestroy``                |
+----------------------------------------------+----------------------------------------------+
| ``custatevecGetErrorString``                 | ``hipstatevecGetErrorString``                |
+----------------------------------------------+----------------------------------------------+

For per-function documentation, see the Doxygen-extracted reference in
``library/include/hipstatevec.h`` and ``library/include/internal/``.
