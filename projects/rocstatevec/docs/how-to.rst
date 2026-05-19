.. meta::
  :description: How to use rocSTATEVEC
  :keywords: rocSTATEVEC, GHZ, sampler, expectation

.. _rocstatevec-howto:

Using rocSTATEVEC
==================

Minimum viable example: prepare a 3-qubit GHZ state, sample 1024 shots,
and compute :math:`\langle ZZZ \rangle`.

.. code-block:: c++

   #include <rocstatevec.h>
   #include <hip/hip_runtime.h>
   #include <complex>
   #include <vector>

   using cd = std::complex<double>;
   constexpr uint32_t n = 3;
   constexpr size_t   N = size_t{1} << n;

   rocstatevec_handle h = nullptr;
   rocstatevec_create_handle(&h);

   cd* dsv = nullptr;
   hipMalloc(&dsv, N * sizeof(cd));
   rocstatevec_initialize_state_vector(
       h, dsv, ROCSTATEVEC_C_64F, n, ROCSTATEVEC_STATE_VECTOR_TYPE_GHZ);

   rocstatevec_pauli   zzz[3]      = {ROCSTATEVEC_PAULI_Z, ROCSTATEVEC_PAULI_Z, ROCSTATEVEC_PAULI_Z};
   int32_t             qubits[3]   = {0, 1, 2};
   const rocstatevec_pauli* paulis_arr[1] = {zzz};
   const int32_t*      qubits_arr[1] = {qubits};
   uint32_t            qcounts[1]  = {3};
   double              expvals[1]  = {0.0};
   rocstatevec_compute_expectations_on_pauli_basis(
       h, dsv, ROCSTATEVEC_C_64F, n, expvals, paulis_arr, 1,
       qubits_arr, qcounts);

   // expvals[0] == 1.0

   hipFree(dsv);
   rocstatevec_destroy_handle(h);

API by category
---------------

Initialization
~~~~~~~~~~~~~~

``rocstatevec_initialize_state_vector`` produces ``ZERO``, ``UNIFORM``, ``GHZ``
or ``W`` initial states.

Gate application
~~~~~~~~~~~~~~~~

``rocstatevec_apply_matrix`` (with ``..._get_workspace_size``) applies an
arbitrary row- or column-major dense matrix to a set of target qubits with
optional control qubits. ``rocstatevec_apply_pauli_rotation`` applies
:math:`\exp(-i \theta/2 \, P_0 \otimes P_1 \otimes \cdots)` directly without
materializing the matrix.

Measurement and sampling
~~~~~~~~~~~~~~~~~~~~~~~~

* ``rocstatevec_measure_on_z_basis`` — single Z-basis measurement with
  optional collapse.
* ``rocstatevec_batch_measure`` / ``..._with_offset`` — multi-qubit
  measurement with optional collapse.
* ``rocstatevec_collapse_on_z_basis`` / ``..._by_bit_string`` — explicit
  post-selection.
* ``rocstatevec_sampler_create`` / ``..._preprocess`` / ``..._sample``
  / ``..._destroy`` — preprocessed multi-shot sampler with cumulative-
  probability + binary-search.

Expectation values
~~~~~~~~~~~~~~~~~~

* ``rocstatevec_compute_expectation`` (with workspace query) — Hermitian
  matrix expectation values.
* ``rocstatevec_compute_expectations_on_pauli_basis`` — batched Pauli-string
  expectation values exploiting Pauli diagonality.

Accessor
~~~~~~~~

``rocstatevec_accessor_create`` / ``..._create_view`` / ``..._destroy``
expose a logical sub-statevector slice. ``..._get`` and ``..._set`` move
amplitudes between device and host.

Permutation
~~~~~~~~~~~

* ``rocstatevec_apply_generalized_permutation_matrix`` (with workspace) —
  permutation + diagonal scaling.
* ``rocstatevec_absorb_diagonal_matrix`` — apply a diagonal in place.
* ``rocstatevec_swap_index_bits`` — qubit relabeling.

Batched
~~~~~~~

``rocstatevec_apply_matrix_batched`` (with workspace) — apply one of N
matrices to each of K state vectors in a single launch context.

Validation
~~~~~~~~~~

``rocstatevec_test_matrix_type`` — compute a residual classifying a matrix
as ``GENERAL`` / ``UNITARY`` / ``HERMITIAN``.
