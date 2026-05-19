.. meta::
  :description: rocSTATEVEC API reference
  :keywords: rocSTATEVEC, API, reference

.. _rocstatevec-reference:

rocSTATEVEC API reference
==========================

The full Doxygen-extracted reference is grouped by category and lives next
to the public headers under ``library/include/`` and
``library/include/internal/``. The category anchor names below correspond
1-to-1 with the cuStateVec API documentation sections so the migration
table in the project ``README.md`` is easy to follow.

.. _rocstatevec-handle:

Library handle and runtime services
-----------------------------------

* ``rocstatevec_create_handle``
* ``rocstatevec_destroy_handle``
* ``rocstatevec_set_stream``
* ``rocstatevec_get_stream``
* ``rocstatevec_get_version``
* ``rocstatevec_get_property``
* ``rocstatevec_get_error_string``
* ``rocstatevec_get_error_name``
* ``rocstatevec_set_device_mem_handler``
* ``rocstatevec_get_device_mem_handler``
* ``rocstatevec_logger_set_callback`` / ``..._set_callback_data`` /
  ``..._set_file`` / ``..._open_file`` / ``..._set_level`` /
  ``..._set_mask`` / ``..._force_disable``

.. _rocstatevec-init:

State-vector initialization
---------------------------

* ``rocstatevec_initialize_state_vector``

.. _rocstatevec-apply:

Gate / matrix application
-------------------------

* ``rocstatevec_apply_matrix_get_workspace_size`` / ``rocstatevec_apply_matrix``
* ``rocstatevec_apply_pauli_rotation``
* ``rocstatevec_apply_generalized_permutation_matrix_get_workspace_size``
  / ``rocstatevec_apply_generalized_permutation_matrix``
* ``rocstatevec_absorb_diagonal_matrix``

.. _rocstatevec-measure:

Measurement and collapse
------------------------

* ``rocstatevec_abs2_sum_on_z_basis`` / ``rocstatevec_abs2_sum_array``
* ``rocstatevec_measure_on_z_basis``
* ``rocstatevec_batch_measure`` / ``rocstatevec_batch_measure_with_offset``
* ``rocstatevec_collapse_on_z_basis`` / ``rocstatevec_collapse_by_bit_string``

.. _rocstatevec-sample:

Sampler
-------

* ``rocstatevec_sampler_create``
* ``rocstatevec_sampler_destroy``
* ``rocstatevec_sampler_preprocess``
* ``rocstatevec_sampler_get_squared_norm``
* ``rocstatevec_sampler_apply_sub_sv_offset``
* ``rocstatevec_sampler_sample``

.. _rocstatevec-expect:

Expectation values
-------------------

* ``rocstatevec_compute_expectation_get_workspace_size``
  / ``rocstatevec_compute_expectation``
* ``rocstatevec_compute_expectations_on_pauli_basis``

.. _rocstatevec-accessor:

Accessor
--------

* ``rocstatevec_accessor_create`` / ``..._create_view`` / ``..._destroy``
* ``rocstatevec_accessor_set_extra_workspace``
* ``rocstatevec_accessor_get`` / ``rocstatevec_accessor_set``

.. _rocstatevec-permute:

Index-bit permutation
---------------------

* ``rocstatevec_swap_index_bits``

.. _rocstatevec-batched:

Batched apply
-------------

* ``rocstatevec_apply_matrix_batched_get_workspace_size``
  / ``rocstatevec_apply_matrix_batched``

.. _rocstatevec-test:

Matrix property test
--------------------

* ``rocstatevec_test_matrix_type_get_workspace_size``
  / ``rocstatevec_test_matrix_type``
