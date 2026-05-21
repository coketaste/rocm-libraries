.. meta::
  :description: rocDENSITYMAT API reference
  :keywords: rocDENSITYMAT, API, reference

.. _rocdensitymat-reference:

rocDENSITYMAT API reference
============================

The full reference is grouped by category and lives next to the public
headers under ``library/include/`` and ``library/include/internal/``.
The category anchor names below correspond 1-to-1 with the cuDensityMat
v0.1.0 API documentation sections.

.. _rocdensitymat-handle:

Library handle and runtime services
-----------------------------------

* ``rocdensitymat_create``
* ``rocdensitymat_destroy``
* ``rocdensitymat_reset_random_seed``
* ``rocdensitymat_get_version``
* ``rocdensitymat_get_property``
* ``rocdensitymat_get_error_string``
* ``rocdensitymat_get_error_name``
* ``rocdensitymat_set_device_mem_handler`` /
  ``rocdensitymat_get_device_mem_handler``
* ``rocdensitymat_logger_set_callback`` / ``..._set_file`` /
  ``..._set_level`` / ``..._set_mask`` / ``..._force_disable``

.. _rocdensitymat-workspace:

Workspace
---------

* ``rocdensitymat_create_workspace``
* ``rocdensitymat_destroy_workspace``
* ``rocdensitymat_workspace_get_memory_size``
* ``rocdensitymat_workspace_set_memory``
* ``rocdensitymat_workspace_get_memory``

.. _rocdensitymat-state:

State
-----

* ``rocdensitymat_create_state`` / ``rocdensitymat_destroy_state``
* ``rocdensitymat_state_get_num_components`` /
  ``rocdensitymat_state_get_component_info`` /
  ``rocdensitymat_state_attach_component_buffer``
* ``rocdensitymat_state_initialize_zero`` /
  ``rocdensitymat_state_initialize_uniform`` /
  ``rocdensitymat_state_initialize_basis``
* ``rocdensitymat_state_compute_norm`` /
  ``rocdensitymat_state_compute_trace`` /
  ``rocdensitymat_state_compute_overlap``

.. _rocdensitymat-operator:

Operator algebra
----------------

* ``rocdensitymat_create_elementary_operator`` /
  ``rocdensitymat_destroy_elementary_operator``
* ``rocdensitymat_create_operator_term`` /
  ``rocdensitymat_operator_term_append_elementary_product`` /
  ``rocdensitymat_destroy_operator_term``
* ``rocdensitymat_create_operator`` /
  ``rocdensitymat_operator_append_term`` /
  ``rocdensitymat_destroy_operator``

.. _rocdensitymat-action:

Operator action and expectation
-------------------------------

* ``rocdensitymat_operator_prepare_action``
* ``rocdensitymat_operator_compute_action``
* ``rocdensitymat_operator_compute_expectation``

.. _rocdensitymat-ode:

Master-equation solver
----------------------

* ``rocdensitymat_create_master_equation_solver`` /
  ``rocdensitymat_destroy_master_equation_solver``
* ``rocdensitymat_master_equation_step``
* ``rocdensitymat_master_equation_step_n``

.. _rocdensitymat-properties:

State properties
----------------

* ``rocdensitymat_state_compute_norm``
* ``rocdensitymat_state_compute_trace``
* ``rocdensitymat_state_compute_overlap``

Wired but not yet supported (returns
``ROCDENSITYMAT_STATUS_NOT_SUPPORTED`` in v0.1.0)
--------------------------------------------------

* ``rocdensitymat_reset_distributed_configuration`` /
  ``rocdensitymat_get_num_ranks`` / ``rocdensitymat_get_proc_rank``
* ``rocdensitymat_operator_compute_eigenspectrum`` (and prepare)
* ``rocdensitymat_operator_compute_action_backward_diff`` (and prepare)
* ``rocdensitymat_create_state`` with ``ROCDENSITYMAT_STATE_PURITY_MPS``
* Lindblad-collapse compute path inside
  ``rocdensitymat_operator_compute_action``

These entry points are kept ABI-stable so v0.1.0 callers do not break
when the v0.2 implementation lands.
