.. meta::
  :description: How to use rocDENSITYMAT
  :keywords: rocDENSITYMAT, master equation, Liouvillian, RK4

.. _rocdensitymat-howto:

Using rocDENSITYMAT
====================

Minimum viable example: drive a single qubit on resonance and integrate
its master equation through one Rabi cycle with the fixed-step RK4
solver.

.. code-block:: c++

   #include <rocdensitymat.h>
   #include <hip/hip_runtime.h>
   #include <complex>
   #include <vector>

   using cd = std::complex<double>;
   constexpr int64_t hilbert_dim = 2;

   rocdensitymat_handle handle = nullptr;
   rocdensitymat_create(&handle);

   // 1) Build the Hamiltonian H = (Omega/2) sigma_x.
   const cd half_omega = {0.5, 0.0};
   rocdensitymat_elementary_operator_handle sigma_x = nullptr;
   rocdensitymat_create_elementary_operator(
       handle, 1, &hilbert_dim,
       ROCDENSITYMAT_ELEMENTARY_PAULI_X,
       ROCDENSITYMAT_C_64F, nullptr, nullptr, &sigma_x);

   rocdensitymat_operator_term_handle term = nullptr;
   rocdensitymat_create_operator_term(handle, 1, &hilbert_dim, &term);
   const int32_t state_modes[1] = {0};
   const int32_t mode_action_duality[1] = {0};
   rocdensitymat_operator_term_append_elementary_product(
       handle, term, 1, &sigma_x, state_modes, mode_action_duality,
       half_omega, nullptr);

   rocdensitymat_operator_handle op = nullptr;
   rocdensitymat_create_operator(handle, 1, &hilbert_dim, &op);
   rocdensitymat_operator_append_term(
       handle, op, term, 0, {1.0, 0.0}, nullptr);

   // 2) Run the RK4 master-equation stepper across [0, t_final].
   rocdensitymat_master_equation_solver_handle solver = nullptr;
   rocdensitymat_create_master_equation_solver(
       handle, op, ROCDENSITYMAT_SOLVER_RK4, &solver);

   // ... allocate state, attach buffer, then ...
   rocdensitymat_master_equation_step_n(
       handle, solver, /*y*/ state, /*t0*/ 0.0,
       /*dt*/ 1e-3, /*nSteps*/ 1000, /*params*/ nullptr,
       /*workspace*/ ws);

   rocdensitymat_destroy_master_equation_solver(solver);
   rocdensitymat_destroy_operator(op);
   rocdensitymat_destroy_operator_term(term);
   rocdensitymat_destroy_elementary_operator(sigma_x);
   rocdensitymat_destroy(handle);

API by category
---------------

Library context
~~~~~~~~~~~~~~~

``rocdensitymat_create`` / ``rocdensitymat_destroy`` /
``rocdensitymat_reset_random_seed`` plus version, property, logger, and
device-memory-handler queries.

States
~~~~~~

``rocdensitymat_create_state`` produces ``PURITY_PURE`` and
``PURITY_MIXED`` states. ``PURITY_MPS`` is reserved for v0.2.

Operators
~~~~~~~~~

Build elementary operators (Identity / Pauli / dense / diagonal),
compose them into ``OperatorTerm`` products, then sum into an
``Operator`` with per-term coefficients. Time-dependent coefficients
are supplied through host callbacks.

Compute
~~~~~~~

``rocdensitymat_operator_prepare_action`` queries the workspace size,
``rocdensitymat_operator_compute_action`` evaluates ``-i H |psi>``
(pure) or ``-i [H, rho]`` (mixed). ``rocdensitymat_operator_compute_expectation``
returns ``<O>``.
