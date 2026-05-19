API reference
=============

The public C API surface is defined by the umbrella header
``roctensornet.h``, which transitively pulls in
``roctensornet-types.h``, ``roctensornet-functions.h``, and the
``internal/`` per-module headers.

The naming mapping with cuTensorNet 2.x is mechanical:

* ``cutensornet*`` -> ``roctensornet_*`` (snake_case)
* ``cutensornet*Handle_t`` -> ``roctensornet_*_handle`` or opaque
  pointer typedef.
* ``CUTENSORNET_*`` enum tokens -> ``ROCTENSORNET_*``

Full module index:

* Handle / library: ``roctensornet-auxiliary.h``
* Network / tensor descriptor: ``internal/roctensornet-network.h``
* Contraction optimizer: ``internal/roctensornet-optimizer.h``
* Workspace: ``internal/roctensornet-workspace.h``
* Contraction plan / autotune / execute: ``internal/roctensornet-contraction.h``
* Tensor SVD / QR / gate split: ``internal/roctensornet-svd.h``
* Gradient: ``internal/roctensornet-gradient.h``
* Network operator: ``internal/roctensornet-network-operator.h``
* Network state: ``internal/roctensornet-state.h``
* Derived (marginal / sampler / expectation / accessor):
  ``internal/roctensornet-derived.h``
