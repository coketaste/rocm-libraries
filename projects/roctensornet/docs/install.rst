Installation
============

Build stand-alone
-----------------

.. code-block:: bash

   cmake -S projects/roctensornet -B build/roctensornet \
         -DBUILD_CLIENTS_TESTS=ON
   cmake --build build/roctensornet -j
   ctest --test-dir build/roctensornet/clients/tests --output-on-failure

Build via the rocm-libraries superbuild
---------------------------------------

.. code-block:: bash

   cmake -S . -B build -DTHEROCK_ENABLE_TENSORNET=ON
   cmake --build build -j

Dependencies
------------

* HIP runtime (any ROCm-supported version)
* rocPRIM 4.0.0 or newer
* (optional, reserved for future revisions) hipTENSOR and rocSOLVER
