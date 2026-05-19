# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
#
# Resolve the runtime backends rocTENSORNET depends on. Hard dependencies
# are HIP and rocPRIM only; hipTENSOR and rocSOLVER slots in the handle
# are reserved for future revisions but not required to link or run
# v0.1.0 (contractions and SVD/QR are implemented with direct HIP
# kernels and host helpers).
#
# All find_package calls are skipped when the corresponding target is
# already present in the parent CMake project (i.e. the rocm-libraries
# superbuild).
# ########################################################################

if(NOT TARGET hip::host)
    find_package(hip QUIET CONFIG)
endif()

set(MIN_ROCPRIM_VERSION "4.0.0")
if(NOT TARGET roc::rocprim)
    find_package(rocprim ${MIN_ROCPRIM_VERSION} QUIET CONFIG)
endif()

# Optional: backends that future revisions can opt-into without an ABI
# break. We probe for them but do not fail if absent.
if(NOT TARGET hiptensor::hiptensor)
    find_package(hiptensor QUIET CONFIG)
endif()
if(NOT TARGET roc::rocsolver)
    find_package(rocsolver QUIET CONFIG)
endif()
