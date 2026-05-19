# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

if(NOT TARGET hip::host)
    find_package(hip QUIET CONFIG)
endif()

# hipTENSORNET requires roctensornet at link time. Under the
# rocm-libraries superbuild the target is registered via add_subdirectory
# and is already visible; for a stand-alone build we look it up via the
# installed CMake config package.
if(NOT TARGET roc::roctensornet)
    find_package(roctensornet QUIET CONFIG)
endif()
