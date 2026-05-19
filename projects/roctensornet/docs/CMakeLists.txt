# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
#
# Optional Sphinx documentation build. Activated when -DBUILD_DOCS=ON
# is passed and Sphinx is available on the system.
# ########################################################################

find_program(SPHINX_BUILD sphinx-build)
if(NOT SPHINX_BUILD)
    message(WARNING "roctensornet docs: sphinx-build not found; skipping")
    return()
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/_toc.yml.in
    ${CMAKE_CURRENT_BINARY_DIR}/sphinx/_toc.yml @ONLY)

add_custom_target(roctensornet_docs
    COMMAND ${SPHINX_BUILD} -b html
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/html
    COMMENT "Building rocTENSORNET Sphinx HTML")
