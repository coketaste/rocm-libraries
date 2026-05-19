# ########################################################################
# Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################
"""Sphinx configuration for the rocSTATEVEC documentation site."""

project   = "rocSTATEVEC"
author    = "Advanced Micro Devices, Inc."
copyright = "2024-2026, Advanced Micro Devices, Inc."

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

templates_path  = ["_templates"]
exclude_patterns = ["_build", ".DS_Store"]

html_theme            = "rocm_docs_theme"
html_theme_options    = {"flavor": "rocm"}
html_static_path      = ["_static"]
external_toc_path     = "sphinx/_toc.yml"
external_projects_current_project = "rocstatevec"
