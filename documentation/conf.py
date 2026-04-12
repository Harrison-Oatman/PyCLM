import os
import sys

# Repo root (for finding pyclm as an installed package or editable install)
sys.path.insert(0, os.path.abspath(".."))
# src layout — lets the zoo extension import pyclm directly without a venv install
sys.path.insert(0, os.path.abspath("../src"))
# Extension directory
sys.path.insert(0, os.path.abspath("ext"))

project = "PyCLM"
copyright = "2026, Harrison-Oatman"
author = "Harrison-Oatman"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "zoo_gallery",
]

templates_path = ["_templates"]
# Exclude the gallery fragment so it is not processed as a standalone page;
# it is only included by method_zoo.md.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_zoo_gallery.md"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
