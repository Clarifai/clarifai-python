# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# General information about the project.
project = 'Clarifai-Python'
copyright = '2023, Clarifai'
author = 'Clarifai'
release = '9.7.2'

version = '9.7.2'
# General configuration
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]

templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store', 'tests', 'scripts', 'clarifai.client.auth.*'
]
add_module_names = False

# Options for HTML output
html_theme = 'sphinx_rtd_theme'
