import os
import sys
from unittest.mock import MagicMock

# Function to simulate the MPI module
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# List of modules to mock
MOCK_MODULES = ['mpi4py', 'mpi4py.MPI']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# Get the absolute path to the directory containing conf.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Add the project root to sys.path
sys.path.insert(0, project_root)

print(f"Current directory: {current_dir}")
print(f"Project root: {project_root}")

# -- Project information -----------------------------------------------------
project = 'ProGRESS'
copyright = '2024, Atri Bera'
author = 'Atri Bera'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

# Optional: Configure autodoc options
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'show-inheritance': True,
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']
