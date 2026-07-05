"""Pytest configuration file to set up sys.path for test resolution."""

import os
import sys

# Add the 'src' directory to the python path to allow tests to import modules directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
