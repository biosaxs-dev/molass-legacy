"""

    OurImporter.py

    Copyright (c) 2024, Masatsuyo Takahashi, KEK-PF

"""
import sys
import importlib.util

"""
    https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
    https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
"""
def import_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
