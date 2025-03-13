"""
    circular_import_check.py
"""
import os
import sys
dir_ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(dir_)
sys.path.append(dir_)
from molass_legacy.Models.RateTheory.EDM import guess_multiple, edm_impl