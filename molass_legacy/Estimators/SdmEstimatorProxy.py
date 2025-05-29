"""
    Estimators.SdmEstimatorProxy.py

    temporary fix to make the get_colparam_bounds available

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import os
import numpy as np
from .SdmEstimator import SdmEstimator
from molass_legacy.Optimizer.TheUtils import FILES

class SdmEstimatorProxy(SdmEstimator):
    def __init__(self, jobfolder):
        bounds_text_path = os.path.join(jobfolder, FILES[7])    # "bounds.txt"
        self.bounds = np.loadtxt(bounds_text_path)