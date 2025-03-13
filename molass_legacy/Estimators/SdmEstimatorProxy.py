"""
    Estimators.SdmEstimatorProxy.py

    temporary fix to make the get_colparam_bounds available

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
import numpy as np
from Estimators.SdmEstimator import SdmEstimator
from Optimizer.TheUtils import FILES

class SdmEstimatorProxy(SdmEstimator):
    def __init__(self, jobfolder):
        bounds_text_path = os.path.join(jobfolder, FILES[7])    # "bounds.txt"
        self.bounds = np.loadtxt(bounds_text_path)