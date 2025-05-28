"""
    Estimators.BaseEstimator.py

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
"""
import logging
from molass_legacy._MOLASS.SerialSettings import set_setting
from molass_legacy.SecTheory.T0UpperBound import estimate_t0upper_bound

class BaseEstimator:
    def __init__(self, editor):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.editor = editor
        self.sd = editor.sd
        self.corrected_sd = editor.corrected_sd
        self.nc = editor.get_n_components()     # this includes baseline
        self.ecurves = editor.ecurves
        self.t0_upper_bound = None

    def get_t0_upper_bound(self):
        if self.t0_upper_bound is None:
            ecurve = self.sd.get_xray_curve()
            t0_upper_bound = estimate_t0upper_bound(ecurve)
            self.logger.info("t0_upper_bound has been estimated to be %g", t0_upper_bound)
            set_setting("t0_upper_bound", t0_upper_bound)
            self.t0_upper_bound = t0_upper_bound
        return self.t0_upper_bound
