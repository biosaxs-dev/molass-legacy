"""
    BoundedSecParams.py

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
"""
import numpy as np
from .SimpleSecParams import initial_guess, sec_comformance
from molass_legacy.SecTheory.RetensionTime import estimate_conformance_params
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.SecTheory.MonoPore import MEMP_LIMIT

class BoundedSecParams:
    def __init__(self, *args):      # *args are not used
        self.init_method = initial_guess
        self.conf_method = self.bounded_sec_comformance
        self.estm_method = estimate_conformance_params
        self.nump_adjust = 0
        self.t0_upper_bound = get_setting("t0_upper_bound")

    def bounded_sec_comformance(self, xr_params, rg_params, seccol_params, poresize_bounds=None):
        conformance = sec_comformance(xr_params, rg_params, seccol_params, poresize_bounds=poresize_bounds)
        t0, K, rp, m = seccol_params
        penalty = max(0, t0 - self.t0_upper_bound)**2 + min(0, MEMP_LIMIT - m)**2 + min(0, m)**2
        return conformance + penalty
