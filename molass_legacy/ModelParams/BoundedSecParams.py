"""
    BoundedSecParams.py

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
"""
from .SimpleSecParams import initial_guess
from molass_legacy.SecTheory.RetensionTime import estimate_conformance_params
from molass_legacy._MOLASS.SerialSettings import get_setting

class BoundedSecParams:
    def __init__(self, *args):      # *args are not used
        self.init_method = initial_guess
        self.estm_method = estimate_conformance_params
        self.nump_adjust = 0
        self.t0_upper_bound = get_setting("t0_upper_bound")
