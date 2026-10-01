"""
    SimpleSecParams.py

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.BasicUtils import Struct
from molass_legacy.SecTheory.RetensionTime import (make_initial_guess,
                                     estimate_conformance_params,
                                     estimate_conformance_params_fixed_poreexponent,
                                    )

def initial_guess(xr_params):
    params, bounds = make_initial_guess(xr_params[:,1])
    """
    note: this won't improve bounds for rp, m in the succession of "Known Best"
    """
    return Struct(params=params, bounds=bounds)

class SimpleSecParams:
    def __init__(self, poresize, poreexponent):
        self.poresize = poresize
        self.poreexponent = poreexponent

        if poreexponent is None or poreexponent == 0:   # poreexponent == 0 is used since set_setting("poreexponent", None) doesn't seem to work
            self.init_method = initial_guess
            self.estm_method = estimate_conformance_params
            self.nump_adjust = 0
        else:
            self.init_method = self.initial_guess_fixed_poreexponent
            self.estm_method = self.estimate_conformance_params_fixed_poreexponent
            self.nump_adjust = -1

    def initial_guess_fixed_poreexponent(self, xr_params):
        guess = initial_guess(xr_params)
        return Struct(params=guess.params[0:3], bounds=guess.bounds[0:3])

    def estimate_conformance_params_fixed_poreexponent(self, rgs, trs):
        result = estimate_conformance_params_fixed_poreexponent(rgs, trs, self.poreexponent)
        result.x = np.concatenate([result.x, [self.poreexponent]])
        return result
