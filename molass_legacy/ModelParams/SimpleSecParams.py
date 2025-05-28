"""
    SimpleSecParams.py

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.BasicUtils import Struct
from molass_legacy.SecTheory.RetensionTime import (make_initial_guess, compute_retention_time,
                                     estimate_conformance_params,
                                     estimate_conformance_params_fixed_poreexponent,
                                    )
from molass_legacy.SecTheory.ColumnConstants import SECCONF_LOWER_BOUND, BAD_CONFORMANCE_REDUCE

SEC_PENALTY_SCALE = 1e8
MIN_RETENTION_TIME_RATIO = 0.2

def initial_guess(xr_params):
    params, bounds = make_initial_guess(xr_params[:,1])
    """
    note: this won't improve bounds for rp, m in the succession of "Known Best"
    """
    return Struct(params=params, bounds=bounds)

def sec_comformance(xr_params, rg_params, seccol_params, poresize_bounds=None):
    # model_trs = compute_retention_time(seccol_params, rg_params)
    # task: unify compute_retention_time and the code below

    Npc, rp, tI, t0, P, m = seccol_params
    rhos = rg_params/rp
    rhos[rhos > 1] = 1
    model_trs = tI + P*(1 - rhos)**m

    if poresize_bounds is None:
        poresize_penalty = 0
    else:
        poresize_penalty = min(0, rp - poresize_bounds[0])**2 + max(0, rp - poresize_bounds[1])**2

    min_t = P*MIN_RETENTION_TIME_RATIO
    other_penalty = min(0, t0 - tI - min_t)**2 + min(0, m - 1)**2 + max(0, m - 3)**2
    log_conformance = np.log10(np.average((model_trs - xr_params[:,1])**2) + SEC_PENALTY_SCALE*(poresize_penalty + other_penalty))
    if log_conformance > 0:
        log_conformance *= BAD_CONFORMANCE_REDUCE   # large conformance at early stages can be misleading

    return max(SECCONF_LOWER_BOUND, log_conformance)

class SimpleSecParams:
    def __init__(self, poresize, poreexponent):
        self.poresize = poresize
        self.poreexponent = poreexponent

        if poreexponent is None or poreexponent == 0:   # poreexponent == 0 is used since set_setting("poreexponent", None) doesn't seem to work
            self.init_method = initial_guess
            self.conf_method = sec_comformance
            self.estm_method = estimate_conformance_params
            self.nump_adjust = 0
        else:
            self.init_method = self.initial_guess_fixed_poreexponent
            self.conf_method = self.sec_comformance_fixed_poreexponent
            self.estm_method = self.estimate_conformance_params_fixed_poreexponent
            self.nump_adjust = -1

    def initial_guess_fixed_poreexponent(self, xr_params):
        guess = initial_guess(xr_params)
        return Struct(params=guess.params[0:3], bounds=guess.bounds[0:3])

    def sec_comformance_fixed_poreexponent(self, xr_params, rg_params, seccol_params):
        return sec_comformance(xr_params, rg_params, np.concatenate([seccol_params, [self.poreexponent]]))

    def estimate_conformance_params_fixed_poreexponent(self, rgs, trs):
        result = estimate_conformance_params_fixed_poreexponent(rgs, trs, self.poreexponent)
        result.x = np.concatenate([result.x, [self.poreexponent]])
        return result
