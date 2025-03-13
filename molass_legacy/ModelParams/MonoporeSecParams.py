"""
    MonoporeSecParams.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.BasicUtils import Struct
from molass_legacy._MOLASS.SerialSettings import get_setting
from SecTheory.RetensionTime import (make_initial_guess, compute_retention_time,
                                     estimate_conformance_params,
                                     estimate_conformance_params_fixed_poreexponent,
                                    )
from SecTheory.MonoPore import MEMP_LIMIT, N_LIMIT
from SecTheory.ColumnConstants import SECCONF_LOWER_BOUND, BAD_CONFORMANCE_REDUCE
from molass_legacy._MOLASS.SerialSettings import get_setting

NUM_SEC_PARAMS = 6
MEMP_LOWER = 1e-8
PENALTY_SCALE = 1e8

def initial_guess(xr_params):
    """
    note: this won't improve bounds for rp, m in the succession of "Known Best"
    """
    
    init_sec_params = get_setting("init_sec_params")
    return Struct(params=init_sec_params, bounds=None)

zeros5 = np.zeros(NUM_SEC_PARAMS - 1)

def sec_comformance_impl(t0_upper_bound, xr_params, rg_params, monopore_params, poresize_bounds=None):
    t0, rp, N, me, T, mp = monopore_params
    if poresize_bounds is None:
        poresize_penalty = 0
    else:
        poresize_penalty = min(0, rp - poresize_bounds[0])**2 + max(0, rp - poresize_bounds[1])**2
    conformance = max(0, t0 - t0_upper_bound)**2 + min(0, MEMP_LIMIT - (me+mp))**2 + PENALTY_SCALE*min(0, N_LIMIT - N)**2 + np.sum(np.min([zeros5, monopore_params[1:]], axis=0)**2)
    log_conformance = np.log10(max(MEMP_LOWER, conformance + PENALTY_SCALE*poresize_penalty))
    if log_conformance > 0:
        log_conformance *= BAD_CONFORMANCE_REDUCE
    return max(SECCONF_LOWER_BOUND, log_conformance)

class MonoporeSecParams:
    def __init__(self, poresize, poreexponent):
        self.poresize = poresize
        self.poreexponent = poreexponent

        if poreexponent is None or poreexponent == 0:   # poreexponent == 0 is used since set_setting("poreexponent", None) doesn't seem to work
            self.init_method = initial_guess
            self.conf_method = self.sec_comformance
            self.estm_method = None
            self.nump_adjust = 0
        else:
            assert False

        self.t0_upper_bound = get_setting("t0_upper_bound")

    def sec_comformance(self, xr_params, rg_params, monopore_params, poresize_bounds=None):
        return sec_comformance_impl(self.t0_upper_bound, xr_params, rg_params, monopore_params, poresize_bounds)
