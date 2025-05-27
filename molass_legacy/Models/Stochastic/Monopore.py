"""
    Models.Stochastic.Monopore.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.Models.Tentative import Model
from molass_legacy.SecTheory.BasicModels import robust_single_pore_pdf
from molass_legacy.Models.ElutionModelUtils import x_from_height_ratio_impl

def monopore_func(x, scale, N, T, t0, me, mp, poresize, rg):
    rho = min(1, rg/poresize)
    ni_ = N * (1 - rho)**me
    ti_ = T * (1 - rho)**mp
    return scale * robust_single_pore_pdf(x - t0, ni_, ti_)

class Monopore(Model):
    def __init__(self, **kwargs):
        super(Monopore, self).__init__(monopore_func, **kwargs)
        self.fx = None

    def get_name(self):
        return "STC"

    def is_traditional(self):
        return False

    def eval(self, params=None, x=None):
        return self.func(x, *params)
    
    def x_from_height_ratio(self, ecurve, ratio, params):
        assert self.fx is not None
        return x_from_height_ratio_impl(monopore_func, ecurve, ratio, *params, needs_ymax=True, full_params=True, fx=self.fx)
    
    def set_fx_for_height_ratio(self, fx):
        self.fx = fx

    def get_params_string(self, params):
        return 'scale=%g, N=%g, T=%g, t0=%g, me=%s, mp=%g, rp=%g, rg=%g' % tuple(params)