"""
    Models.Stochastic.SDM.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.Models.Tentative import Model
from molass_legacy.Models.Stochastic.DispersivePdf import dispersive_monopore_pdf
from molass_legacy.Models.ElutionModelUtils import x_from_height_ratio_impl

class SDM(Model):
    def __init__(self, **kwargs):
        super(SDM, self).__init__(dispersive_monopore_pdf, **kwargs)
        self.fx = None

    def get_name(self):
        return "SDM"

    def is_traditional(self):
        return False

    def eval(self, params=None, x=None):
        return self.func(x, *params)
    
    def x_from_height_ratio(self, ecurve, ratio, params):
        assert self.fx is not None
        return x_from_height_ratio_impl(dispersive_monopore_pdf, ecurve, ratio, *params, needs_ymax=True, full_params=True, fx=self.fx)
    
    def set_fx_for_height_ratio(self, fx):
        self.fx = fx

    def get_params_string(self, params):
        return 'scale=%g, N=%g, T=%g, t0=%g, me=%s, mp=%g, rp=%g, rg=%g' % tuple(params)