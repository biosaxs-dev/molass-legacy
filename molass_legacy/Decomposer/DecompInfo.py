# coding: utf-8
"""
    DecompInfo.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import copy

class DecompInfo:
    def __init__(self, opt_recs, opt_recs_uv, fx):
        self.opt_recs = opt_recs
        self.opt_recs_uv = opt_recs_uv
        self.fx = fx

    def get_scaled_recs(self, conc_factor):
        ret_recs = []
        for rec_xr, rec_uv in zip(self.opt_recs, self.opt_recs_uv):
            rec = rec_xr.scale_to(rec_uv, conc_factor)
            ret_recs.append(rec)
        return ret_recs
