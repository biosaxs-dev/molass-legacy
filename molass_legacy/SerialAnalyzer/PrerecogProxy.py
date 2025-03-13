# coding: utf-8
"""
    PrerecogProxy.py

    Copyright (c) 2020-2021, SAXS Team, KEK-PF
"""
import logging
from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition

class FlowChangeProxy:
    def __init__(self, uv):
        j_slice = uv.j_slice
        jmin = 0 if j_slice.start is None  else  j_slice.start
        jmax = uv.data.shape[1]-1 if j_slice.stop is None  else  j_slice.stop-1
        self.j_points = (jmin, jmax)

    def get_flow_changes(self):
        return self.j_points

class PrerecogProxy(PreliminaryRecognition):
    def __init__(self, md):
        self.logger = logging.getLogger(__name__)
        self.md = md
        self.cs = md.cs
        self.flowchange = FlowChangeProxy(md.uv)
        self.mapped_info = self.cs.mapped_info
        self.pre_rg = None

    def get_xray_shift(self):
        """
        overriding PreliminaryRecognition.get_xray_shift
        so that get_setting('xr_restrict_list') not be required.
        """
        xr = self.md.xr
        shift = xr.j_slice.start
        if shift is None:
            shift = 0
        return shift

    def get_rg(self):
        if self.pre_rg is None:
            self.sd = self.md.get_sd()
            self.get_default_angle_range(self.sd)
        return self.pre_rg.sg.Rg
