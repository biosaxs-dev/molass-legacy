"""
    Selective.DataFilter.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from molass_legacy.Models.ElutionCurveModels import EGH, EGHA
from Selective.LrfSource import LrfSource
from molass_legacy.Models.ModelUtils import compute_cy_list, compute_area_props

class DataFilter(LrfSource):
    def __init__(self, lrf_src):
        self.logger = logging.getLogger(__name__)
        self.apply_filter(lrf_src)
    
    def apply_filter(self, lrf_src):
        self.corrected_sd = lrf_src.corrected_sd
        self.uv_x = lrf_src.uv_x
        self.uv_y = lrf_src.uv_y
        self.xr_x = lrf_src.xr_x
        self.xr_y = lrf_src.xr_y
        self.baselines = lrf_src.baselines
        uv_peaks = lrf_src.uv_peaks
        xr_peaks = lrf_src.xr_peaks
        if xr_peaks.shape[1] == 4:
            self.model = EGH()
        else:
            self.model = EGHA()
        indeces = self.make_filter_indeces(uv_peaks, xr_peaks)
        self.uv_peaks = uv_peaks[indeces]
        self.xr_peaks = xr_peaks[indeces]
        self.rg_info = None
        self.egh_moments_list = None       
        # self.draw()
        self._compute_rgs()

    def make_filter_indeces(self, uv_peaks, xr_peaks):
        # AhRR: remove 0 for the reason of minor peaks near a major peak top
        num_peaks = len(xr_peaks)
        cy_list = compute_cy_list(self.model, self.xr_x, xr_peaks)
        props = compute_area_props(cy_list)
        full_indeces = np.arange(num_peaks)
        remove_indeces = []
        topxv = xr_peaks[:,1]
        max_sigma = np.max(xr_peaks[:,2])
        print("props=", props, "topxv=", topxv, "max_sigma=", max_sigma)
        for k, p in enumerate(props):
            if p < 0.05:
                if k < num_peaks-1:
                    dx = topxv[k+1] - topxv[k]
                    if dx/max_sigma < 1:
                        remove_indeces.append(k)
                if k > 0:
                    dx = topxv[k] - topxv[k-1]
                    if dx/max_sigma < 1:
                        remove_indeces.append(k)
        if len(remove_indeces) > 0:
            indeces = np.setdiff1d(full_indeces, remove_indeces)
            self.logger.info("make_filter_indeces: filtering by %s as in AhRR with a centered minor peak", indeces)
        else:
            indeces = full_indeces
        return indeces

    def get_filtered_src(self):
        return self