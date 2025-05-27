"""
    Batch.DataBridge.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
from molass_legacy.Batch.FullBatch import FullBatch

class DataBridge(FullBatch):
    def __init__(self, sd, pre_recog, treat=None):
        self.logger = logging.getLogger(__name__)
        self.pre_recog = pre_recog
        FullBatch.__init__(self)
        if treat is None:
            from SecSaxs.DataTreatment import DataTreatment
            # task: inspect to make these consistent with OptStategyDialog
            correction = 1
            trimming = 2
            treat = DataTreatment(route="v2", trimming=trimming, correction=correction, unified_baseline_type=self.unified_baseline_type)

        # these duplicated procesures are currently required to get base_curve_info correctly
        # i.e., current benefits from using this class are to avoid duplication of loading and pre-recognition
        self.sd = trimmed_sd = treat.get_trimmed_sd(sd, pre_recog)
        self.corrected_sd = treat.get_corrected_sd(sd, pre_recog, trimmed_sd)

        self.base_curve_info = treat.get_base_curve_info()

    def prepare_for_lrf_source(self, num_components=None):
        self.exact_num_peaks = num_components
        uv_x, uv_y, xr_x, xr_y, baselines = self.get_curve_xy(return_baselines=True)
        uv_y_ = uv_y - baselines[0]
        xr_y_ = xr_y - baselines[1]
        self.get_modeled_peaks(uv_x, uv_y_, xr_x, xr_y_, num_peaks=num_components)
        self.set_lrf_src_args1(uv_x, uv_y, xr_x, xr_y, baselines)
