"""
    Batch.FullBatch.py

    eventually to be the baseclass of V2 procedures

    Copyright (c) 2023-2025, SAXS Team, KEK-PF
"""
import re
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting, clear_temporary_settings
from molass_legacy.Peaks.PeakParamsSet import PeakParamsSet

class FullBatch:
    def __init__(self):
        # to be moved from molass_legacy.Peaks.PeakEditor
        self.unified_baseline_type = get_setting("unified_baseline_type")
        self.elution_model = get_setting("elution_model")
        self.ecurve_info = None

    def load_data(self, in_folder):
        from molass_legacy.Batch.StandardProcedure import StandardProcedure
        self.logger.info("loading data from %s", in_folder)
        sp = StandardProcedure()
        sd = sp.load_old_way(in_folder)
        return sd

    def prepare(self, sd, num_peaks=None, min_num_peaks=None, debug=False):
        from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
        from molass_legacy.Optimizer.DefaultNumPeaks import get_default_num_peaks
        from SecSaxs.DataTreatment import DataTreatment
        self.pre_recog = PreliminaryRecognition(sd)
        if num_peaks is None:
            num_peaks = get_default_num_peaks(sd)
            if min_num_peaks is not None:
                num_peaks = max(min_num_peaks, num_peaks)

        self.logger.info("preparing as num_peaks=%d", num_peaks)

        treat = DataTreatment(route="v2", correction=1)
        trimmed_sd = treat.get_trimmed_sd(sd, self.pre_recog, debug=debug)

        # set PeakEditor attributes
        self.sd = trimmed_sd
        self.corrected_sd = treat.get_corrected_sd(sd, self.pre_recog, trimmed_sd)
        self.base_curve_info = treat.get_base_curve_info()
        self.exact_num_peaks = num_peaks
        self.ecurve_info = None
        self.fullopt_class = None

        """
        task: remove this (may be irrelevant)
        uv_x, uv_y, xr_x, xr_y, baselines = self.get_curve_xy(return_baselines=True)
        uv_peaks, xr_peaks = self.get_modeled_peaks(uv_x, uv_y, xr_x, xr_y)
        """

    def get_pre_recog_mapping_params(self):
        cs = self.pre_recog.cs
        return cs.slope, cs.intercept

    def get_curve_xy(self, return_baselines=False, debug=False):
        if debug:
            import QuickAnalysis.ModeledPeaks
            from importlib import reload
            reload(QuickAnalysis.ModeledPeaks)
        from molass_legacy.QuickAnalysis.ModeledPeaks import get_curve_xy_impl

        if self.ecurve_info is None:
            self.ecurve_info = get_curve_xy_impl(self.sd, baseline_type=self.unified_baseline_type, return_details=True, debug=debug)
            details = self.ecurve_info[-1]
            self.ecurves = details.uv_curve, details.xr_curve
            self.baseline_objects = details.baseline_objects
            self.baselines = details.baselines
            # self.baselines[0] = self.get_uv_baseline()    # get_uv_baseline_deprecated
            self.baseline_params = details.baseline_params
            self.logger.info("got xr baseline params as %s with unified_baseline_type=%d", str(self.baseline_params[1]), self.unified_baseline_type)

        if return_baselines:
            return *self.ecurve_info[0:4], self.baselines
        else:
            return self.ecurve_info[0:4]

    def get_uv_baseline(self, xy=None):
        # task: verify that this is correct
        if xy is None:
            return self.baselines[0]
        else:
            raise NotImplementedError("get_uv_baseline with xy argument is not implemented yet. Use get_uv_baseline_deprecated instead.")

    def get_uv_baseline_deprecated(self, xy=None):
        if xy is None:
            uv_x, uv_y = self.ecurve_info[0:2]
        else:
            uv_x, uv_y = xy
        uv_base_curve, uv_baseparams = self.base_curve_info
        intreg_scale, y_ = uv_base_curve.guess_integral_scale(uv_x, uv_y, uv_baseparams[0:7])
        uv_baseparams[-1] = intreg_scale    # note that this updated params will be used later in self.get_uv_base_params()
        return uv_base_curve(uv_x, uv_baseparams, y_)

    def get_modeled_peaks(self, uv_x, uv_y, xr_x, xr_y, num_peaks=None, affine=True, min_area_prop=None, debug=False):
        if debug:
            import molass_legacy.QuickAnalysis.ModeledPeaks
            from importlib import reload
            reload(molass_legacy.QuickAnalysis.ModeledPeaks)
        from molass_legacy.QuickAnalysis.ModeledPeaks import get_modeled_peaks_impl
        if num_peaks is None:
            num_peaks = self.exact_num_peaks
        a, b = self.get_pre_recog_mapping_params()
        uv_peaks, xr_peaks = get_modeled_peaks_impl(a, b, uv_x, uv_y, xr_x, xr_y, num_peaks, exact_num_peaks=num_peaks,
                                                affine=affine, min_area_prop=min_area_prop, debug=debug)        
        self.peak_params_set = PeakParamsSet(uv_peaks, xr_peaks, a, b)     # for backward compatibility
        return uv_peaks, xr_peaks

    def get_peak_params_set(self):
        return self.peak_params_set

    def set_lrf_src_args1(self, uv_x, uv_y, xr_x, xr_y, baselines):
        self.lrf_src_args1 = (uv_x, uv_y, xr_x, xr_y, baselines)

    def get_lrf_source(self, in_folder=None, min_num_peaks=None, clear=True, use_mapping=False, devel=False):
        if in_folder is None:
            """
            if in_folder is None,
            it is currently assumed that the lrf_src_args1 and peak_params_set are set
            as in Peaks.PeakEditor or Batch.DataBridge
            """
        else:
            """
            it is currently assumed that this is called as a LiteBatch method
            """
            if clear:
                clear_temporary_settings()      # this is usually harmless, be careful though.
            sd = self.load_data(in_folder)
            self.prepare(sd, min_num_peaks=min_num_peaks)
            uv_x, uv_y, xr_x, xr_y, baselines = self.get_curve_xy(return_baselines=True)
            uv_y_ = uv_y - baselines[0]
            xr_y_ = xr_y - baselines[1]
            if use_mapping:
                # note that this is a method of the MappingBatch class
                self.modelled_peaks_method(uv_x, uv_y_, xr_x, xr_y_)
            else:
                uv_peaks, xr_peaks = self.get_modeled_peaks(uv_x, uv_y_, xr_x, xr_y_, affine=True, min_area_prop=None)               
            self.set_lrf_src_args1(uv_x, uv_y, xr_x, xr_y, baselines)  # task: do not require users to do this
            set_setting('in_folder', in_folder)

        if devel:
            from importlib import reload
            import molass_legacy.Selective.LrfSource
            reload(molass_legacy.Selective.LrfSource)
        from molass_legacy.Selective.LrfSource import LrfSource
        return LrfSource(self.sd, self.corrected_sd, self.lrf_src_args1, self.peak_params_set, pre_recog=self.pre_recog)

    def get_n_components(self):
        num_base_elements = 1
        return len(self.peak_params_set[1]) + num_base_elements

    def get_function_class(self):
        if self.fullopt_class is None:
            fullopt_key = self.key_list[self.func_info.default_index]
            fullopt_class = self.func_dict[fullopt_key]
            code_re = re.compile(r"^(\w+)\s*:")
            m = code_re.search(fullopt_key)
            if m:
                class_code = m.group(1)
            else:
                assert False
        else:
            fullopt_class = self.fullopt_class
            class_code = self.class_code
        return fullopt_class, class_code

    def get_model_name(self):
        from molass_legacy.Optimizer.OptimizerUtils import get_model_name
        fullopt_class, class_code = self.get_function_class()
        return get_model_name(class_code)

    def construct_optimizer(self, fullopt_class=None):
 
        if fullopt_class is None:
            fullopt_class, class_code = self.get_function_class()
        n_components = self.get_n_components()
        """
        if self.base_curve_info is None:
            uv_base_curve = None
        else:
            uv_base_curve = self.base_curve_info[0]
        """
        uv_base_curve = self.baseline_objects[0]
        xr_base_curve = self.baseline_objects[1]
        self.uv_base_curve = uv_base_curve
        
        self.optimizer = fullopt_class(
            self.dsets,
            n_components,
            uv_base_curve=uv_base_curve,
            xr_base_curve=xr_base_curve,
            qvector=self.sd.qvector,    # trimmed sd
            wvector=self.sd.lvector,
            )

        self.fullopt = self.optimizer   # for backward compatibility
        self.params_type = self.fullopt.params_type

    def get_optimizer(self):
        return self.optimizer

    def compute_init_params(self, debug=False):

        estimator = self.params_type.get_estimator(self)

        try:
            self.init_params = estimator.estimate_params(debug=debug)
        except:
            # for classes not supporing debug
            self.init_params = estimator.estimate_params()

        self.logger.info("len(init_params)=%d, init_params[-7:]=%s", len(self.init_params), str(self.init_params[-7:]))

        return self.init_params

    def get_mapped_params(self):
        uv_peaks, xr_peaks, a, b = self.peak_params_set
        num_peaks_mapping = len(xr_peaks)
        return a, b, num_peaks_mapping

    def get_uv_base_params(self, xyt=None, debug=False):
        if debug:
            import Baseline.UvBaseline
            from importlib import reload
            reload(Baseline.UvBaseline)
        from molass_legacy.Baseline.UvBaseline import get_uv_base_params_impl
        return get_uv_base_params_impl(self, xyt=xyt, debug=debug)

if __name__ == '__main__':
    pass
