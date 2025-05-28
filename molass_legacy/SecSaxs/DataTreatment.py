"""
    SecSaxs.DataTreatment.py

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
"""
import os
import logging
from collections import OrderedDict
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import set_setting
from molass_legacy.Mapping.MappingV2Utils import make_mapped_info_for_v2
from .DataSet import copy_create_dataset_from_sd
class DataTreatment:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        self._dict = OrderedDict()
        self.path = None
        self.base_curve_info = None
        self.corrected_sd = None
        if len(kwargs) > 0:
            self._dict["route"] = kwargs.pop("route", None)
            self._dict["trimming"] =  kwargs.pop("trimming", 1)
            self._dict["correction"] = kwargs.pop("correction", 0)
            self._dict["unified_baseline_type"] = kwargs.pop("unified_baseline_type", 1)
            assert len(kwargs) == 0

    def save(self, path=None):      
        if path is None:
            from molass_legacy.Optimizer.TheUtils import get_treatment_path
            path = get_treatment_path()
        with open(path, "w") as fh:
            fh.write(self.__repr__())

    def load(self, path=None, optimizer_folder=None):
        from molass_legacy.KekLib.EvalUtils import eval_file
        if path is None:
            from molass_legacy.Optimizer.TheUtils import get_treatment_path
            path = get_treatment_path(optimizer_folder=optimizer_folder)
        self.path = path

        def replacer(code):
            return code.replace("DataTreatment", "OrderedDict")

        self._dict = eval_file(path,
                               locals_=globals(),   # for OrderedDict
                               replacer=replacer)

        unified_baseline_type = self.get("unified_baseline_type")
        set_setting("unified_baseline_type", unified_baseline_type)
        self.logger.info("treatment loaded with unified_baseline_type=%d", unified_baseline_type)

    def keys(self):
        return ["route", "trimming", "correction", "unified_baseline_type"]

    def get(self, key):
        return self._dict.get(key)

    def __repr__(self):
        return str(self._dict).replace("OrderedDict", "DataTreatment")

    def get_trimmed_sd(self, sd, pre_recog, debug=False):
        if debug:
            import Trimming.OptimalTrimming
            from importlib import reload
            reload(Trimming.OptimalTrimming)
        from molass_legacy.Trimming.OptimalTrimming import OptimalTrimming

        mt = OptimalTrimming(sd, pre_recog, debug=debug)
        base_curve_info = mt.get_base_curve_info()

        trimming = self._dict["trimming"]

        if trimming == 1 or True:
            sd_, new_prerecog = sd._get_analysis_copy_impl(pre_recog, return_also_new_prerecog=True, debug=True)
            trimmed = "trimmed"
            # no mt.set_info() to keep the trimming info

        elif trimming == 2:
            trimmed = "minimally-trimmed"
            mt.set_info()
            sd_ = sd._get_analysis_copy_impl(pre_recog)
        else:
            sd_ = sd.get_copy()
            trimmed = "non-trimmed"
            base_curve_info = None

        sd_copy = copy_create_dataset_from_sd(sd, sd_)
    
        if False:
            xr_curve0 = sd.get_xray_curve()
            xr_curve1 = sd_copy.get_xr_curve()
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("get_trimmed_sd")
                ax1.plot(xr_curve0.x, xr_curve0.y)
                ax2.plot(xr_curve1.x, xr_curve1.y)
                fig.tight_layout()
                plt.show()

        self.base_curve_info = base_curve_info
        self.logger.info("got %s data by treatment from %s", trimmed, self.path)

        self.sd_orig = sd
        self.pre_recog = new_prerecog
        self.trimmed_sd = sd_copy

        return sd_copy

    def get_corrected_sd(self, sd, pre_recog, trimmed_sd, debug=False):
        if self._dict["correction"]:
            from molass_legacy.Baseline.BaselineUtils import get_corrected_sd_impl
            x_curve = trimmed_sd.get_xray_curve()
            mapped_info = make_mapped_info_for_v2(x_curve)
            sd_copy = trimmed_sd.copy()
            sd_copy = get_corrected_sd_impl(sd_copy, sd, pre_recog, mapped_info=mapped_info)
            corrected = "corrected"
            self.corrected_sd = sd_copy
        else:
            sd_copy = trimmed_sd
            corrected = "non-corrected"

        self.logger.info("got %s data by treatment", corrected)
        return sd_copy

    def get_treated_sd(self, sd, pre_recog, debug=False):
        trimmed_sd = self.get_trimmed_sd(sd, pre_recog, debug=debug)
        corrected_sd = self.get_corrected_sd(sd, pre_recog, trimmed_sd)
        return corrected_sd

    def get_base_curve_info(self):
        return self.base_curve_info

    def get_base_curve(self):
        if self.base_curve_info is None:
            uv_base_curve = None
        else:
            uv_base_curve = self.base_curve_info[0]
        return uv_base_curve

def demo(in_folder, sd):
    import molass_legacy.KekLib.DebugPlot as plt
    from molass_legacy.DataStructure.MatrixData import simple_plot_3d
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition

    print(in_folder)
    pre_recog = PreliminaryRecognition(sd)
    treat = DataTreatment(route="v2", trimming=2, correction=1)
    trimmed_sd = treat.get_trimmed_sd(sd, pre_recog)
    corrected_sd = treat.get_corrected_sd(sd, pre_recog, trimmed_sd)

    D0, E0, qv0, xr_curve0 = sd.get_xr_data_separate_ly()
    D1, E1, qv1, xr_curve1 = trimmed_sd.get_xr_data_separate_ly()
    D2, E2, qv2, xr_curve2 = corrected_sd.get_xr_data_separate_ly()

    with plt.Dp():
        fig, axes = plt.subplots(ncols=3, figsize=(18,5), subplot_kw=dict(projection="3d"))
        simple_plot_3d(axes[0], D0, x=qv0, y=xr_curve0.x)
        simple_plot_3d(axes[1], D1, x=qv1, y=xr_curve1.x)
        simple_plot_3d(axes[2], D2, x=qv2, y=xr_curve2.x)
        fig.tight_layout()
        plt.show()
