"""
    V2PropOptimizer.V1PreviewAdapter.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy._MOLASS.SerialSettings import set_setting
from molass_legacy.Mapping.MapperConstructor import create_mapper
from molass_legacy.SerialAnalyzer.AnalyzerUtil import compute_conc_factor_util
from Extrapolation.PreviewData import PreviewOptions

class ZxDialogProxy:
    def __init__(self, parent):
        self.parent = parent

class JudgeHolderProxy:
    def __init__(self, optimizer):
        peak_top_x = optimizer.xr_curve.peak_top_x.copy()
        cd_colors = ['green'] * len(peak_top_x)
        set_setting('mapper_cd_color_info', (peak_top_x, cd_colors))

class MapperProxy:
    def __init__(self, optimizer):
        self.x_curve = optimizer.xr_curve       # better be corrected
        self.optimizer = optimizer

    def get_conc_vector(self, conc_factor):
        optimizer = self.optimizer
        x = optimizer.xr_curve.x
        a, b = optimizer.separate_params[3]
        uv_x = a*x+b
        uv_y = optimizer.uv_curve.spline(uv_x)
        return uv_y * conc_factor

class PdataProxy:
    def __init__(self, vp_analysis, treat, devel=True):
        if devel:
            from importlib import reload
            import V2PropOptimizer.V1RangeCoverter
            reload(V2PropOptimizer.V1RangeCoverter)
        from .V1RangeCoverter import convert_to_v1_model_ranges

        prop_optimizer = vp_analysis.prop_optimizer
        peaks = vp_analysis.get_current_peaks()
        paired_ranges = vp_analysis.paired_ranges
        self.cnv_ranges, ret_ranges = convert_to_v1_model_ranges(prop_optimizer, peaks, paired_ranges)
        self.paired_ranges = ret_ranges     # see convert_to_v1_model_ranges, DecompEditorFrame.make_range_info
        self.num_ranges = np.sum([len(prange.get_fromto_list()) for prange in self.cnv_ranges])
        optimizer = vp_analysis.optimizer
        self.judge_holder = JudgeHolderProxy(optimizer)
        self.is_for_sec = True
        self.conc = None
        self.conc_curves = None
        # self.mapper = create_mapper(None, treat.trimmed_sd, treat.sd_orig, treat.pre_recog)
        self.mapper = MapperProxy(optimizer)

        sd = treat.corrected_sd
        """
        consider moving these to SecSaxs.DataSet
        """
        sd.pre_recog = treat.pre_recog     # to fix AttributeError: 'DataSet' object has no attribute 'pre_recog', but not sure whether this fix is ok?
        self.conc_factor = compute_conc_factor_util()
        sd.mc_vector = self.mapper.get_conc_vector(self.conc_factor)
        sd.xr_j0 = sd.get_xr_curve().x[0]
        sd.get_xray_elution_vector()        # to set sd.xr_index

        self.sd = sd

    def make_conc_vector(self):
        # y_ = self.mapper.make_uniformly_scaled_vector(scale=1)
        self.mc_vector = self.sd.mc_vector
        return self.mc_vector

class V1PreviewAdapter:
    def __init__(self, dialog):
        self.dialog = dialog

    def show_zx_preview_using_pool(self, pdata, popts, devel=True):
        """
        adaption to PreviewButtonFrame.show_zx_preview_using_pool
        """

        if devel:
            from importlib import reload
            import LRF.LrfResultPool
            reload(LRF.LrfResultPool)       
        from LRF.LrfResultPool import LrfResultPool
        self.pool = LrfResultPool(pdata, popts)
        self.pool.run_solver(self.dialog)
        self.pool.show_dialog(self.dialog.parent)

def observe_it_as_v1_preview(vp_analysis):
    print("observe_it_as_v1_preview")

    dialog = ZxDialogProxy(vp_analysis.parent)
    adapter = V1PreviewAdapter(dialog)

    optinit_info = vp_analysis.js_canvas.optinit_info       # InitialInfo
    treat = optinit_info.treat

    pdata = PdataProxy(vp_analysis, treat)
    conc_dependence = 1
    set_setting('conc_dependence', conc_dependence)
    popts = PreviewOptions(conc_depend=conc_dependence, use_elution_models=True)
    concentration_datatype = 0      # 0: XR model, 1: XR data, 2: UV model, 3: UV data
    set_setting('concentration_datatype', concentration_datatype)

    set_setting('uv_baseline_type', 1)      # required in MappingParams.get_uv_correction_str()
    set_setting('xray_baseline_type', 1)    # 

    adapter.show_zx_preview_using_pool(pdata, popts)
