"""
    Estimators.SdmEstimator.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SecTheory.RetensionTime import estimate_init_rgs
from molass_legacy.Peaks.PeProgressConstants import MAXNUM_STEPS, STOCH_INIT_STEPS
from .BaseEstimator import BaseEstimator

class SdmEstimator(BaseEstimator):
    def __init__(self, editor):
        BaseEstimator.__init__(self, editor)

    def estimate_params(self, lrf_src=None, edm_available=False, debug=False):
        return self.compute_sdm_init_params(self.nc, lrf_src=lrf_src, edm_available=edm_available, debug=debug)

    def compute_sdm_init_params(self, nc_b, lrf_src=None, edm_available=False, debug=False):
        if debug:
            from importlib import reload
            import molass_legacy.Models.Stochastic.DispersiveMonopore
            reload(molass_legacy.Models.Stochastic.DispersiveMonopore)
            import molass_legacy.Models.Stochastic.MonoporeUvScaler
            reload(molass_legacy.Models.Stochastic.MonoporeUvScaler)
            import molass_legacy.Estimators.SdmEstimatorImpl
            reload(molass_legacy.Estimators.SdmEstimatorImpl)
            import molass_legacy.Peaks.PeakFronting
            reload(molass_legacy.Peaks.PeakFronting)
        from molass_legacy.Models.Stochastic.DispersiveMonopore import guess_params_using_moments
        from molass_legacy.Models.Stochastic.MomentUtils import compute_egh_moments
        
        from .SdmEstimatorImpl import guess_exec_spec, edit_to_full_sdmparams
        from molass_legacy.Peaks.PeakFronting import has_fronting_peak
 
        editor = self.editor
        optimizer = editor.fullopt
    
        progress = MAXNUM_STEPS - STOCH_INIT_STEPS
        editor.update_status_bar("Estimating stochastic initial parameters.")
     
        nc = nc_b - 1   # num components without baseline
        (xr_curve, D), rg_curve, (uv_curve, U) = editor.dsets

        if lrf_src is None:
            lrf_src = editor.get_lrf_source(devel=True)
            if lrf_src is None:
                assert False, "No LRF source"

        info = lrf_src.compute_rgs(want_num_components=nc, debug=False)
        if info is None:
            return

        rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = info
        peaks = lrf_src.xr_peaks[indeces]
        if edm_available:
            # removing the area M[0]
            egh_moments_list = [M[1:] for M in optimizer.compute_moments_list(debug=True)]
        else:
            egh_moments_list = compute_egh_moments(peaks)
        x = lrf_src.xr_x
        y = lrf_src.xr_y
        exec_spec = guess_exec_spec(peak_rgs, props, qualities)
        self.logger.info("init_params are estimated using exec_spec: %s", exec_spec)
        fronting = has_fronting_peak(xr_curve, debug=debug)
        if debug:
            print("indeces=", indeces, "qualities=", qualities)
            print("fronting=", fronting)                   
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("compute_sdm_init_params debug")
                if edm_available:
                    axt = ax.twinx()
                    axt.grid(False)
                    axis_info = (fig, (None, ax, None, axt))
                    optimizer.objective_func(optimizer.init_params, plot=True, axis_info=axis_info)
                else:
                    model = lrf_src.model
                    ax.plot(x, y, label="data")
                    for k, (params, rg, quality) in enumerate(zip(peaks, peak_rgs, qualities)):
                        cy = model(x, params)
                        ax.plot(x, cy, ":", label="Rg=%.3g (%.3g)" % (rg, quality))
                    ax.legend()
                fig.tight_layout()
                ret = plt.show()
            if not ret:
                return
        ret = guess_params_using_moments(x, y, egh_moments_list, peak_rgs, qualities, props,
                                            fronting=fronting,
                                            exec_spec=exec_spec, debug=debug)
        if ret is None:
            return
    
        progress += 1
        editor.pbar["value"] = progress
        editor.update()

        sdm_params, corrected_rgs, bounds = ret
        self.bounds = bounds

        init_params = edit_to_full_sdmparams(editor, sdm_params, corrected_rgs, uv_curve, debug=debug)
        if init_params is None:
            return

        progress += 1
        editor.pbar["value"] = progress
        editor.update_status_bar("Stochastic initial parameters are ready.")
        self.logger.info("init_params=%s", str(init_params))

        return init_params

    def get_colparam_bounds_bug(self):
        from molass_legacy.Models.Stochastic.DispersiveUtils import NUM_SDMCOL_PARAMS

        
        est_col_bounds = list(self.bounds[0:NUM_SDMCOL_PARAMS])
        return est_col_bounds[0:4] + [(1600, 60000)] + est_col_bounds[4:]

    def get_colparam_bounds(self):
        # temporary fix for bug
        from molass_legacy.Models.Stochastic.ParamLimits import MNP_BOUNDS
        # MNP_BOUNDS = [N_BOUND, KT_BOUND, T0_BOUND, PORESIZE_BOUNDS]
        # N, K, x0, poresize, N0, tI = sdmcol_params
        mnp_bounds = MNP_BOUNDS.copy()
        return mnp_bounds + [(1600, 60000), (-1000, 0)]

def onthefly_test(editor):
    estimator = SdmEstimator(editor)
    print("estimating...")
    init_params = estimator.estimate_params(debug=True)
    print("done.")
    if init_params is not None:
        editor.fullopt.params_type.set_estimator(estimator)
        def components_plot_debug():
            from importlib import reload
            import molass_legacy.Estimators.SdmEstimatorDebug
            reload(molass_legacy.Estimators.SdmEstimatorDebug)
            from .SdmEstimatorDebug import components_plot_debug_impl
            components_plot_debug_impl(editor.fullopt, init_params)

        with plt.Dp(extra_button_specs=[("Components Plot Debug", components_plot_debug)]):
            fig, ax = plt.subplots()
            ax.set_title("components_plot_debug")
            ret = plt.show()
        if not ret:
            print("debug done.")
            return
        editor.draw_scores(init_params, create_new_optimizer=False)
        print("redraw done.")