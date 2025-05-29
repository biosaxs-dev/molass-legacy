"""
    Estimators.StcEstimator.py

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SecTheory.RetensionTime import estimate_init_rgs
from molass_legacy.Peaks.PeProgressConstants import MAXNUM_STEPS, STOCH_INIT_STEPS
from .BaseEstimator import BaseEstimator

class StcEstimator(BaseEstimator):
    def __init__(self, editor):
        BaseEstimator.__init__(self, editor)

    def estimate_params(self, debug=False):
        return self.compute_stochastic_init_params(self.nc, self.t0_upper_bound, debug=debug)

    def compute_stochastic_init_params(self, nc_b, t0_upper_bound, debug=False):
        if debug:
            from importlib import reload
            import molass_legacy.SecTheory.MonoPore
            reload(molass_legacy.SecTheory.MonoPore)
            import molass_legacy.Models.Stochastic.MonoporeUvScaler
            reload(molass_legacy.Models.Stochastic.MonoporeUvScaler)
        from molass_legacy.SecTheory.MonoPore import estimate_monopore_params, get_modified_params, split_params, estimate_uv_scale_params
        from molass_legacy.Models.Stochastic.MonoporeUvScaler import adjust_to_uv_scales

        editor = self.editor

        progress = MAXNUM_STEPS - STOCH_INIT_STEPS
        editor.update_status_bar("Estimating stochastic initial parameters.")
     
        nc = nc_b - 1   # num components without baseline
        (xr_curve, D), rg_curve, (uv_curve, U) = editor.dsets

        lrf_src = editor.get_lrf_source(devel=True)
        if lrf_src is None:
            ret = estimate_monopore_params(xr_curve, rg_curve, nc, optimizer=editor.fullopt,
                                           t0_upper_bound=t0_upper_bound, logger=self.logger, debug=debug)
            init_params_for_uv = ret.x
            monopore_params, rgs, xr_w = split_params(ret.x, nc)
        else:
            ret = lrf_src.guess_monopore_params(debug=debug)
            if ret is None:
                return
            xr_init_params, corrected_rgs = ret
            N, T, x0, me, mp, poresize = xr_init_params[0:6]
            xr_w = xr_init_params[6:]
            rgs = corrected_rgs
            monopore_params = np.array([x0, poresize, N, me, T, mp])    # t0, rp, N, me, T, mp

        progress += 1
        editor.pbar["value"] = progress
        editor.update()

        _, _, xr_x, xr_y, baselines = editor.get_curve_xy(return_baselines=True)
        a, b = editor.peak_params_set[-2:]
        uv_x = xr_x*a + b
        uv_y = uv_curve.spline(uv_x)

        uv_baseline = editor.get_uv_baseline(xy=(uv_x, uv_y))

        uv_y_ = uv_y - uv_baseline
        xr_y_ = xr_y - baselines[1]

        ret = adjust_to_uv_scales(xr_x, xr_y_, uv_x, uv_y_, xr_init_params, corrected_rgs, debug=debug)
        if ret is None:
            return
        else:
            uv_w, uv_ty = ret

        # uv_w, uv_ty = estimate_uv_scale_params(xr_curve, rg_curve, uv_curve, nc, init_params_for_uv, xr_w, uv_x, uv_y_, xr_x, xr_y_, optimizer=editor.fullopt, debug=debug)

        uv_base_params = editor.get_uv_base_params(xyt=(uv_x, uv_y, uv_ty))

        if debug:
            self.logger.info("uv_base_params=%s, average(uv_ty)=%g", str(uv_base_params), np.average(uv_ty))
            uv_bl_computed = editor.base_curve_info[0](uv_x, uv_base_params, uv_ty)
            with plt.Dp():
                fig,(ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("compute_stochastic_init_params")
                ax1.plot(uv_x, uv_y)
                ax1.plot(uv_x, uv_ty + uv_baseline, ":")
                ax1.plot(uv_x, uv_baseline, color="red")
                ax1.plot(uv_x, uv_bl_computed, color="yellow")
                ax2.plot(xr_x, xr_y)
                ax2.plot(xr_x, baselines[1], color="red")
                fig.tight_layout()
                plt.show()

        param_list = [xr_w, editor.baseline_params[1], rgs, (a, b), uv_w, uv_base_params, xr_x[[0,-1]], monopore_params]
        init_params = np.concatenate(param_list)

        progress += 1
        editor.pbar["value"] = progress
        editor.update_status_bar("Stochastic initial parameters are ready.")

        return init_params

def onthefly_test(editor):
    estimator = StcEstimator(editor)
    print("estimating...")
    init_params = estimator.estimate_params(debug=True)
    print("done.")
    if init_params is not None:
        editor.draw_scores(init_params)
        print("redraw done.")