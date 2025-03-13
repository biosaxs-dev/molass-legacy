"""
    Estimators.EdmEstimator.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Peaks.PeProgressConstants import MAXNUM_STEPS, STOCH_INIT_STEPS
from .EghEstimator import EghEstimator

class EdmEstimator(EghEstimator):
    def __init__(self, editor, n_components):
        self.n_components = n_components
        EghEstimator.__init__(self, editor)

    def estimate_params(self, debug=False):
        if debug:
            from importlib import reload
            import Models.RateTheory.EDM
            reload(Models.RateTheory.EDM)
        from molass_legacy.Models.RateTheory.EDM import guess_multiple_impl, edm_impl

        init_xr_params, init_xr_baseparams, temp_rgs, init_mapping, init_uv_heights, init_uv_baseparams, init_mappable_range, seccol_params = self.estimate_egh_params()

        editor = self.editor
        progress = MAXNUM_STEPS - STOCH_INIT_STEPS
        editor.update_status_bar("Estimating EDM initial parameters.")

        nc = self.n_components - 1   # num components without baseline

        uv_curve, xr_curve = self.ecurves

        x = xr_curve.x
        y = xr_curve.y

        xr_params = guess_multiple_impl(x, y, nc, debug=debug)
        uv_w = np.array([uv_curve.max_y/xr_curve.max_y] * nc)

        progress += 1
        editor.pbar["value"] = progress
        editor.update()

        baseline_type = get_setting("unified_baseline_type")
        if baseline_type >= 2:
            # recompute uv_baseparams
            # to be implemented

            uv_base_params = init_uv_baseparams
        else:
            uv_base_params = init_uv_baseparams

        Tz = np.average(xr_params[:,0])
        init_params = np.concatenate([xr_params.flatten(), init_xr_baseparams, temp_rgs, init_mapping, uv_w, uv_base_params, init_mappable_range, [Tz]])

        progress += 1
        editor.pbar["value"] = progress
        editor.update_status_bar("EDM initial parameters are ready.")

        return init_params

def onthefly_test(editor):
    optimizer = editor.optimizer
    n_components = optimizer.params_type.n_components
    estimator = EdmEstimator(editor, n_components)
    print("estimating...")
    init_params = estimator.estimate_params(debug=True)
    print("done.")
    if init_params is None:
        return
    
    # optimizer.params_type.set_estimator(estimator)
    def draw_params(params, fig, axes):
        optimizer.objective_func(params, plot=True, axis_info=(fig, axes))

    with plt.Dp():
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))
        axt = ax2.twinx()
        axt.grid(False)
        fig.suptitle("EdmEstimator onthefly_test at PeakEditor")
        draw_params(init_params, fig, (ax1, ax2, ax3, axt))
        fig.tight_layout()
        ret = plt.show()
    if not ret:
        print("debug done.")
        return

    editor.draw_scores(init_params, create_new_optimizer=False)
    print("redraw done.")
