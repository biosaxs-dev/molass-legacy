"""
    Estimators.NedmEstimator.py

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Peaks.PeProgressConstants import MAXNUM_STEPS, STOCH_INIT_STEPS
from .EghEstimator import EghEstimator

class NedmEstimator(EghEstimator):
    def __init__(self, editor, n_components):
        self.n_components = n_components
        EghEstimator.__init__(self, editor)

    def estimate_params(self, debug=False):
        """G2010/NEDM (non-constrained EDM) init.

        molass-library has no implementation of this non-shared-column variant
        (its free-EDM path was removed; ``upgrade('EDM')`` now always returns
        the shared-column/constrained model used by G2020), so there is no
        library fast path here -- this always uses the legacy per-component
        EDM fitting approach below.
        """
        if debug:
            from importlib import reload
            import molass_legacy.Models.RateTheory.EDM
            reload(molass_legacy.Models.RateTheory.EDM)
        from molass_legacy.Models.RateTheory.EDM import edm_impl
        from .EghEstimator import _EghCurveAdapter

        init_xr_params, init_xr_baseparams, temp_rgs, init_mapping, init_uv_heights, init_uv_baseparams, init_mappable_range, seccol_params = self.estimate_egh_params()

        editor = self.editor
        progress = MAXNUM_STEPS - STOCH_INIT_STEPS
        editor.update_status_bar("Estimating NEDM initial parameters.")

        nc = self.n_components - 1   # num components without baseline

        uv_curve, xr_curve = self.ecurves

        x = xr_curve.x
        y = xr_curve.y

        try:
            from molass.SEC.Models.EdmEstimatorImpl import guess_multiple_impl
            adapters = [_EghCurveAdapter(x, init_xr_params[k]) for k in range(nc)]
            xr_params = guess_multiple_impl(x, y, adapters, debug=debug)
        except Exception:
            from molass_legacy.Models.RateTheory.EDM import guess_multiple_impl as _legacy_gmi
            xr_params = _legacy_gmi(x, y, nc, debug=debug)

        # Per-component UV weights via peak-lookup — same method as UvOptimizer.
        # Evaluate each NEDM component curve; look up UV value at mapped peak frame.
        from .EghEstimator import estimate_uv_weights_from_peaks
        model_curves = [edm_impl(x, *xr_params[k]) for k in range(nc)]
        uv_w = estimate_uv_weights_from_peaks(
            model_curves, x, init_mapping, uv_curve.x, uv_curve.y)

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
        editor.update_status_bar("NEDM initial parameters are ready.")

        return init_params

def onthefly_test(editor):
    optimizer = editor.optimizer
    n_components = optimizer.params_type.n_components
    estimator = NedmEstimator(editor, n_components)
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
        fig.suptitle("NedmEstimator onthefly_test at PeakEditor")
        draw_params(init_params, fig, (ax1, ax2, ax3, axt))
        fig.tight_layout()
        ret = plt.show()
    if not ret:
        print("debug done.")
        return

    editor.draw_scores(init_params, create_new_optimizer=False)
    print("redraw done.")
