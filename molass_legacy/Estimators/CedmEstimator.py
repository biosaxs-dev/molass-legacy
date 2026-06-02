"""
    Estimators.CedmEstimator.py

    Estimates initial parameters for Constrained-EDM (G2020 / CedmParams).

    CEDM layout (see CedmParams.py):
        xr_params       : (nc × 3)  [a_k, b_k, cinj_k]   per component
        xr_baseparams   : num_baseparams
        rg_params       : nc
        mapping         : (a_mp, b_mp)
        uv_params       : nc
        uv_baseparams   : 5 + num_baseparams
        mappable_range  : (c, d)
        cedm_colparams  : [t0_sh, u_sh, e_sh, Dz_sh]  (shared, appended at end)

    Strategy: delegate to molass-library's EdmEstimatorImpl.estimate_cedm_shared_params
    via a thin adapter.  This keeps the algorithm in a single place — any
    improvement to the library implementation is automatically inherited here.

      1. Get EGH component params from estimate_egh_params().
      2. Wrap each EGH component as an _EghCurveAdapter (.x, .y, get_y()).
      3. Call molass.SEC.Models.EdmEstimatorImpl.estimate_cedm_shared_params
         → (cedm_colparams [t0_sh, u_sh, e_sh, Dz_sh], abc_params [nc×3]).
         This runs guess_multiple_impl (rough) then a shared-column L-BFGS-B
         optimisation, producing physically varied b values per component.

    Copyright (c) 2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.Peaks.PeProgressConstants import MAXNUM_STEPS, STOCH_INIT_STEPS
from .EghEstimator import EghEstimator, _EghCurveAdapter


class CedmEstimator(EghEstimator):
    def __init__(self, editor, n_components):
        self.n_components = n_components
        EghEstimator.__init__(self, editor)

    def estimate_params(self, debug=False):
        if debug:
            from importlib import reload
            import molass.SEC.Models.EdmEstimatorImpl
            reload(molass.SEC.Models.EdmEstimatorImpl)
        from molass.SEC.Models.EdmEstimatorImpl import estimate_cedm_shared_params

        (init_xr_params, init_xr_baseparams, temp_rgs, init_mapping,
         init_uv_heights, init_uv_baseparams, init_mappable_range,
         seccol_params) = self.estimate_egh_params()

        editor = self.editor
        progress = MAXNUM_STEPS - STOCH_INIT_STEPS
        editor.update_status_bar("Estimating CEDM initial parameters.")

        nc = self.n_components - 1   # num components without baseline

        uv_curve, xr_curve = self.ecurves
        x = xr_curve.x
        y = xr_curve.y

        # Delegate to library: rough EDM fit → shared-column L-BFGS-B optimisation.
        # Returns cedm_colparams [t0_sh, u_sh, e_sh, Dz_sh] and abc_params (nc×3).
        adapters = [_EghCurveAdapter(x, init_xr_params[k]) for k in range(nc)]
        cedm_colparams, abc_params = estimate_cedm_shared_params(x, y, adapters, debug=debug)

        progress += 1
        editor.pbar["value"] = progress
        editor.update()

        # Per-component UV weights via peak-lookup — same method as UvOptimizer.
        # Evaluate each CEDM component curve using shared column params.
        from .EghEstimator import estimate_uv_weights_from_peaks
        from molass_legacy.Models.RateTheory.EDM import edm_impl
        t0_sh, u_sh, e_sh, Dz_sh = cedm_colparams
        model_curves = [
            edm_impl(x, t0_sh, u_sh, abc_params[k, 0], abc_params[k, 1],
                     e_sh, Dz_sh, abc_params[k, 2])
            for k in range(nc)
        ]
        uv_w = estimate_uv_weights_from_peaks(
            model_curves, x, init_mapping, uv_curve.x, uv_curve.y)

        editor.update_status_bar("CEDM initial parameters are ready.")

        return np.concatenate([
            abc_params.flatten(),       # nc × 3: [a_k, b_k, cinj_k]
            init_xr_baseparams,         # num_baseparams
            temp_rgs,                   # nc
            init_mapping,               # (a_mp, b_mp)
            uv_w,                       # nc
            init_uv_baseparams,         # 5 + num_baseparams
            init_mappable_range,        # (c, d)
            cedm_colparams,             # [t0_sh, u_sh, e_sh, Dz_sh]
        ])
