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

    Strategy: delegate to molass-library's EdmEstimatorImpl.guess_multiple_impl
    via a thin adapter.  This keeps the algorithm in a single place — any
    improvement to the library implementation is automatically inherited here.

      1. Get EGH component params from estimate_egh_params().
      2. Wrap each EGH component as an _EghCurveAdapter (has get_y()).
      3. Call molass.SEC.Models.EdmEstimatorImpl.guess_multiple_impl → (nc × 7)
         [t0, u, a, b, e, Dz, cinj] per component, sorted by peak position.
      4. Derive shared CEDM column params as mean of [t0, u, e, Dz].
      5. Extract per-component [a, b, cinj].

    Copyright (c) 2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.Peaks.ElutionModels import egh
from molass_legacy.Peaks.PeProgressConstants import MAXNUM_STEPS, STOCH_INIT_STEPS
from .EghEstimator import EghEstimator


class _EghCurveAdapter:
    """Adapts EGH params to the get_y() interface used by EdmEstimatorImpl."""
    def __init__(self, x, egh_params):
        self._x = x
        self._params = egh_params

    def get_y(self):
        return np.maximum(egh(self._x, *self._params[:4]), 0.0)


class CedmEstimator(EghEstimator):
    def __init__(self, editor, n_components):
        self.n_components = n_components
        EghEstimator.__init__(self, editor)

    def estimate_params(self, debug=False):
        if debug:
            from importlib import reload
            import molass.SEC.Models.EdmEstimatorImpl
            reload(molass.SEC.Models.EdmEstimatorImpl)
        from molass.SEC.Models.EdmEstimatorImpl import guess_multiple_impl

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

        # Delegate to library: fit EDM per component then jointly optimise cinj.
        # Returns (nc × 7) sorted by peak position: [t0, u, a, b, e, Dz, cinj].
        adapters = [_EghCurveAdapter(x, init_xr_params[k]) for k in range(nc)]
        edm_xr_params = guess_multiple_impl(x, y, adapters, debug=debug)

        progress += 1
        editor.pbar["value"] = progress
        editor.update()

        # Shared column params: mean of per-component [t0(0), u(1), e(4), Dz(5)]
        t0_sh = float(np.mean(edm_xr_params[:, 0]))
        u_sh  = float(np.mean(edm_xr_params[:, 1]))
        e_sh  = float(np.mean(edm_xr_params[:, 4]))
        Dz_sh = float(np.mean(edm_xr_params[:, 5]))
        cedm_colparams = np.array([t0_sh, u_sh, e_sh, Dz_sh])

        abc_params = edm_xr_params[:, [2, 3, 6]]   # nc × 3: [a, b, cinj]

        uv_w = np.array([uv_curve.max_y / xr_curve.max_y] * nc)

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
