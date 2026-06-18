"""
    Estimators.LkmEstimator.py

    Estimates initial parameters for LKM (Lumped Kinetic Model) rigorous
    optimization (G1400 / LkmParams).

    LKM layout (see LkmParams.py):
        xr_params       : nc            [scale_k per component]
        xr_baseparams   : num_baseparams
        rg_params       : nc
        mapping         : (a_mp, b_mp)
        uv_params       : nc
        uv_baseparams   : 5 + num_baseparams
        mappable_range  : (c, d)
        lkm_colparams   : [Pe, t0, R_0, k_MT_0, R_1, k_MT_1, ..., R_{nc-1}, k_MT_{nc-1}]

    Strategy: delegate to molass-library's LkmEstimator.estimate_lkm_init_params
    via a thin adapter.  This keeps the algorithm in a single place — any
    improvement to the library implementation is automatically inherited here.

      1. Get EGH component params from estimate_egh_params().
      2. Wrap each EGH component as an _EghCurveAdapterForLkm (has get_xy()).
      3. Wrap adapters in _FakeDecomp (has xr_ccurves attribute).
      4. Call molass.SEC.Models.LkmEstimator.estimate_lkm_init_params →
         (Pe, t0, k_MT_list, R_list, scale_list).
      5. Build lkm_colparams: [Pe, t0, R_0, k_MT_0, R_1, k_MT_1, ...].
      6. Use scale_list as xr heights (area = scale for LKM PDF normalization).

    Copyright (c) 2026, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.Peaks.ElutionModels import egh
from molass_legacy.Peaks.PeProgressConstants import MAXNUM_STEPS, STOCH_INIT_STEPS
from .EghEstimator import EghEstimator


class _EghCurveAdapterForLkm:
    """Adapts EGH params to the get_xy() interface used by estimate_lkm_init_params."""
    def __init__(self, x, egh_params):
        self._x = x
        self._params = egh_params

    def get_xy(self):
        y = np.maximum(egh(self._x, *self._params[:4]), 0.0)
        return self._x, y


class _FakeDecomp:
    """Minimal decomposition-like object carrying xr_ccurves for LkmEstimator."""
    def __init__(self, xr_ccurves):
        self.xr_ccurves = xr_ccurves


class LkmEstimator(EghEstimator):
    def __init__(self, editor, n_components):
        self.n_components = n_components
        EghEstimator.__init__(self, editor)

    def estimate_params(self, debug=False):
        # Fast path: use library LKM upgrade result directly.
        editor = self.editor
        model_decomp = getattr(editor, 'model_decomposition', None)
        if model_decomp is not None and getattr(model_decomp, 'model', None) == 'lkm':
            try:
                from molass.Rigorous.LegacyBridgeUtils import make_basecurves_from_decomposition
                _, baseparams = make_basecurves_from_decomposition(model_decomp)
                init_params = model_decomp.make_rigorous_initparams(baseparams)
                self.logger.info("LkmEstimator: used library LKM upgrade result directly")
                return init_params
            except Exception as _e:
                self.logger.warning(
                    "LkmEstimator: library fast path failed (%s); falling back to legacy path", _e
                )

        if debug:
            from importlib import reload
            import molass.SEC.Models.LkmEstimator
            reload(molass.SEC.Models.LkmEstimator)
        from molass.SEC.Models.LkmEstimator import estimate_lkm_init_params

        (init_xr_params, init_xr_baseparams, temp_rgs, init_mapping,
         init_uv_heights, init_uv_baseparams, init_mappable_range,
         seccol_params) = self.estimate_egh_params()

        editor = self.editor
        progress = MAXNUM_STEPS - STOCH_INIT_STEPS
        editor.update_status_bar("Estimating LKM initial parameters.")

        nc = self.n_components - 1   # num components without baseline

        uv_curve, xr_curve = self.ecurves
        x = xr_curve.x

        # Delegate to library: moment-matching to estimate Pe, t0, R_i, k_MT_i.
        # Returns Pe, t0, k_MT_list, R_list, scale_list (nc values each for lists).
        adapters = [_EghCurveAdapterForLkm(x, init_xr_params[k]) for k in range(nc)]
        fake_decomp = _FakeDecomp(adapters)
        Pe, t0, k_MT_list, R_list, scale_list = estimate_lkm_init_params(
            fake_decomp, debug=debug)

        progress += 1
        editor.pbar["value"] = progress
        editor.update()

        # Build lkm_colparams: [Pe, t0, R_0, k_MT_0, R_1, k_MT_1, ...]
        lkm_colparams = [Pe, t0]
        for i in range(nc):
            lkm_colparams.append(R_list[i])
            lkm_colparams.append(k_MT_list[i])
        lkm_colparams = np.array(lkm_colparams)

        # xr heights: area under each EGH component curve.
        # G1400 uses xr_w * lkm_pdf(x, ...) where lkm_pdf integrates to ~1,
        # so xr_w should equal the total area under the component curve.
        xr_heights = np.array(scale_list)

        # Per-component UV weights: preserve the UV/XR height ratio from EGH,
        # scaled to the LKM area-based xr scale.
        # uv_w[k] = scale_list[k] * (U_k / H_k)
        # This gives area-unit UV weights consistent with xr_heights (= scale_list).
        # Note: EDM/CEDM use the peak-lookup approach (different unit convention).
        egh_xr_heights = np.array([p[0] for p in init_xr_params])  # H_k from EGH
        egh_uv_heights = np.array(init_uv_heights)
        safe_egh_xr = np.where(egh_xr_heights > 0, egh_xr_heights, 1.0)
        uv_w = xr_heights * (egh_uv_heights / safe_egh_xr)

        editor.update_status_bar("LKM initial parameters are ready.")

        return np.concatenate([
            xr_heights,             # nc: scale per component
            init_xr_baseparams,     # num_baseparams
            temp_rgs,               # nc
            init_mapping,           # (a_mp, b_mp)
            uv_w,                   # nc
            init_uv_baseparams,     # 5 + num_baseparams
            init_mappable_range,    # (c, d)
            lkm_colparams,          # [Pe, t0, R_0, k_MT_0, ...]
        ])
