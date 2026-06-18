"""
    Estimators.EghEstimator.py

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.SecTheory.SecEstimator import guess_initial_secparams
from molass_legacy.Baseline.Constants import SLOPE_SCALE
from molass_legacy.Peaks.ElutionModels import egh


class _EghCurveAdapter:
    """Adapts EGH params to the curve interface used by EdmEstimatorImpl.

    Exposes .x and .y arrays (needed by optimize_edm_xr_decomposition for
    EGH peak-frame detection) as well as the get_y() method (used by
    guess_multiple_impl).
    """
    def __init__(self, x, egh_params):
        self._params = egh_params
        self.x = x
        self.y = np.maximum(egh(x, *egh_params[:4]), 0.0)

    def get_y(self):
        return self.y

if True:
    from importlib import reload
    import molass_legacy.Estimators.BaseEstimator
    reload(molass_legacy.Estimators.BaseEstimator)
from .BaseEstimator import BaseEstimator

class EghEstimator(BaseEstimator):
    def __init__(self, editor):
        BaseEstimator.__init__(self, editor)

    def estimate_egh_params(self, lrf_src=None, debug=False):
        editor = self.editor

        init_xr_params, init_xr_baseparams, init_mapping, init_uv_heights, temp_uv_baseparams = get_peak_params_advanced(editor, lrf_src=lrf_src, debug=debug)

        # Replace legacy EGH params with library-quality values when available.
        # editor.decomposition is built in background by _build_library_decomposition,
        # which uses len(peak_params_set[1]) components with proportions=[1]*nc.
        decomp = editor.decomposition
        if decomp is not None and len(decomp.xr_ccurves) == len(init_xr_params):
            lib_params = np.array([cc.params[:4] for cc in decomp.xr_ccurves])
            order = np.argsort(lib_params[:, 1])          # sort by tR (ascending)
            init_xr_params = lib_params[order]
            editor.logger.info(
                "estimate_egh_params: replaced legacy EGH with library EghPeeler results (n=%d)",
                len(init_xr_params),
            )
            # Also inject library UV heights — bypasses timing dependency on peak_params_set[0].
            # uv_ccurves are built by optimize_uv_decomposition with proportional XR seeds,
            # so their heights reflect actual UV absorption per component.
            if decomp.uv_ccurves is not None and len(decomp.uv_ccurves) == len(init_xr_params):
                lib_uv = sorted(decomp.uv_ccurves, key=lambda cc: cc.params[1])  # sort by tR
                init_uv_heights = np.array([cc.params[0] for cc in lib_uv])
                editor.logger.info(
                    "estimate_egh_params: replaced legacy UV heights with library uv_ccurves (n=%d)",
                    len(init_uv_heights),
                )

        init_uv_baseparams = temp_uv_baseparams.copy()
        init_uv_baseparams[4:6] /=SLOPE_SCALE


        (xr_curve, D), rg_curve = editor.dsets[0:2]

        init_rgs = rg_curve.get_rgs_from_trs(init_xr_params[:,1])
        Npc, rp, tI, t0, P, m = guess_initial_secparams(init_xr_params, init_rgs)
        init_sec_params = np.array([Npc, rp, tI, t0, P, m])
    
        x = xr_curve.x
        init_mappable_range = (x[0], x[-1])

        if debug:
            self.logger.info("init_xr_params=%s", str(init_xr_params))
            self.logger.info("init_xr_baseparams=%s", str(init_xr_baseparams))
            self.logger.info("init_rgs=%s", str(init_rgs))
            self.logger.info("init_mapping=%s", str(init_mapping))
            self.logger.info("init_uv_heights=%s", str(init_uv_heights))
            self.logger.info("init_mappable_range=%s", str(init_mappable_range))
            self.logger.info("init_sec_params=%s", str(init_sec_params))

        return init_xr_params, init_xr_baseparams, init_rgs, init_mapping, init_uv_heights, init_uv_baseparams, init_mappable_range, init_sec_params

    def estimate_params(self, lrf_src=None, debug=False):
        init_xr_params, init_xr_baseparams, init_rgs, init_mapping, init_uv_heights, init_uv_baseparams, init_mappable_range, seccol_params = self.estimate_egh_params(lrf_src=lrf_src, debug=debug)
        init_params = np.concatenate([init_xr_params.flatten(), init_xr_baseparams, init_rgs, init_mapping, init_uv_heights, init_uv_baseparams, init_mappable_range, seccol_params])

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            with plt.Dp():
                fig, axes = plt.subplots(ncols=2, figsize=(12,5))
                axis_info = (fig, (*axes, None, None)) 
                self.editor.fullopt.objective_func(init_params, plot=True, axis_info=axis_info)
                fig.tight_layout()
                plt.show()

        return init_params

def estimate_uv_weights_from_peaks(model_curves, x, mapping, uv_x, uv_y):
    """
    Per-component UV scale: UV data value at each model component's peak
    position divided by the XR model peak value.

    This is the same approach used by molass.SEC.Models.UvOptimizer and
    can be called from any model estimator (EDM, CEDM, LKM, ...).

    Parameters
    ----------
    model_curves : list of 1-D array
        Evaluated XR model curve for each component on grid ``x``.
    x : 1-D array
        XR frame positions.
    mapping : (float, float)
        (a_mp, b_mp) mapping XR frame → UV frame  (uv_frame = a*x + b).
    uv_x, uv_y : 1-D array
        UV elution curve x / y arrays.

    Returns
    -------
    uv_w : 1-D array  (nc,)
        UV scale per component, clipped to [1e-3, 1e3].
    """
    a_mp, b_mp = mapping
    uv_w = []
    for cy in model_curves:
        peak_xr_idx   = int(np.argmax(cy))
        peak_xr_val   = float(cy[peak_xr_idx])
        peak_xr_frame = float(x[peak_xr_idx])
        peak_uv_frame = a_mp * peak_xr_frame + b_mp
        uv_idx  = int(np.argmin(np.abs(uv_x - peak_uv_frame)))
        uv_val  = float(uv_y[uv_idx])
        s0 = uv_val / peak_xr_val if peak_xr_val > 1e-15 else 1.0
        uv_w.append(float(np.clip(s0, 1e-3, 1e3)))
    return np.array(uv_w)


def get_peak_params_advanced(editor, lrf_src=None, affine=False, debug=False):
    # note that this is called from the estimators

    a, b = editor.get_pre_recog_mapping_params()
    if lrf_src is None:
        uv_peaks, xr_peaks = editor.peak_params_set[0:2]
    else:
        # consider making this a method of the LrfSource class to take into account the rg_info indeces
        uv_peaks = lrf_src.uv_peaks
        xr_peaks = lrf_src.xr_peaks

    if not affine:
        non_affine_params = []
        for params in xr_peaks:
            non_affine_params.append(params[0:4])
        xr_peaks = np.asarray(non_affine_params)

    xr_base_params = editor.baseline_params[1]
    editor.logger.info("get_peak_params_advanced: xr_base_params=%s", str(xr_base_params))

    init_mapping = a, b
    uv_heights = [peak[0] for peak in uv_peaks]
    editor.logger.info("get_peak_params_advanced: uv_heights=%s", str(uv_heights))

    uv_base_params = editor.get_uv_base_params(debug=debug)

    return np.array(xr_peaks), np.array(xr_base_params), np.array(init_mapping), np.array(uv_heights), uv_base_params
