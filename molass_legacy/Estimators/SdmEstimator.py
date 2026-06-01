"""
    Estimators.SdmEstimator.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SecTheory.RetensionTime import estimate_init_rgs
from molass_legacy.Peaks.PeProgressConstants import MAXNUM_STEPS, STOCH_INIT_STEPS
from .BaseEstimator import BaseEstimator

class _CurveProxy:
    """Minimal curve proxy providing both .get_xy() and direct .x / .y attributes.

    ``optimize_sdm_xr_decomposition`` accesses ``c.x[c.y.argmax()]`` directly;
    the moment estimators use ``ccurve.get_xy()``.
    """
    def __init__(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)

    def get_xy(self):
        return self.x, self.y

    def get_max_y(self):
        return float(np.max(self.y))


class _ProxyDecomposition:
    """Minimal Decomposition-like object for driving the library's 3-stage
    lognormal init pipeline from ``lrf_src`` data.

    Used by ``_estimate_lognormal`` to call, in order:
      1. ``estimate_sdm_column_params``   — multi-start mono-pore NM
      2. ``optimize_sdm_xr_decomposition`` — converged mono-pore curves
      3. ``estimate_sdm_lognormal_from_monopore`` — geometric mean + moment matching
    """
    def __init__(self, xr_x, xr_y, peaks, model, peak_rgs):
        self.xr_ccurves = [_CurveProxy(xr_x, model(xr_x, p)) for p in peaks]
        self.xr_icurve = _CurveProxy(xr_x, xr_y)
        self.num_components = len(peaks)
        self._rgs = list(np.asarray(peak_rgs))
        self.uv = None

    def get_rgs(self):
        return self._rgs


class SdmEstimator(BaseEstimator):
    def __init__(self, editor, pore_dist='mono', t0_upper_bound=None):
        BaseEstimator.__init__(self, editor, t0_upper_bound=t0_upper_bound)
        self.pore_dist = pore_dist
    
    def estimate_params(self, lrf_src=None, edm_available=False, debug=False):
        if self.pore_dist == 'lognormal':
            return self._estimate_lognormal(lrf_src=lrf_src, edm_available=edm_available, debug=debug)
        else:
            return self._estimate_mono(lrf_src=lrf_src, edm_available=edm_available, debug=debug)

    def _estimate_mono(self, lrf_src=None, edm_available=False, debug=False):
        """G1200 init: mono-pore + gamma — appends k_gamma=2.0 to the 6-element sdmcol."""
        init_params = self.compute_sdm_init_params(self.nc, lrf_src=lrf_src,
                                                   edm_available=edm_available, debug=debug)
        if init_params is None:
            return None
        # Append k_gamma (gamma shape param for pore residence time); initial value 2.0
        return np.append(init_params, 2.0)

    def _estimate_lognormal(self, lrf_src=None, edm_available=False, debug=False):
        """G1300 init: lognormal pore + gamma.

        Mirrors the library's ``upgrade(model='SDM', pore_dist='lognormal')``
        pipeline as closely as possible:
          1. ``estimate_sdm_column_params``    — multi-start NM → converged mono
          2. ``optimize_sdm_xr_decomposition`` — mono-pore NM → SdmComponentCurves
          3. ``estimate_sdm_lognormal_from_monopore`` — geometric mean poresize +
             moment matching → ``LognormalEnv(N, T, me, mp, N0, t0, mu, sigma)``

        Falls back to the legacy rough estimate (N0=50000, poresize from moments)
        if the library import fails.
        """
        # Run the full legacy mono-pore estimator first — we still need
        # corrected_rgs, self.bounds, self._xr_* attrs, and the UV/baseline
        # portion of init_params (indices 0..-6).
        init_params_6 = self.compute_sdm_init_params(self.nc, lrf_src=lrf_src,
                                                     edm_available=edm_available, debug=debug)
        if init_params_6 is None:
            return None
        # Legacy mono-pore column params: [N, K, x0, poresize, N0, tI]
        N, K, x0, poresize, N0, tI = init_params_6[-6:]

        # Drive the library's 3-stage lognormal init pipeline.
        # This replicates what upgrade() does before its final NM, giving BH
        # a much better starting point than the legacy rough moment estimate.
        try:
            from molass.SEC.Models.SdmEstimator import (
                estimate_sdm_column_params,
                estimate_sdm_lognormal_from_monopore,
            )
            from molass.SEC.Models.SdmOptimizer import optimize_sdm_xr_decomposition
            proxy = _ProxyDecomposition(
                self._xr_x, self._xr_y, self._xr_peaks, self._xr_model, self.peak_rgs
            )
            # Stage 1: multi-start mono-pore column param estimation
            mono_env = estimate_sdm_column_params(proxy)
            # Stage 2: converged mono-pore SdmComponentCurves
            mono_ccurves = optimize_sdm_xr_decomposition(proxy, mono_env)
            # Stage 3: lognormal (mu, sigma, t0, k) refined by moment matching
            ln_env = estimate_sdm_lognormal_from_monopore(
                mono_ccurves, proxy.xr_icurve, decomposition=proxy
            )
            N_lib, T_lib, _me, _mp, N0_lib, t0_lib, mu_lib, sigma_lib = ln_env
            self.logger.info(
                "Library lognormal init: N=%g, T=%g, N0=%g, t0=%g, mu=%g (poresize=%g Å), sigma=%g",
                N_lib, T_lib, N0_lib, t0_lib, mu_lib, np.exp(mu_lib), sigma_lib,
            )
            # Map LognormalEnv → G1300 sdmcol_8: [N, K, x0, mu, sigma, N0, tI, k_gamma]
            # Library's T ↔ legacy's K  (pore-residence time parameter)
            # Library's t0 is used for both x0 and tI (consistent with upgrade())
            sdmcol_8 = np.array([N_lib, T_lib, t0_lib, mu_lib, sigma_lib, N0_lib, t0_lib, 2.0])
        except Exception as e:
            self.logger.warning(
                "Library lognormal init failed (%s); falling back to legacy rough estimate.", e
            )
            N0 = 50000.0
            mu = np.log(max(float(poresize), 1.0))
            sdmcol_8 = np.array([N, K, x0, mu, 0.3, N0, tI, 2.0])

        return np.concatenate([init_params_6[:-6], sdmcol_8])

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
        self.peak_rgs = peak_rgs   # stored for _estimate_lognormal
        self._xr_x = x
        self._xr_y = y              # stored for _estimate_lognormal
        self._xr_peaks = peaks
        self._xr_model = lrf_src.model

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
        from molass_legacy.Models.Stochastic.ParamLimits import MNP_BOUNDS, LN_MU_BOUND, LN_SIGMA_BOUND
        mnp_bounds = MNP_BOUNDS.copy()
        if self.pore_dist == 'lognormal':
            # G1300: [N, K, x0, mu, sigma, N0, tI, k_gamma] (8 params)
            return list(mnp_bounds[:3]) + [LN_MU_BOUND, LN_SIGMA_BOUND,
                                           (1600, 60000), (-1000, 0), (0.5, 10.0)]
        else:
            # G1200: [N, K, x0, poresize, N0, tI, k_gamma] (7 params)
            return mnp_bounds + [(1600, 60000), (-1000, 0), (0.5, 10.0)]

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