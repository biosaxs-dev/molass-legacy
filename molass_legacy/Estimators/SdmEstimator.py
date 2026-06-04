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
        """G1200 init: mono-pore + gamma.

        Mirrors the library's ``upgrade(model='SDM')`` pipeline (2 stages):
          1. ``estimate_sdm_column_params``    — multi-start NM → converged mono env
          2. ``optimize_sdm_xr_decomposition`` — converged mono-pore SdmComponentCurves

        Falls back to the legacy moment-matching estimate if the library import fails.
        """
        init_params_6 = self.compute_sdm_init_params(self.nc, lrf_src=lrf_src,
                                                      edm_available=edm_available, debug=debug)
        if init_params_6 is None:
            return None
        # Legacy mono-pore column params: [N, K, x0, poresize, N0, tI]
        N, K, x0, poresize, N0, tI = init_params_6[-6:]
        xr_scales = None
        nc_xr = 0

        try:
            from molass.SEC.Models.SdmEstimator import estimate_sdm_column_params
            from molass.SEC.Models.SdmOptimizer import optimize_sdm_xr_decomposition
            from molass_legacy._MOLASS.SerialSettings import get_setting
            proxy = _ProxyDecomposition(
                self._xr_x, self._xr_y, self._xr_peaks, self._xr_model, self.peak_rgs
            )
            # Stage 1: multi-start mono-pore column param estimation.
            # Pass the column-specific poresize_bounds (e.g. (71, 81) Å for Superdex 200).
            poresize_bounds = get_setting("poresize_bounds")
            mono_env = estimate_sdm_column_params(proxy, poresize_bounds=poresize_bounds)
            # Stage 2: converged mono-pore SdmComponentCurves
            mono_ccurves = optimize_sdm_xr_decomposition(proxy, mono_env)
            # Extract Stage-2 converged column params (shared across all components)
            N2, T2, _me, _mp, x0_2, tI_2, N0_2, poresize_2, _ts, k_2 = mono_ccurves[0].column.get_params()
            K_lib = N2 * T2   # Legacy K = N*T  (see DispersiveMonopore.py: "T_ = K_/N_")
            self.logger.info(
                "Library mono init (stage2): N=%g, T=%g, K=%g, N0=%g, t0=%g, poresize=%g Å, k=%g",
                N2, T2, K_lib, N0_2, x0_2, poresize_2, k_2,
            )
            xr_scales = np.array([cc.scale for cc in mono_ccurves])
            nc_xr = len(mono_ccurves)
            # Map → G1200 sdmcol_7: [N, K, x0, poresize, N0, tI, k_gamma]
            sdmcol_7 = np.array([N2, K_lib, x0_2, poresize_2, N0_2, tI_2, k_2])
        except Exception as e:
            self.logger.warning(
                "Library mono init failed (%s); falling back to legacy estimate.", e
            )
            sdmcol_7 = np.array([N, K, x0, poresize, N0, tI, 2.0])

        non_col = init_params_6[:-6].copy()
        if xr_scales is not None:
            non_col[:nc_xr] = xr_scales  # replace XR weights with Stage-2 scales
        return np.concatenate([non_col, sdmcol_7])

    def _estimate_lognormal(self, lrf_src=None, edm_available=False, debug=False):
        """G1300 init: lognormal pore + gamma.

        Fully mirrors the library's ``upgrade(model='SDM', pore_dist='lognormal')``
        pipeline (4 stages):
          1. ``estimate_sdm_column_params``              — multi-start NM → converged mono env
          2. ``optimize_sdm_xr_decomposition``           — mono-pore NM → SdmComponentCurves
          3. ``estimate_sdm_lognormal_from_monopore``    — geometric mean poresize +
             moment matching → ``LognormalEnv`` (SV≈52)
          4. ``optimize_sdm_lognormal_xr_decomposition`` — converged lognormal NM (SV≈72)

        Stage 4 matches the library notebook's BH starting point and is the key
        improvement over the previous 3-stage version (SV≈52 → SV≈72 at init).

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

        # Drive the library's 4-stage lognormal init pipeline.
        # This replicates what upgrade() does before its final NM, giving BH
        # a much better starting point than the legacy rough moment estimate.
        # Temporarily scale the progress bar to 4 steps (one per lognormal stage).
        # scipy blocks the main thread, so indeterminate/bouncing mode won't animate;
        # a determinate 0→4 bar ticked after each stage is clearer and more honest.
        editor = self.editor
        editor.pbar.configure(maximum=4, value=0, style='Phase2.Horizontal.TProgressbar')
        try:
            editor.update_status_bar("SDM lognormal init (1/4): estimating mono-pore column parameters...")
            from molass.SEC.Models.SdmEstimator import (
                estimate_sdm_column_params,
                estimate_sdm_lognormal_from_monopore,
            )
            from molass.SEC.Models.SdmOptimizer import optimize_sdm_xr_decomposition
            proxy = _ProxyDecomposition(
                self._xr_x, self._xr_y, self._xr_peaks, self._xr_model, self.peak_rgs
            )
            # Stage 1: multi-start mono-pore column param estimation.
            # Pass the column-specific poresize_bounds from SerialSettings so
            # the library estimator uses the same window as the optimizer
            # (e.g. (71, 81) Å for Superdex 200) instead of the default (70, 300).
            from molass_legacy._MOLASS.SerialSettings import get_setting
            poresize_bounds = get_setting("poresize_bounds")
            mono_env = estimate_sdm_column_params(proxy, poresize_bounds=poresize_bounds)
            editor.pbar["value"] = 1
            editor.update()
            # Stage 2: converged mono-pore SdmComponentCurves
            editor.update_status_bar("SDM lognormal init (2/4): optimizing mono-pore decomposition...")
            mono_ccurves = optimize_sdm_xr_decomposition(proxy, mono_env)
            editor.pbar["value"] = 2
            editor.update()
            # Stage 3: lognormal (mu, sigma, t0, k) refined by moment matching
            editor.update_status_bar("SDM lognormal init (3/4): moment matching to lognormal distribution...")
            ln_env = estimate_sdm_lognormal_from_monopore(
                mono_ccurves, proxy.xr_icurve, decomposition=proxy
            )
            editor.pbar["value"] = 3
            editor.update()
            # Stage 4: converged lognormal NM — mirrors upgrade()'s final optimization step.
            # Lifts init fv from Stage-3 ≈-0.76 (SV≈52) to ≈-1.21 (SV≈72),
            # matching the library notebook's starting point for BH.
            editor.update_status_bar("SDM lognormal init (4/4): refining lognormal parameters; may take more than 10 minutes...")
            from molass.SEC.Models.SdmOptimizer import optimize_sdm_lognormal_xr_decomposition
            ln_pore_sigma_setting = get_setting("sdm_pore_sigma")
            ln_ccurves = optimize_sdm_lognormal_xr_decomposition(proxy, ln_env, ln_pore_sigma=ln_pore_sigma_setting)
            # Extract Stage-4 converged column params (shared across all components)
            N4, T4, _me4, _mp4, x0_4, tI_4, N0_4, mu_4, sigma_4, k_4 = ln_ccurves[0].column.get_params()
            K_lib = N4 * T4   # Legacy K = N*T  (see DispersiveMonopore.py: "T_ = K_/N_")
            _SIGMA_FIXED = sigma_4  # 0.3 — ln_pore_sigma is fixed in Stage 4 by default
            self.logger.info(
                "Library lognormal init (stage4): N=%g, T=%g, K=%g, N0=%g, t0=%g, mu=%g (poresize=%g Å), sigma=%g, k=%g",
                N4, T4, K_lib, N0_4, x0_4, mu_4, np.exp(mu_4), _SIGMA_FIXED, k_4,
            )
            # Update XR weights from Stage-4 converged scales (better BH start)
            xr_scales = np.array([cc.scale for cc in ln_ccurves])
            nc_xr = len(ln_ccurves)
            # Map → G1300 sdmcol_8: [N, K, x0, mu, sigma_fixed, N0, tI, k_gamma]
            sdmcol_8 = np.array([N4, K_lib, x0_4, mu_4, _SIGMA_FIXED, N0_4, tI_4, k_4])
        except Exception as e:
            self.logger.warning(
                "Library lognormal init failed (%s); falling back to legacy rough estimate.", e
            )
            N0 = 50000.0
            mu = np.log(max(float(poresize), 1.0))
            sdmcol_8 = np.array([N, K, x0, mu, 0.3, N0, tI, 2.0])
            xr_scales = None
            nc_xr = 0
        finally:
            editor.pbar.configure(maximum=MAXNUM_STEPS, value=MAXNUM_STEPS,
                                  style='Phase1.Horizontal.TProgressbar')
            editor.update_status_bar("Stochastic initial parameters are ready.")

        non_col = init_params_6[:-6].copy()
        if xr_scales is not None:
            non_col[:nc_xr] = xr_scales  # replace XR weights with Stage-4 scales
        return np.concatenate([non_col, sdmcol_8])

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