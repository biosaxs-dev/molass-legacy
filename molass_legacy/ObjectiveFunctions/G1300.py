"""
    G1300.py — 7-score SDM objective function (lognormal pore + gamma variant)

    Elution model: SDM with lognormal pore distribution and gamma residence times
    Scores: same 7 as G0346
    Extension of G1200: replaces single poresize with lognormal distribution (mu, sigma)

    Parameter layout (split_params_simple order):
        xr_params (nc,)  : XR peak heights
        xr_baseparams    : XR baseline parameters
        rg_params (nc,)  : Rg values per component
        (a, b)           : UV-XR frame mapping
        uv_params (nc,)  : UV/XR intensity ratios ε_i/k (unified architecture)
        uv_baseparams    : UV baseline parameters
        (c, d)           : mappable range
        sdmcol_params    : [N, K, x0, mu, sigma, N0, tI, k]  (lognormal pore)

    Copyright (c) 2026-2026, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
from molass.SEC.Models.LognormalPore import (
    sdm_lognormal_pore_gamma_pdf_fast as elutionmodel_func,
    sdm_lognormal_model_moments,
)
from molass_legacy.Optimizer.BasicOptimizer import BasicOptimizer, PENALTY_SCALE, UV_XR_RATIO_ALLOW, UV_XR_RATIO_SCALE
from molass_legacy.Optimizer.NumericalUtils import safe_ratios
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.ModelParams.SeccolFunctions import rgfit_secconf_eval
from molass_legacy.Optimizer.TheDebugUtils import convert_score_list
from molass_legacy.Optimizer.PenaltyUtils import compute_mapping_penalty

EGH_LOG_ALPHA = np.log(0.1)
TAU_BOUND_RATIO = get_setting("TAU_BOUND_RATIO")    # tau <= sigma*TAU_BOUND_RATIO
LRF_RESIDUAL_FAKED = 10
XR_VALID = 0.001
RG_FITTING_NAN_REPLACE = 100

class G1300(BasicOptimizer):
    """
    Stochastic Dispersive Model — Lognormal pore distribution + Gamma residence times

    Unlike G1100/G1200 (mono-pore), this model does not pre-compute rho = Rg/poresize.
    Instead, Rg is passed directly to the lognormal pore PDF, which integrates over
    the pore size distribution internally.

    sdmcol_params layout: [N, K, x0, mu, sigma, N0, tI, k_gamma]  (8 elements)
    """
    def __init__(self, dsets, n_components, **kwargs):
        self.elutionmodel_func = elutionmodel_func
        self._position_anchor_frames = None   # lazy-initialized on first objective call
        self._position_anchor_scale = None
        if True:
            from importlib import reload
            import molass_legacy.ModelParams.SdmParams
            reload(molass_legacy.ModelParams.SdmParams)
        from molass_legacy.ModelParams.SdmParams import SdmParams

        params_type = SdmParams(n_components, num_col_params=8)
        BasicOptimizer.__init__(self, dsets, n_components, params_type, kwargs)
        self.exports_bounds = True

    def objective_func(self, p, plot=False, debug=False, fig_info=None, axis_info=None, return_full=False, avoid_pinv=False, return_lrf_info=False, **kwargs):
        self.eval_counter += 1
        xr_params, xr_baseparams, rg_params, (a, b), uv_params, uv_baseparams, (c, d), sdmcol_params = self.split_params_simple(p)

        x = self.xr_curve.x
        y = self.xr_curve.y

        N, K, x0, mu, sigma, N0, tI, k_gamma = sdmcol_params
        me = 1.5
        mp = 1.5
        T = K/N
        ty = np.zeros(len(x))

        uv_x = a*x+b
        uv_y = self.uv_curve.spline(uv_x)

        mapping_penalty = compute_mapping_penalty(self.uv_curve, self.xr_curve, self.init_mapping, (a, b), len(self.uv_curve.x), xr_params, uv_params)

        masked_params = p[self.bounds_mask]
        outofbounds_penalty = PENALTY_SCALE * (np.sum(np.max([self.zero_bounds, self.lower_bounds - masked_params], axis=0)) + np.sum(np.max([self.zero_bounds, masked_params - self.upper_bounds], axis=0)))
        if self.eval_counter == 1 and outofbounds_penalty > 0:
            self.logger.info("out of lower bounds: %s", str(np.max([self.zero_bounds, self.lower_bounds - masked_params], axis=0)))
            self.logger.info("out of upper bounds: %s", str(np.max([self.zero_bounds, masked_params - self.upper_bounds], axis=0)))
        if plot:
            overlap = np.zeros(len(x))
            overlap_penalities = []

        xr_cy_list = []
        uv_cy_list = []
        xr_ty = np.zeros(len(x))
        uv_ty = np.zeros(len(uv_x))
        negative_penalty = min(0, T)**2
        T_ = abs(T)
        x_ = x - tI
        t0 = x0 - tI
        for xr_w, rg_, uv_ratio in zip(xr_params, rg_params, uv_params):
            negative_penalty += min(0, xr_w)**2 + min(0, uv_ratio)**2
            # Lognormal pore: pass Rg directly to the PDF (no rho pre-computation)
            pd_cy = elutionmodel_func(x_, 1.0, N, T_, k_gamma, me, mp, mu, sigma, rg_, N0, t0)
            xr_cy = xr_w * pd_cy
            uv_cy = uv_ratio * xr_cy    # unified: ratio × XR curve (Phase 1c)

            xr_ty += xr_cy
            xr_cy_list.append(xr_cy)
            uv_ty += uv_cy
            uv_cy_list.append(uv_cy)

        xr_cy = self.xr_baseline(x, xr_baseparams, xr_ty, xr_cy_list)
        uv_cy = self.uv_baseline(uv_x, uv_baseparams, uv_ty, uv_cy_list)
        xr_ty += xr_cy
        xr_cy_list.append(xr_cy)
        uv_ty += uv_cy
        uv_cy_list.append(uv_cy)

        # Position anchor: lazy-initialize from initial params on first call
        # Prevents component drift across lump boundaries (same mechanism as SdmOptimizer.py).
        # init_params is set by prepare_for_optimization(); fall back to p for standalone calls.
        if self._position_anchor_frames is None:
            init_p = getattr(self, 'init_params', p)
            _, _, rg_init, _, _, _, _, sdmcol_init = self.split_params_simple(init_p)
            N_i, K_i, x0_i, mu_i, sigma_i, N0_i, tI_i, k_i = sdmcol_init
            T_i = K_i / N_i
            t0_i = x0_i - tI_i
            frames = np.array([
                tI_i + sdm_lognormal_model_moments(rg_, N_i, T_i, N0_i, t0_i, k_i, mu_i, sigma_i)[0]
                for rg_ in rg_init
            ], dtype=float)
            self._position_anchor_frames = frames
            initial_error = float(np.sum(self.xr_curve.y ** 2))
            lump_sep = max(float(np.max(frames) - np.min(frames)), 1.0)
            self._position_anchor_scale = initial_error / (lump_sep ** 2)

        # M_1 position penalty — same formula as library SdmOptimizer.py (commit e7c8488)
        positions_ = np.array([
            tI + sdm_lognormal_model_moments(rg_, N, T_, N0, t0, k_gamma, mu, sigma)[0]
            for rg_ in rg_params
        ], dtype=float)
        position_penalty = float(np.sum((positions_ - self._position_anchor_frames) ** 2)
                                 * self._position_anchor_scale)

        lrf_info = None     # initialize before try so plot branch can reference it even if exception occurs (molass-legacy#85)
        penalties = []      # initialize before try so plot branch can reference it if exception occurs before penalties = [...]
        score_list = [0] * self.get_num_scores([])  # initialize before try for same reason
        try:
            lrf_info = self.compute_LRF_matrices(x, y, xr_cy_list, xr_ty, uv_x, uv_y, uv_cy_list, uv_ty, debug=debug)
            if return_lrf_info:
                return lrf_info

            y1, y2 = xr_baseparams[0:2]
            y1_penalty = max(self.y1_allowance, (y1 - self.init_y1)**2) - self.y1_allowance
            y2_penalty = max(self.y2_allowance, (y2 - self.init_y2)**2) - self.y2_allowance
            intercept_penalty = y1_penalty + y2_penalty  # bridge: used in negative_penalty below
            baseline_penalty = y1_penalty*self.y1_penalty_scale + y2_penalty*self.y2_penalty_scale
            negative_penalty = PENALTY_SCALE * (negative_penalty + intercept_penalty)
            order_penalty = 0       # common order_penalty will be added in compute_fv

            penalties = [mapping_penalty, negative_penalty, baseline_penalty, outofbounds_penalty, order_penalty, position_penalty]

            fv, score_list = self.compute_fv(lrf_info, xr_params, rg_params, sdmcol_params, penalties, p, debug=debug)
        except:
            etb = ExceptionTracebacker()
            last_lines = etb.last_lines(n=2)
            if last_lines.find("SVD") > 0:
                if self.svd_error_count == 0:
                    self.logger.warning( "error in objective_func: " + last_lines)
                self.svd_error_count += 1
                svd_error = True
            else:
                self.logger.warning( "error in objective_func: " + last_lines)
                svd_error = False
            lrf_info = self.create_lrf_info_for_debug(x, y, xr_ty, xr_cy_list, uv_x, uv_y, uv_ty, uv_cy_list)
            if return_lrf_info:
                return lrf_info
            # Finite (not inf) penalty so stochastic samplers (NS / UltraNest)
            # can still rank-order failed proposals; np.inf would terminate
            # the run after a single iteration.
            fv = 1e8
            penalties = [0] * 6     # above penalties + [control_penalty]
            score_list = [0] * self.get_num_scores([])      # score_list does not include penalties here

            if svd_error and not avoid_pinv and debug:
                self.objective_func(p, plot=True, fig_info=fig_info, axis_info=axis_info, avoid_pinv=True)
                return fv

        if plot:
            from importlib import reload
            import molass_legacy.ModelParams.SdmPlotUtils
            reload(molass_legacy.ModelParams.SdmPlotUtils)
            from molass_legacy.ModelParams.SdmPlotUtils import plot_objective_state

            print("fv=", fv)

            debug_fv = plot_objective_state((score_list, penalties), fv, None,
                lrf_info,
                overlap, self.rg_curve, rg_params,
                self.get_score_names(),
                fig_info, axis_info,
                self, p,
                avoid_pinv=avoid_pinv,
                **kwargs
                )
            if axis_info is None:
                self.debug_fv = debug_fv

        if return_full:
            score_list = convert_score_list((score_list, penalties))
            return fv, score_list, *lrf_info.matrices
        else:
            return fv

    def get_strategy(self):
        from molass_legacy.Optimizer.Strategies.SdmStrategy import SdmStrategy
        return SdmStrategy(nc=self.n_components - 1)
    
    def is_stochastic(self):
        return True
