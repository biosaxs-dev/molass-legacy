"""
    G1400.py — 7-score LKM (Lumped Kinetic Model) objective function

    Elution model: LKM via PDE characteristic-function + FFT inversion
    Parameter layout:
        xr_params, xr_baseparams, rg_params, (a,b), uv_params, uv_baseparams,
        (c,d), lkmcol_params
    lkmcol_params = [Pe, t0, R_0, k_MT_0, R_1, k_MT_1, ..., R_{nc-1}, k_MT_{nc-1}]

    Copyright (c) 2026, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
from molass.SEC.Models.LkmLinear import lkm_pdf
from molass_legacy.Optimizer.BasicOptimizer import BasicOptimizer, PENALTY_SCALE, UV_XR_RATIO_ALLOW, UV_XR_RATIO_SCALE
from molass_legacy.Optimizer.NumericalUtils import safe_ratios
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Optimizer.TheDebugUtils import convert_score_list
from molass_legacy.Optimizer.PenaltyUtils import compute_mapping_penalty

LRF_RESIDUAL_FAKED = 10
XR_VALID = 0.001


class G1400(BasicOptimizer):
    """
    Lumped Kinetic Model (LKM) rigorous optimizer.

    lkmcol_params layout: [Pe, t0, R_0, k_MT_0, R_1, k_MT_1, ..., R_{nc-1}, k_MT_{nc-1}]
    """

    def __init__(self, dsets, n_components, **kwargs):
        if True:
            from importlib import reload
            import molass_legacy.ModelParams.LkmParams
            reload(molass_legacy.ModelParams.LkmParams)
        from molass_legacy.ModelParams.LkmParams import LkmParams

        nc = n_components - 1
        num_col_params = 2 + 2 * nc
        params_type = LkmParams(n_components)
        BasicOptimizer.__init__(self, dsets, n_components, params_type, kwargs)
        self.exports_bounds = True

    def objective_func(self, p, plot=False, debug=False, fig_info=None, axis_info=None,
                       return_full=False, avoid_pinv=False, return_lrf_info=False, **kwargs):
        self.eval_counter += 1
        xr_params, xr_baseparams, rg_params, (a, b), uv_params, uv_baseparams, (c, d), lkmcol_params = self.split_params_simple(p)

        x = self.xr_curve.x
        y = self.xr_curve.y

        # LKM column params: shared Pe, t0; per-component R_i, k_MT_i
        Pe = lkmcol_params[0]
        t0 = lkmcol_params[1]
        nc = self.n_components - 1

        uv_x = a * x + b
        uv_y = self.uv_curve.spline(uv_x)

        mapping_penalty = compute_mapping_penalty(
            self.uv_curve, self.xr_curve, self.init_mapping,
            (a, b), len(self.uv_curve.x), xr_params, uv_params)

        masked_params = p[self.bounds_mask]
        outofbounds_penalty = PENALTY_SCALE * (
            np.sum(np.max([self.zero_bounds, self.lower_bounds - masked_params], axis=0)) +
            np.sum(np.max([self.zero_bounds, masked_params - self.upper_bounds], axis=0)))
        if self.eval_counter == 1 and outofbounds_penalty > 0:
            self.logger.info("out of lower bounds: %s",
                             str(np.max([self.zero_bounds, self.lower_bounds - masked_params], axis=0)))
            self.logger.info("out of upper bounds: %s",
                             str(np.max([self.zero_bounds, masked_params - self.upper_bounds], axis=0)))

        xr_cy_list = []
        uv_cy_list = []
        xr_ty = np.zeros(len(x))
        uv_ty = np.zeros(len(uv_x))
        negative_penalty = 0.0

        for i, (xr_w, rg_, uv_w) in enumerate(zip(xr_params, rg_params, uv_params)):
            negative_penalty += min(0, xr_w) ** 2 + min(0, uv_w) ** 2
            R_i    = lkmcol_params[2 + 2 * i]
            k_MT_i = lkmcol_params[2 + 2 * i + 1]
            # LKM: x is absolute frame axis; no tI subtraction needed
            pd_cy  = lkm_pdf(x, Pe, t0, k_MT_i, R_i)
            xr_cy  = xr_w * pd_cy
            uv_cy  = uv_w * pd_cy
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

        try:
            lrf_info = self.compute_LRF_matrices(
                x, y, xr_cy_list, xr_ty, uv_x, uv_y, uv_cy_list, uv_ty, debug=debug)
            if return_lrf_info:
                return lrf_info

            y1, y2 = xr_baseparams[0:2]
            y1_penalty = max(self.y1_allowance, (y1 - self.init_y1) ** 2) - self.y1_allowance
            y2_penalty = max(self.y2_allowance, (y2 - self.init_y2) ** 2) - self.y2_allowance
            intercept_penalty = y1_penalty + y2_penalty
            baseline_penalty  = (y1_penalty * self.y1_penalty_scale +
                                  y2_penalty * self.y2_penalty_scale)
            negative_penalty  = PENALTY_SCALE * (negative_penalty + intercept_penalty)
            order_penalty = 0

            penalties = [mapping_penalty, negative_penalty, baseline_penalty,
                         outofbounds_penalty, order_penalty]

            fv, score_list = self.compute_fv(
                lrf_info, xr_params, rg_params, lkmcol_params, penalties, p, debug=debug)

        except:
            etb = ExceptionTracebacker()
            last_lines = etb.last_lines(n=2)
            if last_lines.find("SVD") > 0:
                if self.svd_error_count == 0:
                    self.logger.warning("error in objective_func: " + last_lines)
                self.svd_error_count += 1
                svd_error = True
            else:
                self.logger.warning("error in objective_func: " + last_lines)
                svd_error = False
            lrf_info = self.create_lrf_info_for_debug(
                x, y, xr_ty, xr_cy_list, uv_x, uv_y, uv_ty, uv_cy_list)
            if return_lrf_info:
                return lrf_info
            fv = 1e8
            penalties = [0] * 6
            score_list = [0] * self.get_num_scores([])

            if svd_error and not avoid_pinv and debug:
                self.objective_func(p, plot=True, fig_info=fig_info, axis_info=axis_info,
                                    avoid_pinv=True)
                return fv

        if plot:
            if True:
                from importlib import reload
                import molass_legacy.ModelParams.SdmPlotUtils
                reload(molass_legacy.ModelParams.SdmPlotUtils)
            from molass_legacy.ModelParams.SdmPlotUtils import plot_objective_state
            overlap = np.zeros(len(x))
            print("fv=", fv)
            debug_fv = plot_objective_state(
                (score_list, penalties), fv, None,
                lrf_info, overlap, self.rg_curve, rg_params,
                self.get_score_names(), fig_info, axis_info,
                self, p, avoid_pinv=avoid_pinv, **kwargs)
            if axis_info is None:
                self.debug_fv = debug_fv

        if return_full:
            score_list = convert_score_list((score_list, penalties))
            return fv, score_list, *lrf_info.matrices
        else:
            return fv

    def get_strategy(self):
        if True:
            from importlib import reload
            import molass_legacy.Optimizer.Strategies.BasicStrategy
            reload(molass_legacy.Optimizer.Strategies.BasicStrategy)
        from molass_legacy.Optimizer.Strategies.BasicStrategy import BasicStrategy
        return BasicStrategy()

    def is_stochastic(self):
        return True
