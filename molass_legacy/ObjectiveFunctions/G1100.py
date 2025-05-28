"""
    G1100.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
from molass_legacy.Models.Stochastic.DispersivePdf import dispersive_monopore_pdf as elutionmodel_func
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

class G1100(BasicOptimizer):
    """
    Stochastic Dispersive Model
    """
    def __init__(self, dsets, n_components, **kwargs):
        self.elutionmodel_func = elutionmodel_func
        if True:
            from importlib import reload
            import molass_legacy.ModelParams.SdmParams
            reload(molass_legacy.ModelParams.SdmParams)
        from molass_legacy.ModelParams.SdmParams import SdmParams

        params_type = SdmParams(n_components)
        BasicOptimizer.__init__(self, dsets, n_components, params_type, kwargs)
        self.exports_bounds = True

    def objective_func(self, p, plot=False, debug=False, fig_info=None, axis_info=None, return_full=False, avoid_pinv=False, return_lrf_info=False, **kwargs):
        self.eval_counter += 1
        xr_params, xr_baseparams, rg_params, (a, b), uv_params, uv_baseparams, (c, d), sdmcol_params = self.split_params_simple(p)

        x = self.xr_curve.x
        y = self.xr_curve.y
        # rg = self.rg

        N, K, x0, poresize, N0, tI = sdmcol_params
        me = 1.5
        mp = 1.5
        T = K/N
        rho = rg_params/poresize
        rho[rho > 1] = 1
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
        k = 0
        x_ = x - tI
        t0 = x0 - tI
        for xr_w, r_, uv_w in zip(xr_params, rho, uv_params):
            negative_penalty += min(0, xr_w)**2 + min(0, uv_w)**2
            np_ = N*(1 - r_)**me
            tp_ = T_*(1 - r_)**mp
            pd_cy = elutionmodel_func(x_, np_, tp_, N0, t0)
            xr_cy = xr_w * pd_cy
            uv_cy = uv_w * pd_cy

            xr_ty += xr_cy
            xr_cy_list.append(xr_cy)
            uv_ty += uv_cy
            uv_cy_list.append(uv_cy)
            k += 1

        xr_cy = self.xr_baseline(x, xr_baseparams, xr_ty, xr_cy_list)
        uv_cy = self.uv_baseline(uv_x, uv_baseparams, uv_ty, uv_cy_list)
        xr_ty += xr_cy
        xr_cy_list.append(xr_cy)
        uv_ty += uv_cy
        uv_cy_list.append(uv_cy)

        try:
            lrf_info = self.compute_LRF_matrices(x, y, xr_cy_list, xr_ty, uv_x, uv_y, uv_cy_list, uv_ty, debug=debug)
            if return_lrf_info:
                return lrf_info

            m, s = xr_baseparams[0:2]   # how about r?

            slope_penalty = max(self.slope_allowance, (m - self.init_slope)**2) - self.slope_allowance
            intercept_penalty = max(self.intercept_allowance, (s - self.init_intercept)**2) - self.intercept_allowance
            baseline_penalty = slope_penalty*self.slope_penalty_scale + intercept_penalty*self.intercept_penalt_scale
            negative_penalty = PENALTY_SCALE * (negative_penalty + intercept_penalty)
            order_penalty = 0       # common order_penalty will be added in compute_fv

            penalties = [mapping_penalty, negative_penalty, baseline_penalty, outofbounds_penalty, order_penalty]

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
            fv = np.inf
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