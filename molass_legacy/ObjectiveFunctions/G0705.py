"""
    G0705.py

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
"""
import os
import numpy as np
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from molass_legacy.Peaks.ElutionModels import emg
from molass_legacy.Optimizer.BasicOptimizer import BasicOptimizer, PENALTY_SCALE
from molass_legacy.Optimizer.NumericalUtils import safe_ratios
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.ModelParams.SeccolFunctions import rgfit_secconf_eval
from molass_legacy.SecTheory.ColumnConstants import Ti_LOWER, Ti_UPPER, Np_LOWER, Np_UPPER
from molass_legacy.Optimizer.TheDebugUtils import convert_score_list

UV_B_ALLOW_RATIO = 0.1
EGH_LOG_ALPHA = np.log(0.1)
UV_XR_RATIO_ALLOW = 0.25
BAD_PARAMS_RETURN = 1e8
RATE_R_UPPER_BOUND = get_setting("RATE_R_UPPER_BOUND")      # task: this should be dynamically set properly

class G0705(BasicOptimizer):
    """
    Constrained EGH conformant to Rate Theory (J.J.Hermans)
    """
    def __init__(self, dsets, n_components, **kwargs):
        ecurve = dsets[0][0]

        t0_upper_bound = get_setting("t0_upper_bound")
        if t0_upper_bound is None:
            from SecTheory.T0UpperBound import estimate_t0upper_bound
            t0_upper_bound = estimate_t0upper_bound(ecurve)
            set_setting("t0_upper_bound", t0_upper_bound)

        if True:
            from importlib import reload
            import molass_legacy.ModelParams.RtEmgParams
            reload(molass_legacy.ModelParams.RtEmgParams)
        from molass_legacy.ModelParams.RtEmgParams import RtEmgParams

        params_type = RtEmgParams(n_components)
        BasicOptimizer.__init__(self, dsets, n_components, params_type, kwargs)

    def update_bounds(self, x, debug=False):
        self.sigma_bounds = self.params_type.get_sigma_bounds(x)
        self.logger.info("sigma_bounds have been estimated as %s", str(self.sigma_bounds))
        super().update_bounds(x, debug=debug)

    def objective_func(self, p, plot=False, debug=False, fig_info=None, axis_info=None, return_full=False, return_lrf_info=False):
        self.eval_counter += 1

        try:
            xr_params, xr_baseparams, rg_params, (a, b), uv_params, uv_baseparams, (c, d), seccol_params, R = self.split_params_simple(p)
            # xr_params, xr_baseparams, rg_params, (a, b), uv_params, uv_baseparams, (c, d), seccol_params = self.split_params_simple(p)[0:8]
        except:
            if debug:
                log_exception(self.logger, "objective_func: ")
            return BAD_PARAMS_RETURN

        x = self.xr_curve.x
        y = self.xr_curve.y
        # rg = self.rg

        uv_x = a*x+b
        uv_y = self.uv_curve.spline(uv_x)

        a_init, b_init = self.init_mapping
        ratio = a/a_init
        a_deviation = min(0, ratio - 0.99)**2 + max(0, ratio - 1.05)**2
        b_allowance = len(self.uv_curve.x) * UV_B_ALLOW_RATIO
        b_deviation = max(0, abs(b - b_init) - b_allowance)**2

        uv_xr_ratio = uv_params/xr_params[:,0]
        ratio_deviation = max(0, np.std(uv_xr_ratio)/np.average(uv_xr_ratio) - UV_XR_RATIO_ALLOW)**2
        if debug:
            self.logger.info("uv_xr_ratio=%s, ratio_deviation=%.3g", str(uv_xr_ratio), ratio_deviation)

        mapping_penalty = PENALTY_SCALE * (a_deviation + b_deviation ) + ratio_deviation        # ratio_deviation should not be scaled to PENALTY_SCALE

        xr_cy_list = []
        uv_cy_list = []
        xr_ty = np.zeros(len(x))
        uv_ty = np.zeros(len(uv_x))
        intercept_penalty = 0
        baseline_penalty = 0
        masked_params = p[self.bounds_mask]
        outofbounds_penalty = PENALTY_SCALE * (np.sum(np.max([self.zero_bounds, self.lower_bounds - masked_params], axis=0)) + np.sum(np.max([self.zero_bounds, masked_params - self.upper_bounds], axis=0)))
        outofbounds_penalty += PENALTY_SCALE * ( min(0, R)**2 + min(0, RATE_R_UPPER_BOUND - R)**2)
        if self.eval_counter == 1 and outofbounds_penalty > 0:
            self.logger.info("out of lower bounds: %s", str(np.max([self.zero_bounds, self.lower_bounds - masked_params], axis=0)))
            self.logger.info("out of upper bounds: %s", str(np.max([self.zero_bounds, masked_params - self.upper_bounds], axis=0)))
        if plot:
            overlap = np.zeros(len(x))
        last_mu = None
        last_cy = None
        sec_penality = 0

        order_penalty = 0
        k = 0
        try:
            for h, m, s, t in xr_params:
                xr_cy = emg(x, h, m, s, t)
                uv_cy = emg(uv_x, uv_params[k], a*m+b, a*s, a*t)

                if k > 0:
                    order_penalty += max(0, last_mu - m)**2
                    if debug:
                        overlap += np.abs(np.min([last_cy, xr_cy], axis=0))     # np.abs() is intended to degrade negative elements

                last_mu = m
                last_cy = xr_cy

                xr_ty += xr_cy
                xr_cy_list.append(xr_cy)
                uv_ty += uv_cy
                uv_cy_list.append(uv_cy)
                k += 1
        except:
            # better avoid this case by setting proper bounds
            # xr_ty += xr_cy
            # ValueError: operands could not be broadcast together with shapes (200,) (0,) (200,) 
            file = "bad_params-1.txt"
            if not os.path.exists(file):
                np.savetxt(file, p.reshape((len(p), 1)))
            return BAD_PARAMS_RETURN

        xr_cy = self.xr_baseline(x, xr_baseparams, xr_ty, xr_cy_list)
        uv_cy = self.uv_baseline(uv_x, uv_baseparams, uv_ty, uv_cy_list)
        xr_ty += xr_cy
        xr_cy_list.append(xr_cy)
        uv_ty += uv_cy
        uv_cy_list.append(uv_cy)

        try:
            lrf_info = self.compute_LRF_matrices(x, y, xr_cy_list, xr_ty, uv_x, uv_y, uv_cy_list, uv_ty)
            if return_lrf_info:
                return lrf_info

            m, s = xr_baseparams[0:2]   # how about r?

            slope_penalty = max(self.slope_allowance, (m - self.init_slope)**2) - self.slope_allowance
            intercept_penalty = max(self.intercept_allowance, (s - self.init_intercept)**2) - self.intercept_allowance
            baseline_penalty += slope_penalty*self.slope_penalty_scale + intercept_penalty*self.intercept_penalt_scale
            negative_penalty = PENALTY_SCALE * (  min(0, np.min(xr_params[:,0]))**2     # h
                                                + min(0, np.min(xr_params[:,3]))**2     # tau
                                                + min(0, np.min(uv_params[:]))**2       # h
                                                + min(0, np.min(rg_params))**2          # rg
                                                + intercept_penalty)
            order_penalty *= PENALTY_SCALE
            penalties = [mapping_penalty, negative_penalty, baseline_penalty, outofbounds_penalty, order_penalty]

            fv, score_list = self.compute_fv(lrf_info, xr_params, rg_params, seccol_params, penalties, p, debug=debug)
        except:
            # e.g., numpy.linalg.LinAlgError: SVD did not converge
            log_exception(self.logger, "error in objective_func", n=5)
            fv = np.inf
            score_list = [0] * self.get_num_scores([])
            matrices = [None] * self.get_num_matrices()
            xr_ty = np.zeros(len(y))
            if False:
                debug_ = True
                overlap = np.zeros(len(x))

        if plot:
            from molass_legacy.ModelParams.EmgPlotUtils import plot_objective_state       # EghPlotUtils can be used here because it does not use egh nor emg
            debug_fv = plot_objective_state((score_list, penalties), fv, self.xm,
                lrf_info,
                overlap, self.rg_curve, rg_params,
                self.get_score_names(),
                fig_info, axis_info,
                self, p,
                )
            if axis_info is None:
                self.debug_fv = debug_fv

        if return_full:
            score_list = convert_score_list((score_list, penalties))
            return fv, score_list, *lrf_info.matrices
        else:
            return fv
