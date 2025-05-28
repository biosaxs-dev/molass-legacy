"""
    G2010.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from molass_legacy.Models.RateTheory.EDM import edm_impl
from molass_legacy.Optimizer.BasicOptimizer import BasicOptimizer, PENALTY_SCALE, UV_XR_RATIO_ALLOW, UV_XR_RATIO_SCALE
from molass_legacy.Optimizer.NumericalUtils import safe_ratios
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.ModelParams.SeccolFunctions import rgfit_secconf_eval
from molass_legacy.SecTheory.BoundControl import Penalties
from molass_legacy.Optimizer.TheDebugUtils import convert_score_list
from molass_legacy.Optimizer.PenaltyUtils import compute_mapping_penalty

EGH_LOG_ALPHA = np.log(0.1)
BAD_PARAMS_RETURN = 1e8
IGNORE_OUT_OF_BOUNDS = True

class G2010(BasicOptimizer):
    """
    Equilibrium Dispersive Model
    """
    def __init__(self, dsets, n_components, **kwargs):

        if True:
            from importlib import reload
            import molass_legacy.ModelParams.EdmParams
            reload(molass_legacy.ModelParams.EdmParams)
        from molass_legacy.ModelParams.EdmParams import EdmParams

        params_type = EdmParams(n_components)
        BasicOptimizer.__init__(self, dsets, n_components, params_type, kwargs)
        params_type.set_x(self.xr_curve.x)

    def objective_func(self, p, plot=False, debug=False, fig_info=None, axis_info=None, return_full=False, return_lrf_info=False):
        self.eval_counter += 1
        try:
            xr_params, xr_baseparams, rg_params, (a, b), uv_params, uv_baseparams, (c, d), edm_colparams  = self.split_params_simple(p)
        except ValueError:
            return BAD_PARAMS_RETURN

        x = self.xr_curve.x
        y = self.xr_curve.y
        # rg = self.rg

        uv_x = a*x+b
        uv_y = self.uv_curve.spline(uv_x)

        mapping_penalty = compute_mapping_penalty(self.uv_curve, self.xr_curve, self.init_mapping, (a, b), len(self.uv_curve.x),
                                                  xr_params[:,-1], uv_params)        # xr_params[:,-1] : cinj

        xr_cy_list = []
        uv_cy_list = []
        xr_ty = np.zeros(len(x))
        uv_ty = np.zeros(len(uv_x))
        intercept_penalty = 0
        baseline_penalty = 0
        masked_params = p[self.bounds_mask]
        if IGNORE_OUT_OF_BOUNDS:
            outofbounds_penalty = 0
        else:
            outofbounds_penalty = PENALTY_SCALE * (np.sum(np.max([self.zero_bounds, self.lower_bounds - masked_params], axis=0))
                                                + np.sum(np.max([self.zero_bounds, masked_params - self.upper_bounds], axis=0)) + Penalties[0])
        if (self.eval_counter == 1 or debug) and outofbounds_penalty > 0:
            self.logger.info("out of lower bounds: %s", str(np.max([self.zero_bounds, self.lower_bounds - masked_params], axis=0)))
            self.logger.info("out of upper bounds: %s", str(np.max([self.zero_bounds, masked_params - self.upper_bounds], axis=0)))
        if plot:
            overlap = np.zeros(len(x))
        last_cy = None
        sec_penality = 0

        order_penalty = 0

        Tz = edm_colparams[0]
        k = 0
        for t0, u, a, b, e, Dz, cinj in xr_params:
            xr_cy = edm_impl(x, t0, u, a, b, e, Dz, cinj)
            uv_cy = uv_params[k]*xr_cy

            if k > 0:
                if debug:
                    overlap += np.abs(np.min([last_cy, xr_cy], axis=0))     # np.abs() is intended to degrade negative elements

            last_cy = xr_cy

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
            lrf_info = self.compute_LRF_matrices(x, y, xr_cy_list, xr_ty, uv_x, uv_y, uv_cy_list, uv_ty)
            if return_lrf_info:
                return lrf_info

            m, s = xr_baseparams[0:2]   # how about r?

            slope_penalty = max(self.slope_allowance, (m - self.init_slope)**2) - self.slope_allowance
            intercept_penalty = max(self.intercept_allowance, (s - self.init_intercept)**2) - self.intercept_allowance
            baseline_penalty += slope_penalty*self.slope_penalty_scale + intercept_penalty*self.intercept_penalt_scale
            negative_penalty = PENALTY_SCALE * min(0, np.min(uv_params[:]))**2
            order_penalty *= PENALTY_SCALE
            penalties = [mapping_penalty, negative_penalty, baseline_penalty, outofbounds_penalty, order_penalty]

            fv, score_list = self.compute_fv(lrf_info, xr_params, rg_params, edm_colparams, penalties, p, debug=debug)
        except:
            # e.g., numpy.linalg.LinAlgError: SVD did not converge
            log_exception(self.logger, "error in objective_func", n=5)
            fv = np.inf
            score_list = [0] * self.get_num_scores([])
            matrices = [None] * self.get_num_matrices()
            xr_ty = np.zeros(len(y))

        if plot:
            from molass_legacy.ModelParams.EdmPlotUtils import plot_objective_state
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