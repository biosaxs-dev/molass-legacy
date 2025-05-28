"""
    G0346.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from molass_legacy.Peaks.ElutionModels import egh
from molass_legacy.Peaks.EghSupples import compute_AB
from molass_legacy.Optimizer.BasicOptimizer import BasicOptimizer, BAD_PARAMS_RETURN, WEAK_PENALTY_SCALE, UV_XR_RATIO_ALLOW, UV_XR_RATIO_SCALE
from molass_legacy.ModelParams.BoundedSecParams import BoundedSecParams
from molass_legacy.ModelParams.EghParams import construct_egh_params_type
from molass_legacy.Optimizer.NumericalUtils import safe_ratios
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.Optimizer.TheDebugUtils import convert_score_list
from molass_legacy.Optimizer.PenaltyUtils import compute_mapping_penalty

PENALTY_SCALE = 1e3
EGH_LOG_ALPHA = np.log(0.1)
TAU_BOUND_RATIO = get_setting("TAU_BOUND_RATIO")    # tau <= sigma*TAU_BOUND_RATIO

class G0346(BasicOptimizer):
    """
    SEC conformance, weighted norm, 0 ≦ tau ≦ sigma*0.65, t0 upper bound
    """
    def __init__(self, dsets, n_components, **kwargs):

        t0_upper_bound = get_setting("t0_upper_bound")
        if t0_upper_bound is None:
            from molass_legacy.SecTheory.T0UpperBound import estimate_t0upper_bound
            ecurve = dsets[0][0]
            t0_upper_bound = estimate_t0upper_bound(ecurve)
            set_setting("t0_upper_bound", t0_upper_bound)

        params_type = construct_egh_params_type(n_components, sec_class=BoundedSecParams, baseline_rg=False)
        debug = kwargs.get('debug', False)
        if debug:
            global BasicOptimizer   # this is required for the use where debug=False
            from importlib import reload
            import molass_legacy.Optimizer.BasicOptimizer
            reload(molass_legacy.Optimizer.BasicOptimizer)
            from molass_legacy.Optimizer.BasicOptimizer import BasicOptimizer
        BasicOptimizer.__init__(self, dsets, n_components, params_type, kwargs)

    def objective_func(self, p, plot=False, debug=False, fig_info=None, axis_info=None, return_full=False, return_lrf_info=False, lrf_debug=False):
        self.eval_counter += 1
        xr_params, xr_baseparams, rg_params, (a, b), uv_params, uv_baseparams, (c, d), seccol_params = self.split_params_simple(p)

        x = self.xr_curve.x
        y = self.xr_curve.y
        # rg = self.rg

        uv_x = a*x+b
        uv_y = self.uv_curve.spline(uv_x)

        mapping_penalty = compute_mapping_penalty(self.uv_curve, self.xr_curve, self.init_mapping, (a, b), len(self.uv_curve.x), xr_params[:,0], uv_params)

        xr_cy_list = []
        uv_cy_list = []
        xr_ty = np.zeros(len(x))
        uv_ty = np.zeros(len(uv_x))
        intercept_penalty = 0
        baseline_penalty = 0
        masked_params = p[self.bounds_mask]
        outofbounds_penalty = PENALTY_SCALE * (np.sum(np.max([self.zero_bounds, self.lower_bounds - masked_params], axis=0)) + np.sum(np.max([self.zero_bounds, masked_params - self.upper_bounds], axis=0)))
        if self.eval_counter == 1 and outofbounds_penalty > 0:
            self.logger.info("out of lower bounds: %s", str(np.max([self.zero_bounds, self.lower_bounds - masked_params], axis=0)))
            self.logger.info("out of upper bounds: %s", str(np.max([self.zero_bounds, masked_params - self.upper_bounds], axis=0)))
        if plot:
            overlap = np.zeros(len(x))
        last_mu = None
        last_cy = None
        last_pa = None
        last_pb = None
        sec_penalty = 0
        order_penalty = 0
        k = 0
        for h, m, s, t in xr_params:
            xr_cy = egh(x, h, m, s, t)
            uv_cy = egh(uv_x, uv_params[k], a*m+b, a*s, a*t)
            A, B = compute_AB(EGH_LOG_ALPHA, s, t)
            pa = m - A
            pb = m + B

            if k > 0:
                order_penalty += max(0, last_mu - m)**2
                if debug:
                    overlap += np.abs(np.min([last_cy, xr_cy], axis=0))     # np.abs() is intended to degrade negative elements

                # first-come-first-leave penalty
                sec_penalty += max(0, last_pa - pa)**2 + max(0, last_pb - pb)**2

            last_pa = pa
            last_pb = pb
            last_mu = m
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
            lrf_info = self.compute_LRF_matrices(x, y, xr_cy_list, xr_ty, uv_x, uv_y, uv_cy_list, uv_ty, debug=lrf_debug)
            if return_lrf_info:
                return lrf_info

            m, s = xr_baseparams[0:2]   # how about r?

            slope_penalty = max(self.slope_allowance, (m - self.init_slope)**2) - self.slope_allowance
            intercept_penalty = max(self.intercept_allowance, (s - self.init_intercept)**2) - self.intercept_allowance
            baseline_penalty += slope_penalty*self.slope_penalty_scale + intercept_penalty*self.intercept_penalt_scale
            if self.avoid_peak_fronting:
                negative_penalty = PENALTY_SCALE * (  min(0, np.min(xr_params[:,0]))**2     # h
                                                    + min(0, np.min(xr_params[:,3]))**2     # tau
                                                    + min(0, np.min(xr_params[:,2]*TAU_BOUND_RATIO - xr_params[:,3]))**2  # sigma*TAU_BOUND_RATIO - tau
                                                    + min(0, np.min(uv_params[:]))**2       # h
                                                    + intercept_penalty)
            else:
                negative_penalty = PENALTY_SCALE * (  min(0, np.min(xr_params[:,0]))**2     # h
                                                    + min(0, np.min(xr_params[:,2]*TAU_BOUND_RATIO - abs(xr_params[:,3])))**2  # sigma*TAU_BOUND_RATIO - tau
                                                    + min(0, np.min(uv_params[:]))**2       # h
                                                    + intercept_penalty)

            order_penalty *= PENALTY_SCALE
            order_penalty += WEAK_PENALTY_SCALE * sec_penalty   # weakly penalized to avoid early stage runaway
            penalties = [mapping_penalty, negative_penalty, baseline_penalty, outofbounds_penalty, order_penalty]

            fv, score_list = self.compute_fv(lrf_info, xr_params, rg_params, seccol_params, penalties, p, debug=debug)
        except:
            # e.g., numpy.linalg.LinAlgError: SVD did not converge
            log_exception(self.logger, "error in objective_func", n=5)
            fv = BAD_PARAMS_RETURN
            score_list = [0] * self.get_num_scores([])
            matrices = [None] * self.get_num_matrices()
            xr_ty = np.zeros(len(y))
            if False:
                debug_ = True
                overlap = np.zeros(len(x))

        if plot:
            from molass_legacy.ModelParams.EghPlotUtils import plot_objective_state
            debug_fv = plot_objective_state((score_list, penalties), fv,
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
