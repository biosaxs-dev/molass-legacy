"""
    G2020.py — 7-score Constrained-EDM objective function

    Elution model: Constrained EDM (CEDM)
    Column parameters t0, u, e, Dz are SHARED across all components.
    Per-component free parameters: a (K_SEC), b, cinj.

    Scores: XR_2D_fitting, XR_LRF_residual, UV_2D_fitting, UV_LRF_residual,
            Guinier_deviation, Kratky_smoothness, SEC_conformance

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from molass_legacy.Models.RateTheory.EDM import edm_impl
from molass_legacy.Optimizer.BasicOptimizer import BasicOptimizer, PENALTY_SCALE, UV_XR_RATIO_ALLOW, UV_XR_RATIO_SCALE
from molass_legacy.Optimizer.NumericalUtils import safe_ratios
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.Optimizer.TheDebugUtils import convert_score_list
from molass_legacy.Optimizer.PenaltyUtils import compute_mapping_penalty

BAD_PARAMS_RETURN = 1e8
IGNORE_OUT_OF_BOUNDS = True


class G2020(BasicOptimizer):
    """
    Constrained Equilibrium Dispersive Model (CEDM).

    Four column parameters (t0, u, e, Dz) are shared across all
    components; only a (K_SEC), b, and cinj are free per component.
    """

    def __init__(self, dsets, n_components, **kwargs):
        if True:
            from importlib import reload
            import molass_legacy.ModelParams.CedmParams
            reload(molass_legacy.ModelParams.CedmParams)
        from molass_legacy.ModelParams.CedmParams import CedmParams

        params_type = CedmParams(n_components)
        BasicOptimizer.__init__(self, dsets, n_components, params_type, kwargs)
        (xr_curve_for_x, _), _, _ = dsets
        params_type.set_x(xr_curve_for_x.x)

    # ------------------------------------------------------------------
    # objective function
    # ------------------------------------------------------------------

    def objective_func(self, p, plot=False, debug=False,
                       fig_info=None, axis_info=None,
                       return_full=False, return_lrf_info=False):
        self.eval_counter += 1
        try:
            (xr_params_abc, xr_baseparams, rg_params,
             (map_a, map_b), uv_params, uv_baseparams,
             (c, d), cedm_colparams) = self.split_params_simple(p)
        except ValueError:
            return BAD_PARAMS_RETURN

        t0_sh, u_sh, e_sh, Dz_sh = cedm_colparams

        x = self.xr_curve.x
        y = self.xr_curve.y

        uv_x = map_a * x + map_b
        uv_y = self.uv_curve.spline(uv_x)

        # mapping_penalty: pass cinj (last column of xr_params_abc) as heights
        mapping_penalty = compute_mapping_penalty(
            self.uv_curve, self.xr_curve, self.init_mapping,
            (map_a, map_b), len(self.uv_curve.x),
            xr_params_abc[:, -1],   # cinj
            uv_params
        )

        xr_cy_list = []
        uv_cy_list = []
        xr_ty = np.zeros(len(x))
        uv_ty = np.zeros(len(uv_x))
        baseline_penalty = 0

        masked_params = p[self.bounds_mask]
        if IGNORE_OUT_OF_BOUNDS:
            outofbounds_penalty = 0
        else:
            from molass_legacy.SecTheory.BoundControl import Penalties
            outofbounds_penalty = PENALTY_SCALE * (
                np.sum(np.max([self.zero_bounds, self.lower_bounds - masked_params], axis=0))
                + np.sum(np.max([self.zero_bounds, masked_params - self.upper_bounds], axis=0))
                + Penalties[0]
            )

        order_penalty = 0

        for a_k, b_k, cinj_k in xr_params_abc:
            full = np.array([t0_sh, u_sh, a_k, b_k, e_sh, Dz_sh, cinj_k])
            xr_cy = np.nan_to_num(
                edm_impl(x, *full), nan=0.0, posinf=0.0, neginf=0.0
            )
            uv_cy = uv_params[len(xr_cy_list)] * xr_cy
            xr_ty += xr_cy
            xr_cy_list.append(xr_cy)
            uv_ty += uv_cy
            uv_cy_list.append(uv_cy)

        xr_bl = self.xr_baseline(x, xr_baseparams, xr_ty, xr_cy_list)
        uv_bl = self.uv_baseline(uv_x, uv_baseparams, uv_ty, uv_cy_list)
        xr_ty += xr_bl
        xr_cy_list.append(xr_bl)
        uv_ty += uv_bl
        uv_cy_list.append(uv_bl)

        lrf_info = None
        penalties = []
        try:
            lrf_info = self.compute_LRF_matrices(
                x, y, xr_cy_list, xr_ty,
                uv_x, uv_y, uv_cy_list, uv_ty
            )
            if return_lrf_info:
                return lrf_info

            y1, y2 = xr_baseparams[0:2]
            y1_penalty = max(self.y1_allowance, (y1 - self.init_y1) ** 2) - self.y1_allowance
            y2_penalty = max(self.y2_allowance, (y2 - self.init_y2) ** 2) - self.y2_allowance
            baseline_penalty += y1_penalty * self.y1_penalty_scale + y2_penalty * self.y2_penalty_scale

            negative_penalty = PENALTY_SCALE * min(0, np.min(uv_params[:])) ** 2
            order_penalty *= PENALTY_SCALE
            penalties = [
                mapping_penalty, negative_penalty, baseline_penalty,
                outofbounds_penalty, order_penalty
            ]

            fv, score_list = self.compute_fv(
                lrf_info, xr_params_abc, rg_params, cedm_colparams,
                penalties, p, debug=debug
            )
        except Exception:
            log_exception(self.logger, "error in G2020.objective_func", n=5)
            fv = np.inf
            score_list = [0] * self.get_num_scores([])
            xr_ty = np.zeros(len(y))

        if plot and lrf_info is not None:
            from molass_legacy.ModelParams.EdmPlotUtils import plot_objective_state
            debug_fv = plot_objective_state((score_list, penalties), fv, None,
                lrf_info,
                None, self.rg_curve, rg_params,
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
