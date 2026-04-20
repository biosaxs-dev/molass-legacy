"""
    G0367.py — 4-score variant of G0346 (EGH)

    Uses only: XR_2D_fitting, XR_LRF_residual, UV_2D_fitting, UV_LRF_residual
    Drops: Guinier_deviation, Kratky_smoothness, SEC_conformance

    Created for score-reduction experiment (April 2026).

    Copyright (c) 2021-2026, SAXS Team, KEK-PF
"""
import numpy as np
from .G0346 import G0346
from molass_legacy.Optimizer.BasicOptimizer import (
    WEAK_PENALTY_SCALE, PENALTY_SCALE,
    USE_NORMALIZED_RMSD,
)

if USE_NORMALIZED_RMSD:
    from molass_legacy.Distance.NormalizedRmsd import normalized_rmsd

SUPERIOR_2D_LRF_ALLOW = 0.1
NUM_SCORES_G0367 = 4
VALUE_WEIGHTS4 = np.array([0.25, 0.25, 0.25, 0.25])

MEAN_WEIGHT = 0.8
STDEV_WEIGHT = 0.2


def synthesize4(values, positive_elevate=0):
    """Synthesize fv from 4 equally-weighted scores."""
    values = np.asarray(values) + positive_elevate
    return (MEAN_WEIGHT * np.sqrt(np.sum(VALUE_WEIGHTS4 * values**2))
            + STDEV_WEIGHT * np.std(values)
            - positive_elevate)


class G0367(G0346):
    """
    EGH elution model — 4-score variant (pure fitting quality only).

    Inherits peak model, parameter handling, and objective_func from G0346.
    Overrides only compute_fv to drop Guinier, Kratky, and SEC scores.
    """

    def __init__(self, dsets, n_components, **kwargs):
        super().__init__(dsets, n_components, **kwargs)
        self.NUM_MAJOR_SCORES = NUM_SCORES_G0367
        # Fix plot_scores() which reads the global setting instead of self.NUM_MAJOR_SCORES
        from molass_legacy._MOLASS.SerialSettings import set_setting
        set_setting("NUM_MAJOR_SCORES", NUM_SCORES_G0367)

    def compute_fv(self, lrf_info, xr_params, rg_params, seccol_params, penalties, p, debug=False):
        Pxr, Cxr, Puv, Cuv, mapped_UvD = lrf_info.matrices

        # --- 4 scores only ---
        XR_2D_fitting = normalized_rmsd(lrf_info.xr_ty, lrf_info.y, adjust=self.xr2d_adjust)
        UV_2D_fitting = normalized_rmsd(lrf_info.uv_ty, lrf_info.uv_y, adjust=self.uv2d_adjust)
        XR_LRF_residual = np.log10(np.linalg.norm(self.W_ * (Pxr @ Cxr - self.xrD_)) / self.xr_norm2)
        UV_LRF_residual = np.log10(np.linalg.norm(Puv @ Cuv - mapped_UvD) / self.uv_norm2)

        score_list = [XR_2D_fitting, XR_LRF_residual, UV_2D_fitting, UV_LRF_residual]

        # --- penalties (same as G0346 minus Rg/Kratky/SEC-related) ---

        # negative penalty for P matrices
        for P in [Pxr, Puv]:
            P_ = P[:, :-1]
            penalties[1] += WEAK_PENALTY_SCALE * np.linalg.norm(P_[P_ < 0])

        # Rg order penalty (keep — cheap guard against permutation)
        valid_rg_params = rg_params[self.valid_components]
        if len(valid_rg_params) > 1:
            penalties[4] += PENALTY_SCALE * np.sum(
                np.min([self.zeros_valid_rg, valid_rg_params[:-1] - valid_rg_params[1:]], axis=0) ** 2
            )

        # control penalty: 2D should not be much worse than LRF
        control_penalty = WEAK_PENALTY_SCALE * (
            max(0, XR_2D_fitting - XR_LRF_residual - SUPERIOR_2D_LRF_ALLOW) ** 2
            + max(0, UV_2D_fitting - UV_LRF_residual - SUPERIOR_2D_LRF_ALLOW) ** 2
        )
        penalties.append(control_penalty)

        # UV/XR consistency penalty
        penalties.append(lrf_info.consistency_penalty)

        # --- synthesize ---
        score_array = np.array(score_list)
        fv = synthesize4(score_array, positive_elevate=3) + np.sum(penalties)

        if np.isnan(fv):
            if not self.isnan_logged:
                self.logger.info("fv is NaN: score_array=%s", str(score_array))
                self.isnan_logged = True
            fv = np.inf

        return fv, score_array

    def get_score_names(self, major_only=False):
        names = [
            "XR_2D_fitting", "XR_LRF_residual", "UV_2D_fitting", "UV_LRF_residual",
            "mapping_penalty", "negative_penalty", "baseline_penalty",
            "outofbounds_penalty", "order_penalty", "control_penalty",
            "consistency_penalty",
        ]
        if major_only:
            return names[0:NUM_SCORES_G0367]
        return names

    def get_num_scores(self, penalties):
        return NUM_SCORES_G0367 + len(penalties)
