"""
    Estimators.RtEmgEstimator.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Peaks.ElutionModels import emg
from .EghEstimator import EghEstimator

class RtEmgEstimator(EghEstimator):
    def __init__(self, editor):
        EghEstimator.__init__(self, editor)

    def estimate_params(self, debug=True):
        if debug:
            from importlib import reload
            import SecTheory.HermansEmg
            reload(SecTheory.HermansEmg)
        from SecTheory.HermansEmg import estimate_initial_params, convert_to_xr_params_hermans

        init_xr_params, init_xr_baseparams, temp_rgs, init_mapping, init_uv_heights, init_uv_baseparams, init_mappable_range, seccol_params = self.estimate_egh_params()

        D = self.editor.dsets[0][1]
        temp_xr_params, seccol_params, R = estimate_initial_params(self.ecurves[1], init_xr_params, temp_rgs, seccol_params, D)

        baseline_type = get_setting("unified_baseline_type")
        if baseline_type >= 2:
            # recompute uv_baseparams
            xr_params = convert_to_xr_params_hermans(temp_xr_params, temp_rgs, seccol_params, R)
            uv_curve, xr_curve = self.ecurves
            x = xr_curve.x
            a, b = init_mapping
            uv_x = a*x+b
            uv_y = uv_curve.spline(uv_x)
            uv_ty = np.zeros(len(uv_x))
            k = 0
            for h, m, s, t in xr_params:
                uv_cy = emg(uv_x, init_uv_heights[k], a*m+b, a*s, a*t)
                uv_ty += uv_cy
                k += 1

            uv_base_params = self.editor.get_uv_base_params(xyt=(uv_x, uv_y, uv_ty))
        else:
            uv_base_params = init_uv_baseparams

        return np.concatenate([temp_xr_params.flatten(), init_xr_baseparams, temp_rgs, init_mapping, init_uv_heights, uv_base_params, init_mappable_range, seccol_params, [R]])
