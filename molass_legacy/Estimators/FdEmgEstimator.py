"""
    Estimators.FdEmgEstimator.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Peaks.ElutionModels import emg
from .EghEstimator import EghEstimator

class FdEmgEstimator(EghEstimator):
    def __init__(self, editor):
        EghEstimator.__init__(self, editor)

    def estimate_params(self, debug=False):
        if debug:
            from importlib import reload
            import SecTheory.FoleyDorseyEmg
            reload(SecTheory.FoleyDorseyEmg)
        from SecTheory.FoleyDorseyEmg import fd_estimate_initial_params, convert_to_xr_params_fd

        init_xr_params, init_xr_baseparams, temp_rgs, init_mapping, init_uv_heights, init_uv_baseparams, init_mappable_range, seccol_params = self.estimate_egh_params()

        temp_xr_params, fd_seccol_params = fd_estimate_initial_params(self.ecurves[1], init_xr_params, seccol_params)

        baseline_type = get_setting("unified_baseline_type")
        if baseline_type >= 2:
            # recompute uv_baseparams

            Npc, tI = fd_seccol_params[[0,2]]
            xr_params = convert_to_xr_params_fd(temp_xr_params, tI, Npc)
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

        return np.concatenate([temp_xr_params.flatten(), init_xr_baseparams, temp_rgs, init_mapping, init_uv_heights, uv_base_params, init_mappable_range, fd_seccol_params])
