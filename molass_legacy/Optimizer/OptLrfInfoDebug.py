"""
    OptLrfInfoDebug.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Optimizer.OptLrfInfo import OptLrfInfo

class OptLrfInfoProxy(OptLrfInfo):
    def __init__(self,
                    qv, xrD, xrE,
                    x, y, xr_ty, xr_cy_list,
                    uv_x, uv_y, uv_ty, uv_cy_list,
                    composite):
        self.matrices = (None,) * 5
        self.qv = qv
        self.xrD = xrD
        self.xrE = xrE
        self.x = x
        self.y = y
        self.xr_ty = xr_ty
        self.scaled_xr_cy_array = np.asarray(xr_cy_list)
        self.uv_x = uv_x
        self.uv_y = uv_y
        self.uv_ty = uv_ty
        self.scaled_uv_cy_array = np.asarray(uv_cy_list)
        self.composite = composite