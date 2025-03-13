"""
    Models.Stochastic.DispersiveUvScaler.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.Stochastic.DispersivePdf import dispersive_monopore_pdf 
from molass_legacy.Models.Stochastic.DispersiveUtils import NUM_SDMCUV_PARAMS

def adjust_to_uv_scales(x, y, uv_x, uv_y_, sdm_params, rgs, avoid_vanishing=True, debug=False):
    print("adjust_to_uv_scales: xr_init_params=", sdm_params, "rgs=", rgs)
    N, K, x0, poresize, N0, tI = sdm_params[0:NUM_SDMCUV_PARAMS]
    T = K/N
    me = 1.5
    mp = 1.5
    xr_scales = sdm_params[NUM_SDMCUV_PARAMS:]
    rhov = rgs/poresize
    rhov[rhov > 1] = 1
    cy_list = []
    for rho in rhov:
        np_ = N*(1 - rho)**me
        tp_ = T*(1 - rho)**mp
        cy = dispersive_monopore_pdf(x, np_, tp_, N0, x0)   # note that the scale is not applied here
        cy_list.append(cy)

    def uv_scales_objective(scales, return_cy=False):
        uv_cy_list = []
        for cy, scale in zip(cy_list, scales):
            uv_cy = scale * cy
            uv_cy_list.append(uv_cy)
        uv_ty = np.sum(uv_cy_list, axis=0)
        if return_cy:
            return uv_cy_list, uv_ty
        dev = np.sum((uv_ty - uv_y_)**2)
        return dev

    xr2uv_scale = np.max(uv_y_)/np.max(y)   # task: get earlier and keep this scale

    if avoid_vanishing:
        min_scale = np.min(xr_scales)*xr2uv_scale
    else:
        min_scale = 0
    bounds = [(min_scale, 100)] * len(xr_scales)
    res = minimize(uv_scales_objective, xr_scales, method="Nelder-Mead", bounds=bounds)
    uv_cy_list, uv_ty = uv_scales_objective(res.x, return_cy=True)

    if debug:
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
            fig.suptitle("adjust_to_uv_scales")
            ax1.set_title("UV")

            ax1.plot(uv_x, uv_y_, color="blue")
            for k, cy in enumerate(uv_cy_list):
                ax1.plot(uv_x, cy, ":", label='component-%d' % k)
            ax1.plot(uv_x, uv_ty, ":", color="red", label='model total')
            ax1.legend()

            ax2.set_title("XR")
            ax2.plot(x, y, color="orange")
            xr_cy_list = []
            for k, (cy, scale) in enumerate(zip(cy_list, xr_scales)):
                xr_cy = scale * cy
                ax2.plot(x, xr_cy, ":", label='component-%d' % k)
                xr_cy_list.append(xr_cy)
            xr_ty = np.sum(xr_cy_list, axis=0)
            ax2.plot(x, xr_ty, ":", color="red", label='model total')
            ax2.legend()

            fig.tight_layout()
            ret = plt.show()
    else:
        ret = True

    if ret:
        return res.x, uv_ty
    else:
        return