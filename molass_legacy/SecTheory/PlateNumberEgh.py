"""
    PlateNumberEgh.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Peaks.ElutionModels import egh
from .ColumnConstants import INJECTION_TIME

Ti = INJECTION_TIME
Np = get_setting("num_plates_pc")

def _pre_optimize_params(ecurve, xr_params, debug=False):
    x = ecurve.x
    y = ecurve.y

    n = xr_params.shape[0]
    temp_xr_params = np.zeros((n,3))

    for k, (h, m) in enumerate(xr_params[:,[0,1]]):
        temp_xr_params[k,0] = h
        temp_xr_params[k,1] = m
        temp_xr_params[k,2] = 0

    def objective(p):
        T_, N_ = p
        sqN = np.sqrt(N_)
        my = np.zeros(len(x))
        for k, (h, m, t) in enumerate(temp_xr_params):
            s = (m - T_)/sqN
            my += egh(x, h, m, s, t)
        return np.sum((my - y)**2)

    ret = minimize(objective, (Ti, Np))

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt

        with plt.Dp():
            fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(18,5))
            fig.suptitle("Plate Number Adaption Proof", fontsize=20)
            ax0.set_title("Free EGH Decomposition", fontsize=16)
            ax1.set_title("Plate Number Simple Conversion", fontsize=16)
            ax2.set_title("Plate Number Optimized Adaption", fontsize=16)

            ax0.plot(x, y)
            for k, (h, m, s, t) in enumerate(xr_params):
                ax0.plot(x, egh(x, h, m, s, t), ":")

            def plot_params(ax, Ti, Np):
                ax.plot(x, y)
                for k, (h, m, t) in enumerate(temp_xr_params):
                    s = (m - Ti)/np.sqrt(Np)
                    print([k], h, m, s, t)
                    ax.plot(x, egh(x, h, m, s, t), ":")
            plot_params(ax1, Ti, Np)
            plot_params(ax2, *ret.x)

            fig.tight_layout()
            plt.show()

    return temp_xr_params, ret.x

def pn_estimate_initial_params(ecurve, init_xr_params, seccol_params):
    temp_xr_params, (T_, N_) = _pre_optimize_params(ecurve, init_xr_params)
    pn_seccol_params = np.concatenate([seccol_params, [T_, N_]])
    return temp_xr_params, pn_seccol_params

def convert_to_xr_params(c_xr_params, Ti, Np):
    nc = c_xr_params.shape[0]
    xr_params = np.zeros((nc, 4))
    xr_params[:,[0,1,3]] = c_xr_params
    sqrtNp = np.sqrt(Np)
    for k in range(nc):
        xr_params[k,2] = (xr_params[k,1] - Ti)/sqrtNp
    return xr_params

if __name__ == '__main__':
    pass
