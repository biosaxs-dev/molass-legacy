"""
    LanJorgensonEgh.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, root
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Peaks.ElutionModels import egh, e1, e2
from LPM import get_corrected
from .BoundControl import Penalties
from .ColumnConstants import Ti_LOWER
import molass_legacy.KekLib.DebugPlot as plt

ESTIMATE_INJECTION_TIME = True

def _pre_optimize_params(ecurve, xr_params, seccol_params, debug=False):
    x = ecurve.x
    y_ = ecurve.y

    # use corrected y
    y = get_corrected(y_)

    # Npc, rp, tI, t0, P, m
    Npc, tI = seccol_params[[0,2]]

    n = xr_params.shape[0]
    temp_xr_params = np.zeros((n,3))

    for k, (h, m, t) in enumerate(xr_params[:,[0,1,3]]):
        temp_xr_params[k,0] = h
        temp_xr_params[k,1] = m
        temp_xr_params[k,2] = t

    def objective(p):
        N_, T_ = p
        sqN = np.sqrt(N_)
        my = np.zeros(len(x))
        for k, (h, m, t) in enumerate(temp_xr_params):
            s = compute_sigma(T_, N_, m, t, init_sigma=xr_params[k,2])
            my += egh(x, h, m, s, t)
        return np.sum((my - y)**2)

    ret = minimize(objective, (Npc, tI))

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt

        with plt.Dp():
            fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(18,5))
            fig.suptitle("Lan-Jorgenson Adaption Proof", fontsize=20)
            ax0.set_title("Free EGH Decomposition", fontsize=16)
            ax1.set_title("Lan-Jorgenson Simple Conversion", fontsize=16)
            ax2.set_title("Lan-Jorgenson Optimized Adaption", fontsize=16)

            ax0.plot(x, y)
            for k, (h, m, s, t) in enumerate(xr_params):
                ax0.plot(x, egh(x, h, m, s, t), ":")

            def plot_params(ax, tI, Npc):
                ax.plot(x, y)
                for k, (h, m, t) in enumerate(temp_xr_params):
                    s = compute_sigma(tI, Npc, m, t, init_sigma=xr_params[k,2])
                    print([k], h, m, s, t)
                    ax.plot(x, egh(x, h, m, s, t), ":")
            plot_params(ax1, tI, Npc)
            plot_params(ax2, *ret.x)

            fig.tight_layout()
            plt.show()

    return temp_xr_params, ret.x

def lj_estimate_initial_params(ecurve, init_xr_params, seccol_params, debug=True):
    if debug:
        print("lj_estimate_initial_params")
        x = ecurve.x
        y = ecurve.y
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("lj_estimate_initial_params")
            ax.plot(x, y)
            cy_list = []
            for h, m, s, t in init_xr_params:
                cy = egh(x, h, m, s, t)
                cy_list.append(cy)
                ax.plot(x, cy, ":")
            ty = np.sum(cy_list, axis=0)
            ax.plot(x, ty, ":", color="red")
            fig.tight_layout()
            plt.show()

    temp_xr_params, (N_, T_) = _pre_optimize_params(ecurve, init_xr_params, seccol_params)
    lj_seccol_params = seccol_params.copy()
    lj_seccol_params[[0,2]] = N_, T_
    return temp_xr_params, lj_seccol_params

def lj_estimate_better(caller):
    print("lj_estimate_better")
    init_params_save = caller.init_params.copy()

    init_params = caller.compute_init_params()

    print("init_params_save=", init_params_save)
    print("init_params=", init_params)

    return

def compute_tau(tI, Np, mu, sigma):

    def equation(x):
        tau = x[0]
        tau_ = abs(tau)
        th = np.arctan2(tau_, sigma)
        M1 = mu + tau*e1(th) - tI
        M2 = (sigma**2 + sigma*tau_ + tau**2)*e2(th)
        return [M1**2 - M2*Np]

    ret = root(equation, [0])
    if ret.success:
        return ret.x[0]
    else:
        Penalties[0] += abs(equation(ret.x)[0])
        return 0

def compute_sigma(tI, Np, mu, tau, init_sigma=10):

    tau_ = abs(tau)

    def equation(x):
        sigma = x[0]
        th = np.arctan2(tau_, sigma)
        M1 = mu + tau*e1(th) - tI
        M2 = (sigma**2 + sigma*tau_ + tau**2)*e2(th)
        return [M1**2 - M2*Np]

    ret = root(equation, [init_sigma])

    return ret.x[0]

def convert_to_xr_params_lj(c_xr_params, tI, Np):
    nc = c_xr_params.shape[0]
    xr_params = np.zeros((nc, 4))
    sqrtNp = np.sqrt(Np)
    Penalties[0] = 0
    for k, (h, m, t) in enumerate(c_xr_params):
        xr_params[k,0] = h
        xr_params[k,1] = m
        init_sigma = (m - tI)/sqrtNp
        xr_params[k,2] = compute_sigma(tI, Np, m, t, init_sigma=init_sigma)
        xr_params[k,3] = t
    return xr_params

if __name__ == '__main__':
    pass
