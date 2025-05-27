"""
    FoleyDorseyEmg.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, root
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Peaks.ElutionModels import egh, emg
from LPM import get_corrected
from .BoundControl import Penalties
from .ColumnConstants import Ti_LOWER

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
            s = compute_sigma(T_, N_, m, t)
            my += emg(x, h, m, s, t)
        return np.sum((my - y)**2)

    ret = minimize(objective, (Npc, tI))

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        print("tI=", tI)

        with plt.Dp():
            fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(18,5))
            fig.suptitle("Foley-Dorsey Adaption Proof", fontsize=20)
            ax0.set_title("Free EGH Decomposition", fontsize=16)
            ax1.set_title("Foley-Dorsey Simple Conversion", fontsize=16)
            ax2.set_title("Foley-Dorsey Optimized Adaption", fontsize=16)

            ax0.plot(x, y)
            for k, (h, m, s, t) in enumerate(xr_params):
                ax0.plot(x, egh(x, h, m, s, t), ":")

            def plot_params(ax, tI, Npc):
                ax.plot(x, y)
                for k, (h, m, t) in enumerate(temp_xr_params):
                    s = compute_sigma(tI, Npc, m, t)
                    print([k], h, m, s, t)
                    ax.plot(x, emg(x, h, m, s, t), ":")
            plot_params(ax1, tI, Npc)
            plot_params(ax2, *ret.x)

            fig.tight_layout()
            plt.show()

    return temp_xr_params, ret.x

def fd_estimate_initial_params(ecurve, init_xr_params, seccol_params, debug=False):
    temp_xr_params, (N_, T_) = _pre_optimize_params(ecurve, init_xr_params, seccol_params, debug=debug)
    fd_seccol_params = seccol_params.copy()
    fd_seccol_params[[0,2]] = N_, T_
    return temp_xr_params, fd_seccol_params

def compute_tau_simple(tI, Npc, mu, sigma):
    t = (mu - tI)**2/Npc - sigma**2
    return np.sqrt(t) if t > 0 else 0

def compute_tau(tI, Npc, mu, sigma):
    """
    from sympy.abc import m, s, t, N
    solve([(m+t)**2 - N*(s**2+t**2)], [t])
    [(m/(N - 1) - sqrt(-N*(N*s**2 - m**2 - s**2))/(1 - N),),
     (m/(N - 1) + sqrt(-N*(N*s**2 - m**2 - s**2))/(1 - N),)]
    """
    m = mu - tI
    t2 = Npc*(m**2 + (1 - Npc)*sigma**2)

    if np.isscalar(t2):
        if t2 >= 0:
            t_ = np.sqrt(t2)
        else:
            if fronting_ok:
                t_ = np.sqrt(-t2)
            else:
                t_ = 0
                Penalties[0] += abs(t2)
    else:
        # vectors will be used in ModelParams/FdEmgParams.py
        if np.all(t2 >= 0):
            t_ = np.sqrt(t2)
        else:
            t_ = np.zeros(len(t2))
            Penalties[0] = np.sum(np.abs(np.min([t_, t2], axis=0)))
            t_[t2 > 0] = np.sqrt(t2[t2 > 0])
    return (m + t_)/(Npc - 1)

def compute_sigma(tI, Npc, mu, tau):
    return np.sqrt((mu + tau - tI)**2/Npc - tau**2)

def convert_to_xr_params_fd(c_xr_params, tI, Npc):
    nc = c_xr_params.shape[0]
    xr_params = np.zeros((nc, 4))
    sqrtNp = np.sqrt(Npc)
    Penalties[0] = 0
    for k, (h, m, t) in enumerate(c_xr_params):
        xr_params[k,0] = h
        xr_params[k,1] = m
        xr_params[k,2] = compute_sigma(tI, Npc, m, t)
        xr_params[k,3] = t
    return xr_params

def spike_demo():
    tI = -2000

    x = np.linspace(0, 500, 501)

    t0 = 50
    mu = 200
    sigma = 30
    tau = 40
    y = emg(x, 1, mu, sigma, tau)

    tr = mu + tau
    N = (tr - tI)**2/(sigma**2 + tau**2)
    print("tI=", tI)
    print("N=", N)

    tau_ = compute_tau(tI, N, mu, sigma)
    print("tau_=", tau_)    # tau == 10

    tau_ = compute_tau(np.array([tI, tI]), np.array([N, N]), np.array([mu, mu]), np.array([sigma, sigma]))
    print("tau_=", tau_)    # tau == 10

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(tr, emg(np.array([tr]), 1, mu, sigma, tau), "o", color="yellow")
        ax.plot(mu, emg(np.array([mu]), 1, mu, sigma, tau), "o", color="pink")
        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    spike_demo()
