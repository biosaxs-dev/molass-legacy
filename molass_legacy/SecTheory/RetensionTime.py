"""
    SecTheory.RetensionTime.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, root

def compute_retention_time(x, rgs):
    model_trs = []
    for rg in rgs:
        # x[0] : t0
        # x[1] : P
        # x[2] : rp
        # x[3] : m
        rho = 1 if rg > x[2] else rg/x[2]
        model_trs.append(x[0] + x[1]*(1 - rho)**x[3])
    return np.array(model_trs)

def compute_rg_from_retention_time(seccol_params, trs, init_rgs):
    t0, K, rp, m = seccol_params
    model_rg = []
    for i, tr in enumerate(trs):

        def fun(z):
            rho = 1 if z > rp else z/rp
            return t0 + K*(1 - rho)**m - tr

        sol = root(fun, [init_rgs[i]])
        model_rg.append(sol.x[0])

    return np.array(model_rg)

def make_initial_guess(trs, t0_upper_bound=0):
    Kinit = np.average(trs)*2
    bounds = ((-Kinit, +Kinit), (Kinit*0.2, Kinit*3), (50, 300), (1, 6))
    t0_init = t0_upper_bound - 30
    return (t0_init, Kinit, 80, 3), bounds

NEGATIVE_PENALTY_SCALE = 1e5
TZERO_PENALTY_SCALE = 1e5

def estimate_conformance_params(rgs, trs, rg_curve, t0_upper_bound=None, debug=False):

    if t0_upper_bound is None:
        from molass_legacy._MOLASS.SerialSettings import get_setting
        t0_upper_bound = get_setting("t0_upper_bound")

    min_rg = 0
    lim_tr = rg_curve.x[-1]
    zeros_rgs = np.zeros(len(rgs))

    def objective_func(z):
        # max_tr = compute_retention_time(z, [min_rg])        # i.e., max_tr = t0 + K
        max_tr = z[0] + z[1]    # i.e., max_tr = t0 + K
        t0_penalty = TZERO_PENALTY_SCALE * max(0, z[0] - t0_upper_bound)**2
        model_trs = compute_retention_time(z, rgs)
        max_tr_constraint = min(0, max_tr - lim_tr)**2      # so that max_tr > lim_tr
        return np.sqrt(np.average((model_trs - trs)**2)) + max_tr_constraint + t0_penalty

    initial_guess, bounds = make_initial_guess(trs, t0_upper_bound)

    # method = "Nelder-Mead"
    method = None
    result = minimize(objective_func, initial_guess, method=method, bounds=bounds)

    if debug:
        print("---------- len(rgs)=", len(rgs))
        print("---------- initial_guess=", initial_guess)
        print("---------- bounds=", bounds)
        print("---------- result.x=", result.x)

    return result

def estimate_conformance_params_fixed_poreexponent(rgs, trs, degree):

    def objective_func(x):
        x_ = np.concatenate([x, [degree]])
        model_trs = compute_retention_time(x_, rgs)
        return np.sqrt(np.average((model_trs - trs)**2))

    initial_guess, bounds = make_initial_guess(trs)

    # method = "Nelder-Mead"
    method = None
    result = minimize(objective_func, initial_guess[0:3], method=method, bounds=bounds[0:3])

    return result

def estimate_init_rgs(rg_curve, trs, t0_upper_bound, debug=False):
    x_, y_, rg_ = rg_curve.get_valid_curves()

    result = estimate_conformance_params(rg_, x_, rg_curve, t0_upper_bound)
    seccol_params = result.x

    mean = np.mean(rg_)
    std = np.std(rg_)
    init_rgs = np.linspace(mean+std, mean-std, len(trs))
    if debug:
        print("trs=", trs)
        print("init_rgs=", init_rgs)

    model_rgs = compute_rg_from_retention_time(seccol_params, trs, init_rgs)

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        from DataUtils import get_in_folder
        from scipy.interpolate import UnivariateSpline
        print("seccol_params=", seccol_params)

        in_foler = get_in_folder()
        x = rg_curve.x
        y = rg_curve.y
        t0 = seccol_params[0]
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("Initial Rg Estimation for %s" % in_foler, fontsize=16)
            axt = ax.twinx()
            axt.grid(False)
            if x is None:
                ax.plot(x_, y_)
            else:
                ax.plot(x, y)
            k = 0
            for xs, ys, rs in rg_curve.get_curve_segments():
                label = "observed rg" if k == 0 else None
                axt.plot(xs, rs, color="C1", label=label)
                k += 1
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            t = t0_upper_bound
            ax.plot([t, t], [ymin, ymax], ":", color="red", label="t0_upper_bound")

            spline = UnivariateSpline(x_, rg_, s=0, ext=3)
            xm = x[x > t0]
            rs = spline(xm)
            rm = compute_rg_from_retention_time(seccol_params, xm, rs)
            axt.plot(xm, rm, ":", color="green", label="estimated model rg curve")
            axt.plot(trs, model_rgs, "o", color="yellow", label="estimated initial rg's")

            ax.legend(loc="upper left")
            axt.legend(loc="upper right")
            fig.tight_layout()
            plt.show()

    return model_rgs, seccol_params
