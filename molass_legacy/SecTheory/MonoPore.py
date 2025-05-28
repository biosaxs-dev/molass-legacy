"""
    SecTheory.MonoPore.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting
from .BasicModels import robust_single_pore_pdf_scaled
import molass_legacy.SecTheory.CumsumInverse
from importlib import reload
reload(molass_legacy.SecTheory.CumsumInverse)
import molass_legacy.SecTheory.LocalOptimizer
reload(molass_legacy.SecTheory.LocalOptimizer)
from .LocalOptimizer import PENALTY_SCALE, MEMP_LIMIT, N_LIMIT, RgInfo, SecInfo, LocalOptimizer, split_params
# MEMP_LIMIT and N_LIMIT are imported using this module although they are not used here.

def estimate_monopore_params(ecurve, rg_curve, nc, optimizer=None,
        lrf_src=None,
        elutionmodel_func=robust_single_pore_pdf_scaled,
        t0_upper_bound=None,
        global_opt=False, init_params=None, init_seccol_params=None,
        logger=None,
        debug=False,
        just_get_curves=False):
    """
    to estimate t0, rp, N, me, T, mp

        rho = rho = 1 if rg > rp else rg/rp
        np_ = N*(1 - rho)**me
        tp_ = T*(1 - rho)**mp
        Ksec = (1 - rho)**(me + mp)
        tR = t0 + T*Ksec

    Rg → tR

    """
    if debug:
        import molass_legacy.SecTheory.InitialGuess
        reload(molass_legacy.SecTheory.InitialGuess)
    from .InitialGuess import InitialGuess

    rg_info = RgInfo(rg_curve, nc)
    sec_info = SecInfo(t0_upper_bound=t0_upper_bound)
    localopt = LocalOptimizer(ecurve, rg_info, sec_info, nc, elutionmodel_func, logger=logger)

    params_given = True
    if init_params is None:
        params_given = False
        try:
            initguess = InitialGuess(ecurve, rg_info, sec_info, localopt, debug=debug)
            init_params = initguess.get_params()
            init_tr = initguess.init_tr
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(logger, "InitialGuess failed: ")
            return
    else:
        (t0, rp, N, me, T, mp), rg, w = split_params(init_params, nc)
        rho = rg/rp
        rho[rho > 1] = 1
        init_tr = t0 + N*T*(1 - rho)**(me + mp)

    if just_get_curves:
        return localopt.objective_all(init_params, get_curves=True)

    ret = localopt.optimize(init_params, params_given=params_given, init_tr=init_tr, global_opt=global_opt, debug=debug)

    return ret

def compute_rg(t0, rp, N, me, T, mp, tr):
    from scipy.optimize import root

    def fun(x):
        return t0 + N*T*(1 - x)**(me + mp) - tr

    ret = root(fun, np.ones(len(tr))*0.7)
    return ret.x * rp

def compute_mu_sigma(t0, rp, N, me, T, mp, rg):
    rho = 1 if rg > rp else rg/rp
    np_ = N*(1 - rho)**me
    tp_ = T*(1 - rho)**mp
    return t0 + np_*tp_, np.sqrt(2 * np_ * tp_**2)

def get_modified_params(ecurve, nc, params, elutionmodel_func=robust_single_pore_pdf_scaled, logger=None, debug=False):
    from molass_legacy.QuickAnalysis.ModeledPeaks import recognize_peaks, egh

    x = ecurve.x
    y = ecurve.y

    (t0, rp, N, me, T, mp), rg, w = split_params(params, nc)
    rho = rg/rp
    rho[rho > 1] = 1
    ty = np.zeros(len(x))

    model_trs = t0 + N*T*(1 - rho)**(me + mp)

    penalty = 0
    preference = 0
    cy_list = []
    for w_, r_ in zip(w, rho):
        penalty += min(0, w_)**2
        np_ = N*(1 - r_)**me
        tp_ = T*(1 - r_)**mp
        cy = w_*elutionmodel_func(x - t0, np_, tp_)
        ty += cy
        cy_list.append(cy)

    resid_y = np.max([np.zeros(len(x)), y - ty], axis=0)
    for L, _, R in ecurve.peak_info:
        resid_y[L:R+1] = 0

    total_area = np.sum(ty)
    try:
        peaks = recognize_peaks(x, resid_y, num_peaks=2)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(logger, "recognize_peaks: ")
        return None

    w_pairs = [(i, v) for i, v in zip(np.arange(nc), w)]
    s_pairs = sorted(w_pairs, key=lambda x: x[1])

    i = 0
    modify_list = []
    for k, (h, m, s, t) in enumerate(peaks):
        ry = egh(x, h, m, s, t)
        area_ratio = np.sum(ry)/total_area
        print([k], "area_ratio=", area_ratio)
        if area_ratio > 0.01:
            modify_list.append((s_pairs[i][0], m))
            i += 1

    if len(modify_list) == 0:
        return None

    modify_array = np.array(modify_list)
    rgs_to_add = compute_rg(t0, rp, N, me, T, mp, modify_array[:,1])

    new_rgw_list = []
    i = 0
    for k, (rg_, w_) in enumerate(zip(rg, w)):
        if i < modify_array.shape[0]:
            j = int(modify_array[i,0])
        else:
            j = -1
        print([k], i, j)
        if k == j:
            print("append", i)
            new_rgw_list.append((rgs_to_add[i], w_))
            i += 1
        else:
            new_rgw_list.append((rg_, w_))

    new_rgw_list = sorted(new_rgw_list, key=lambda x: -x[0])    # decending order of Rg
    new_rgw_array = np.array(new_rgw_list)
    new_rg = new_rgw_array[:,0]
    new_w = new_rgw_array[:,1]

    if debug:
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            ax1.plot(x, y)
            for cy, tr in zip(cy_list, model_trs):
                ax1.plot(x, cy, ":")
                j = int(round(tr))
                if j < len(cy):
                    py = cy[j]
                    ax1.plot(tr, py, "o", color="yellow")
            ax1.plot(x, ty, ":", color="red")
            ax2.plot(x, resid_y)
            for peak in peaks:
                ax2.plot(x, egh(x, *peak), ":")
            fig.tight_layout()
            plt.show()

    return np.concatenate([(t0, rp, N, me, T, mp), new_rg, new_w])

def estimate_uv_scale_params(xr_curve, rg_curve, uv_curve, nc, init_params, xr_w, uv_x, uv_y, xr_x, xr_y, optimizer=None, debug=False):
    # refactoring memo: too many arguments. better make it to a class

    cy_list, ty = estimate_monopore_params(xr_curve, rg_curve, nc, init_params=init_params, optimizer=optimizer, just_get_curves=True, debug=debug)
    uv_w = xr_w/xr_curve.max_y * uv_curve.max_y

    pd_cy_list = [cy/w for cy, w in zip(cy_list, xr_w)]

    # optimize uv_w
    def objective_func(p):
        ty = np.zeros(len(uv_y))
        negetive_penalty = 0
        for w, pd_cy in zip(p, pd_cy_list):
            negetive_penalty += min(0, w)**2
            ty += w*pd_cy
        negetive_penalty *= PENALTY_SCALE
        return np.sum((ty - uv_y)**2) + negetive_penalty

    ret_uv = minimize(objective_func, uv_w)
    uv_w = ret_uv.x

    baseline_type = get_setting("unified_baseline_type")
    if baseline_type == 2:
        uv_ty = np.zeros(len(uv_x))
        for k, pd_cy in enumerate(pd_cy_list):
            uv_cy = pd_cy*uv_w[k]
            uv_ty += uv_cy
    else:
        uv_ty = None    # not used in linear baseline

    if debug:
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("estimate_uv_scale_params")
            ax1.plot(uv_x, uv_y, color="blue", label="data")
            ax2.plot(xr_x, xr_y, color="orange", label="data")
            uv_ty = np.zeros(len(uv_x))
            for k, pd_cy in enumerate(pd_cy_list):
                uv_cy = pd_cy*uv_w[k]
                uv_ty += uv_cy
                ax1.plot(uv_x, uv_cy, ":", label="component-%d" % (k+1))
                xr_cy = pd_cy*xr_w[k]
                ax2.plot(xr_x, xr_cy, ":", label="component-%d" % (k+1))

            ax1.plot(uv_x, uv_ty, ":", color="red", label="component-total")
            ax2.plot(xr_x, ty, ":", color="red", label="component-total")
            ax1.legend()
            ax2.legend()
            fig.tight_layout()
            plt.show()

    return uv_w, uv_ty
