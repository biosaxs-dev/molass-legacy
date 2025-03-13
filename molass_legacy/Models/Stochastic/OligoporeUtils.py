"""
    Models.Stochastic.OligoporeUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from matplotlib.widgets import Slider, Button
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from molass_legacy.Models.Stochastic.ParamLimits import USE_K, M1_WEIGHT, M2_WEIGHT, BASINHOPPING_SCALE
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

def draw_exclusion_cuve_oligopore(axt, split_params, olgp_params, trs, rgs):
    # this exclusion cuve is not yet implement oligopore
    (N, T, x0), pszv, pszp, p_rgs, N0 = split_params(olgp_params)
    me, mp = 1.5, 1.5
    max_poresize = np.max(pszv)
    rv = np.linspace(max_poresize, 10, 100)
    rhov = rv/max_poresize
    tv = x0 + N*T*(1 - rhov)**(me+mp)
    axt.plot(tv, rv, lw=2, color="yellow")
    axt.plot(trs, rgs, "o", color="red")
    axt.axvline(x=x0, color="red")

def plot_oligopore_moments_state(x, y, cy_list, ty, egh_moments_list, moments, split_params, olgp_params, rgs, params_scaler, title=None, save_fig_as=None):
    top_params, pszv, pszp, p_rgs, N0 = split_params(olgp_params)

    print("pszv=", pszv, "pszp=", pszp)

    plot_params = np.asarray(olgp_params).copy()
    print("plot_params=", plot_params)

    slider_specs = [    ("N", 0, 6000, plot_params[0]),
                        ("T", 0, 3, plot_params[1]),
                        ("t0",  -1000, 1000, plot_params[2]),
                        ("me", 0, 3, plot_params[3]),
                        ("mp", 0, 3, plot_params[4]),
                        ("poresize", 20, 400, plot_params[5]),
                        ]

    with plt.Dp():
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        fig.suptitle(title)
        ax1.set_title("Elution Curve")
        ax1.plot(x, y, color='orange')
        for k, (cy, M, eghM) in enumerate(zip(cy_list, moments, egh_moments_list)):
            ax1.plot(x, cy, ':')
            ax1.axvline(x=M[0], color="green")
            ax1.axvline(x=eghM[0], ls=":", color="cyan")
        bounds = params_scaler.get_bounds()
        t0_upper = bounds[2][1]
        ax1.axvline(x=t0_upper, color="pink")
        # ax1.plot(x, ty, ':', color="red")
        axt = ax1.twinx()
        axt.grid(False)
        trs = [M[0] for M in moments]
        draw_exclusion_cuve_oligopore(axt, split_params, olgp_params, trs, rgs)
        ax2.set_title("Pore Size Distribution")
        ax2.bar(pszv, pszp, width=10)
        fig.tight_layout()
        ret = plt.show()

def plot_oligopore_conformance_impl(ax, axt, N, K, t0, pszs, pszp, plot_info,
                                    rg_info=None, confrg_info=None,
                                    plot_cuvatures=False, ax2=None):

    if rg_info is None:
        max_rg = 120
    else:
        max_rg = max(120, np.max(rg_info[0]) * 1.1)
    rv = np.linspace(10, max_rg, 100)

    trs_list = []
    for poresize, p in zip(pszs, pszp):    
        rhov = rv/poresize
        rhov[rhov > 1] = 1
        trs_list.append(p * K*np.power(1 - rhov,3))
    tv = t0 + np.sum(trs_list, axis=0)
    x = plot_info.x
    y = plot_info.y
    model = plot_info.model
    cy_list = []
    for params in plot_info.peaks:
        cy = model(x, params)
        cy_list.append(cy)

    ax.set_title("Exclusion Points of %s against Exclusion Curve" % get_in_folder(), fontsize=16)
    ax.set_xlabel("Time (Frames)")
    ax.set_ylabel("Intensity")
    ax.plot(x, y, color="orange")
    for cy in cy_list:
        ax.plot(x, cy, ":")

    ax.axvline(x=t0, color="red", label="$t_0$=%d" % int(t0))
    ax.legend(loc="upper left")

    # axt = ax.twinx()
    axt.grid(False)
    axt.set_ylabel(r"$R_g (\AA_)$")
    def join_values(values):
        return ','.join(["%.4g" % v for v in values])
    PSD = "(%s), (%s)" % (join_values(pszs), join_values(pszp))
    axt.plot(tv, rv, color='yellow', lw=3, label="Estimated Exclusion Curve: K=%.4g, PSD=%s" % (K, PSD))
    ry = None
    if rg_info is not None:
        rgs, trs = rg_info[0:2]
        axt.plot(trs, rgs, "o")
        if len(rg_info) > 2:
            from scipy.interpolate import UnivariateSpline
            try:
                if rg_info[2]:  # draw spline
                    spline = UnivariateSpline(trs, rgs)
                ry = spline(tv)
                moderate = ry < 100     # avoid distortion caused by drawing extreme values as in 20201211
                axt.plot(tv[moderate], ry[moderate], color="cyan", alpha=0.5, lw=3, label="Spline of Exclusion Points")
            except:
                log_exception(None, "UnivariateSpline ry failure: ")
            
    if confrg_info is not None:
        rgs, trs = confrg_info
        axt.plot(trs, rgs, "o")

    ymin, ymax = axt.get_ylim()
    axt.set_ylim(max(0, ymin), max_rg)

    axt.legend(loc="upper right")

    if plot_cuvatures:
        from molass_legacy.KekLib.OurCurvature import curvature_curve
        ax2.set_title("Corresponding Curvatures", fontsize=16)
        ax2.set_xlabel("Time (Frames)")
        ax2.set_ylabel("Curvature")

        # to avoid ValueError: x must be strictly increasing if s = 0
        t = np.flip(tv)     
        diff_t = np.diff(t)
        diff_t_pos = np.where(diff_t > 0)[0]
        p = diff_t_pos[0]
        t_ = t[p:]
        try:
            cy = curvature_curve(t_, np.flip(rv)[p:])

            # to avoid anomalies
            m = np.argmax(cy)
            # there can be anomalies beyond m as in 20191006_OA_Ald_Ferr
            diff_cy = np.diff(cy[m:])
            diff_cy_pos = np.where(diff_cy > 0)[0]
            if len(diff_cy_pos) > 0:
                m_ = m + diff_cy_pos[-1] + 1
            else:
                # as in 20210727/data02 
                m_ = m

            ax2.plot(t_[m_:], cy[m_:], color="yellow", lw=3, label="Estimated Exclusion Curve Curvature")
        except:                
            log_exception(None, "curvature_curve rv failure: ")
        if ry is not None:
            try:
                cy_ = curvature_curve(t_, np.flip(ry)[p:])
                ax2.plot(t_, cy_, color="cyan", alpha=0.5, lw=3, label="Spline Curvature of Exclusion Points")
            except:
                log_exception(None, "curvature_curve ry failure: ")
        ax2.legend()

def plot_oligopore_conformance(N, K, t0, pszs, pszp, plot_info, rg_info=None, confrg_info=None, plot_cuvatures=False):
    from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

    with plt.Dp():
        if plot_cuvatures:
            fig, (ax,ax2) = plt.subplots(ncols=2, figsize=(18,5))
        else:
            fig, ax = plt.subplots(figsize=(9,5))
            ax2 = None

        fig.suptitle("Exclusion Curve Estimate on %s" % get_in_folder(), fontsize=20)

        axt = ax.twinx()
        plot_oligopore_conformance_impl(ax, axt, N, K, t0, pszs, pszp, plot_info,
                                    rg_info=rg_info, confrg_info=confrg_info,
                                    plot_cuvatures=plot_cuvatures, ax2=ax2)

        fig.tight_layout()
        ret = plt.show()

    return ret

class OligoParamSplitter:
    def __init__(self, num_pszs, num_unreliables=0, dispersive=False):
        assert num_unreliables == 0
        assert not dispersive
        self.num_pszs = num_pszs
        self.num_unreliables = num_unreliables

    def __call__(self, p):
        num_pszs = self.num_pszs
        N, K, t0 = p[0:3]
        pszs = p[3:3+num_pszs]
        pszp = np.zeros(num_pszs)
        pszp[0:-1] = p[3+num_pszs:3+2*num_pszs-1]
        pszp[-1] = 1 - np.sum(pszp[0:-1])
        return N, K, t0, pszs, pszp

def guess_oligopore_colparams(num_pszs, rgs, qualities, egh_moments_list, init_params, m_props, debug=False, plot_info=None):
    from molass_legacy.Models.Stochastic.ParamLimits import OligoporeParamsScaler

    print("guess_oligopore_colparams: init_params=",init_params)

    assert USE_K    # support only USE_K

    split = OligoParamSplitter(num_pszs)

    me = 1.5
    mp = 1.5

    use_timescale = True
    use_basinhopping = True
    num_timescales = 1 if use_timescale else 0
    ORDER_PENALTY_SCALE = 1e6
    params_scaler = OligoporeParamsScaler(num_pszs, egh_moments_list, 0, rgs, num_timescales=num_timescales, allow_near_t0=True, t0_lower=None)    # t0_lower=700 for 20220716/FER_OA_302

    def objective(p, debug=False):
        if use_basinhopping:
            p = params_scaler.scale_back(p)
        N, K, t0, pszv, pszp = split(p)
        T = K/N
        if use_timescale:
            T *= p[-1]
            t0 *= p[-1]

        order_penalty = abs(min(0, np.min(np.diff(pszv)))) * ORDER_PENALTY_SCALE

        dev_list = []
        for k, (M, rg) in enumerate(zip(egh_moments_list, rgs)):
            rhov = rg/pszv
            rhov[rhov > 1] = 1
            M1_ = t0 + N * T * np.sum(pszp * (1 - rhov)**(me + mp))
            M2_ = 2 * N * T**2 * np.sum(pszp * (1 - rhov)**(me + 2*mp))
            dev_list.append(M1_WEIGHT*(M1_ - M[0])**2 + M2_WEIGHT*(M2_ - M[1])**2)

        return np.sum(np.asarray(dev_list) * m_props) + order_penalty

    bounds = params_scaler.get_bounds()
    print("bounds=", bounds)
    timescale = [1] if use_timescale else []
    init_params = np.concatenate([init_params, timescale])

    if use_basinhopping:
        init_params_ = params_scaler.scale(init_params)
        bounds_ = [(0, BASINHOPPING_SCALE)] * len(init_params)
        minimizer_kwargs = dict(method='Nelder-Mead', bounds=bounds_)
        res = basinhopping(objective, init_params_, minimizer_kwargs=minimizer_kwargs)
        res.x = params_scaler.scale_back(res.x)     # do not forget to scale back the result
    else:
        res = minimize(objective, init_params, method='Nelder-Mead', bounds=bounds)

    N, K, t0, pszs, pszp = split(res.x)
    if use_timescale:
        K *= res.x[-1]
        t0 *= res.x[-1]

    if debug and plot_info is not None:
        print("res.x=", res.x)
        trs = np.array([M[0] for M in egh_moments_list])
        ret = plot_oligopore_conformance(N, K, t0, pszs, pszp, plot_info, rg_info=(rgs, trs, True), plot_cuvatures=False)
        if not ret:
            return

    # return excluding time scale
    return N, K, t0, pszs, pszp