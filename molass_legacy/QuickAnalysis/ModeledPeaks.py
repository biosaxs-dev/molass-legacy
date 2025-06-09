"""
    QuickAnalysis.ModeledPeaks.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve, minimize
from scipy.special import erf
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.ElutionCurveModels import egh, egha
from molass_legacy.Peaks.ElutionModels import compute_moments, compute_egh_params
from molass_legacy._MOLASS.SerialSettings import get_setting

RESID_RATIO_LIMIT = 0.03
H_RATIO_LIMIT = 0.03
PENALTY_SCALE = 1e3
TAU_BOUND_RATIO = get_setting("TAU_BOUND_RATIO")    # tau <= sigma*TAU_BOUND_RATIO
MAPPING_ADJUST_ALLOW_RATIO = 0.1
DEFAULT_MIN_AREA_PROP = 0.02

def gaussian(x, h, m, s):
    return h * np.exp(-((x-m)/s)**2)

def gaussian_integral(x, h, m, s):
    return np.sqrt(np.pi) * h * s * erf((x-m)/s) / 2

def get_a_peak(x, y, refine=True, model=None, affine=False, debug=False):
    if model is None:
        from molass_legacy.Models.ElutionCurveModels import EGH, EGHA
        model = EGHA() if affine else EGH()
    model_func = model.func     # task: remove model_func

    pt = np.argmax(y)
    mu_init = x[pt]
    max_y = y[pt]
    smax = len(x)//2
    s_init = smax/2

    for s in range(3, smax):
        slice_ = slice(max(0, pt-s), min(len(x), pt+s))
        x_ = x[slice_]
        y_ = y[slice_]
        gy = gaussian(x_, max_y, mu_init, s)
        if False:
            print("slice_=", slice_)
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("%s plot" % str([s]))
                ax.plot(x, y)
                ax.plot(x_, gy, ":")
                plt.show()
        negative_ratio = len(np.where(y_ - gy < 0)[0])/len(x_)
        if debug:
            print([s], "negative_ratio=", negative_ratio)
        if negative_ratio > 0.5:
            s_init = s
            break

    w = np.where(y[0:pt] < 0)[0]
    fx_lim = x[0] if len(w) == 0 else x[w[-1]]
    w = np.where(y[pt:] < 0)[0]
    tx_lim = x[-1] if len(w) == 0 else x[pt + w[0]]
    max_width = tx_lim - fx_lim

    bnds = ((0, max_y*1.5), (x[0], x[-1]), (0, max_width), (0, max_width))
    y_for_opt = y.copy()
    width = int(s_init*2)
    left = max(0, pt-width)
    right = min(len(x), pt+width)
    y_for_opt[0:left] = 0
    y_for_opt[right:] = 0
    y_for_opt[y_for_opt < 0] = 0

    if debug:
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("y_for_opt proof")
            ax.plot(x, y)
            ax.plot(x, y_for_opt, ":")
            plt.show()

    M = compute_moments(x, y_for_opt)
    init_params = mu_init, s_init, 0
    tR, sigma, tau = compute_egh_params(init_params, M)

    h_init = y[pt]

    if affine:
        a_init = s_init * 0.5
        affine_bnds = bnds + ((-a_init, a_init),)

        def obective_affine(p):
            h, mu, sigma, tau, a = p
            my = model_func(x, h, mu, sigma, tau, a)
            return np.sum((my - y_for_opt)**2) + PENALTY_SCALE*(min(0, tau)**2 + min(0, sigma*TAU_BOUND_RATIO - tau)**2)

        ret = minimize(obective_affine, (h_init, tR, sigma, tau, 0), bounds=affine_bnds)
    else:
        def obective(p):
            h, mu, sigma, tau = p
            my = model_func(x, h, mu, sigma, tau)
            return np.sum((my - y_for_opt)**2) + PENALTY_SCALE*(min(0, tau)**2 + min(0, sigma*TAU_BOUND_RATIO - tau)**2)

        # cons = ({'type': 'ineq', 'fun': lambda x:  x[2] - x[3] },)
        # it seems better to have this included in the obective func
        # than to add constraints=cons below
        ret = minimize(obective, (h_init, tR, sigma, tau), bounds=bnds)

    if debug:
        print("s_init=", s_init)
        print("sigma=", sigma)
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("get_a_peak: s_init, opt_sinit")
            ax1.plot(x, y)
            ax1.plot(x, gaussian(x, max_y, mu_init, s_init), label="gaussian(..., s_init)")
            ax1.plot(x, gaussian(x, max_y, mu_init, sigma), label="gaussian(..., sigma)")
            ax2.plot(x, y)
            ax2.plot(x, y_for_opt, ":")
            ax2.plot(x, model_func(x, h_init, tR, sigma, tau), label="egh(..., sigma, ...)")
            ax2.plot(x, model_func(x, *ret.x), label="egh(optimized)")
            for ax in ax1, ax2:
                ax.legend()
            fig.tight_layout()
            plt.show()

    return ret.x

def recognize_peaks(x, y, num_peaks=5, exact_num_peaks=None, affine=False, model=None, min_area_prop=None, correct=True, debug=False):
    """
    moved here due to 
    ImportError: cannot import name 'get_corrected' from partially initialized module 'LPM' (most likely due to a circular import) (.../DataStructure/LPM.py)
    """
    from molass_legacy.DataStructure.LPM import get_corrected

    if model is None:
        from molass_legacy.Models.ElutionCurveModels import EGH, EGHA
        if affine:
            model = EGHA()
        else:
            model = EGH()

    decide_num_peaks = False
    if min_area_prop is None:
        min_area_prop = DEFAULT_MIN_AREA_PROP
    else:
        decide_num_peaks = True

    get_exact_num_peaks = exact_num_peaks is not None
    max_num_peaks = exact_num_peaks if get_exact_num_peaks else num_peaks

    if max_num_peaks is None:
        max_num_peaks = 7   # MAX_NUM_PEAKS

    if debug:
        print("recognize_peaks: max_num_peaks=", max_num_peaks, "decide_num_peaks=", decide_num_peaks)

    y_max = np.max(y)

    if correct:
        y_copy = get_corrected(y, x=x)
    else:
        y_copy = y.copy()

    peaks_list = []
    if decide_num_peaks:
        total_area = np.sum(y_copy)
    for k in range(max_num_peaks):
        params = get_a_peak(x, y_copy, refine=k == 0, model=model, affine=affine, debug=debug)
        if decide_num_peaks:
            y_model = model(x, params)
            area = np.sum(y_model)
            area_ratio = area/total_area
            if area_ratio < min_area_prop:
                break
        else:
            h_ratio = params[0]/y_max
            if h_ratio < H_RATIO_LIMIT:
                print([k], "h_ratio=", h_ratio)
                if not get_exact_num_peaks:
                    break
            y_model = model(x, params)
        y_resid = y_copy.copy()
        y_resid -= y_model

        resid_ratio = np.max(y_resid)/y_max
        if debug:
            print("resid_ratio=", resid_ratio)
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("recognize_peaks with num_peaks=%d" % num_peaks)
                ax.plot(x, y)
                ax.plot(x, y_model)
                ax.plot(x, y_resid)
                plt.show()

        peaks_list.append(params)

        if get_exact_num_peaks:
            pass
        else:
            if resid_ratio < RESID_RATIO_LIMIT:
                break

        y_copy = y_resid

    if decide_num_peaks:
        ret_peaks_list = []
        props = model.get_proportions(x, peaks_list)
        for i in np.where(props > min_area_prop)[0]:
            ret_peaks_list.append(peaks_list[i])
    else:
        ret_peaks_list = peaks_list

    sorted_list = sorted(ret_peaks_list, key=lambda x: x[1])

    return sorted_list

def plot_curve(ax, x, y, peaks, color='C0', model=None, baseline=None, dcurves=None, legend=True, return_markers=False):
    if model is None:
        from molass_legacy.Models.ElutionCurveModels import EGH, EGHA
        if len(peaks[0]) == 5:
            model = EGHA()
        else:
            model = EGH()

    ax.plot(x, y, color=color, label='data', lw=3)
    ty = np.zeros(len(y))
    k = 0
    markers = []
    for param in peaks:
        cy = model(x, param)
        ty += cy
        cy_ = cy if baseline is None else cy + baseline
        if dcurves is None:
            ax.plot(x, cy_, ':', label='component-%d' % (k+1))
        else:
            dcurves.add( x, cy_, ":", label='component-%d' % (k+1))
        px, py = model.get_peaktop_xy(x, param)
        if px > x[0] and px < x[-1] and py > 0:
            marker, = ax.plot(px, py, "o", color="yellow")
            markers.append(marker)

        k += 1
    ty_ = ty if baseline is None else ty + baseline
    ax.plot(x, ty_, ':', color='red', label='component total', lw=3)
    if baseline is not None:
        ax.plot(x, baseline, color='red', label='baseline')

    if legend:
        ax.legend()

    if return_markers:
        return markers

    return ty

def adjust_peak_heights(a, b, xr_x, xr_y, peaks, uv_x, uv_y, affine=False, debug=False):
    uv_allow = len(uv_x)*MAPPING_ADJUST_ALLOW_RATIO

    def objective(p):
        ty = np.zeros(len(uv_y))
        k = 0
        # range_penalty is required in cases like 20201007_2
        # which might be better treated in trimming
        range_penalty = max(0, abs(xr_x[0]*a + b - uv_x[0]) - uv_allow)**2 + max(0, abs(xr_x[-1]*a + b - uv_x[-1]) - uv_allow)**2
        for params in peaks:
            h, mu, sigma, tau = params[0:4]
            h_ = p[k]
            mu_ = a*mu + b
            sigma_ = a*sigma
            tau_ = a*tau
            if affine:
                ty += egha(uv_x, h_, mu_, sigma_, tau_, params[4])
            else:
                ty += egh(uv_x, h_, mu_, sigma_, tau_)
            k += 1
        return np.sum((ty - uv_y)**2) + range_penalty*PENALTY_SCALE

    h_init = []
    uv_spline = UnivariateSpline(uv_x, uv_y, s=0, ext=3)    # exit=3 would be safe for end points

    if debug:
        debug_peaks = []
    init_heights = []
    for params in peaks:
        h, mu, sigma, tau = params[0:4]
        init_heights.append(h)
        mu_ = a*mu + b
        h_ = uv_spline(mu_)
        h_init.append(h_)
        if debug:
            sigma_ = a*sigma
            tau_ = a*tau
            if affine:
                debug_peaks.append((h_, mu_, sigma_, tau_, params[4]))
            else:
                debug_peaks.append((h_, mu_, sigma_, tau_))

    if debug:
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("adjust_peak_heights initial state")
            plot_curve(ax1, uv_x, uv_y, debug_peaks)
            plot_curve(ax2, xr_x, xr_y, peaks, color="orange")
            fig.tight_layout()
            plt.show()

    min_h = max(0, min(np.min(init_heights), np.min(h_init)))
    bounds = [(min_h, 100)] * len(peaks)
    # print("min_h=", min_h, "h_init=", h_init)

    ret = minimize(objective, h_init, method="Nelder-Mead", bounds=bounds)

    if debug:
        a_ = 1/a
        b_ = - b/a
        mp_x = uv_x*a_ + b_
        ratio = np.max(xr_y)/np.max(uv_y)
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("adjust_peak_heights optimized state")
            plot_curve(ax1, uv_x, uv_y, debug_peaks)
            plot_curve(ax2, xr_x, xr_y, peaks, color="orange", legend=False)
            ax2.plot(mp_x, uv_y*ratio, label="mapped")
            ax2.legend()
            fig.tight_layout()
            plt.show()

    ret_peaks = []
    k = 0
    for params in peaks:
        h, mu, sigma, tau = params[0:4]
        h_ = ret.x[k]
        mu_ = a*mu + b
        sigma_ = a*sigma
        tau_ = a*tau
        if affine:
            ret_peaks.append((h_, mu_, sigma_, tau_, params[4]))
        else:
            ret_peaks.append((h_, mu_, sigma_, tau_))
        k += 1

    return ret_peaks

def get_curve_xy_impl(sd, baseline_type=1, return_details=False, debug=False):
        if debug:
            print("get_curve_xy_impl: baseline_type=", baseline_type)

        baselines = None
        baseline_params = None

        X, E, qv, xr_curve = sd.get_xr_data_separate_ly()
        U, _, wv, uv_curve = sd.get_uv_data_separate_ly()
        xr_x = xr_curve.x
        xr_y = xr_curve.y
        uv_x = uv_curve.x
        uv_y = uv_curve.y

        if return_details:
            from molass_legacy.KekLib.BasicUtils import Struct
            from molass_legacy.Baseline.Constants import SLOPE_SCALE

            if baseline_type == 0:
                assert False
            else:
                if baseline_type == 1:
                    if debug:
                        from importlib import reload
                        import Baseline.LinearBaseline
                        reload(Baseline.LinearBaseline)
                        import UV.UvPreRecog
                        reload(UV.UvPreRecog)
                    from molass_legacy.Baseline.LinearBaseline import LinearBaseline as XrBaselineClass
                    from molass_legacy.UV.UvPreRecog import UvPreRecog
                elif baseline_type == 2:
                    if debug:
                        from importlib import reload
                        import Baseline.IntegralBaseline
                        reload(Baseline.IntegralBaseline)            
                    from molass_legacy.Baseline.IntegralBaseline import IntegralBaseline as XrBaselineClass
                elif baseline_type == 3:
                    if debug:
                        from importlib import reload
                        import Baseline.FoulingBaseline
                        reload(Baseline.FoulingBaseline)            
                    from molass_legacy.Baseline.FoulingBaseline import FoulingBaseline as XrBaselineClass
                else:
                    assert False

                baseline_objects = []
                baselines = []
                baseline_params = []

                pre_recog = sd.pre_recog
                if pre_recog is None:
                    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
                    pre_recog = PreliminaryRecognition(sd)

                upr = UvPreRecog(sd, pre_recog, debug=False)
                base_curve = upr.base_curve
                baseline_objects.append(base_curve)
                ty = None   # dummy
                params = upr.init_params.copy()
                params[4:6] /= SLOPE_SCALE
                baselines.append(base_curve(uv_x, params, ty))
                baseline_params.append(params)

                # for x, y in [(uv_x, uv_y), (xr_x, xr_y)]:
                for x, y in [(xr_x, xr_y)]:
                    baseline = XrBaselineClass(x, y)
                    baseline_objects.append(baseline)
                    baselines.append(baseline.yb)
                    baseline_params.append(baseline.params)
                # uv_y = uv_y - baselines[0]
                # xr_y = xr_y - baselines[1]

            details = Struct(
                        uv_curve=uv_curve,
                        xr_curve=xr_curve,
                        baseline_objects=baseline_objects,
                        baselines=baselines,
                        baseline_params=baseline_params
                        )

            if debug:
                with plt.Dp():
                    fig, axes = plt.subplots(ncols=2, figsize=(12,5))
                    fig.suptitle("get_curve_xy_impl return_details")
                    for ax, (x, y), bl in zip(axes, [(uv_x, uv_y), (xr_x, xr_y)], baselines):
                        ax.plot(x, y)
                        if bl is not None:
                            ax.plot(x, bl, color="red")
                    fig.tight_layout()
                    plt.show()

            return uv_x, uv_y, xr_x, xr_y, details
        else:
            return uv_x, uv_y, xr_x, xr_y

def get_modeled_peaks_impl(a, b, uv_x, uv_y, xr_x, xr_y, num_peaks,
                           peaks=None,
                           exact_num_peaks=None,
                           affine=False, min_area_prop=None, debug=False):
    if debug:
        print("uv_y=", uv_y)
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            ax1.set_title("UV")
            ax2.set_title("XR")
            ax1.plot(uv_x, uv_y)
            ax2.plot(xr_x, xr_y)
            fig.tight_layout()
            plt.show()

    if peaks is None:
        peaks = recognize_peaks(xr_x, xr_y, num_peaks=num_peaks, exact_num_peaks=exact_num_peaks,
                            affine=affine, min_area_prop=min_area_prop, debug=debug)

    uv_peaks = adjust_peak_heights(a, b, xr_x, xr_y, peaks, uv_x, uv_y, affine=affine, debug=debug)
    xr_peaks = adjust_peak_heights(1, 0, xr_x, xr_y, peaks, xr_x, xr_y, affine=affine)

    if debug:
        from .ModeledPeaksTester import plot_modeled_peaks
        plot_modeled_peaks(uv_x, uv_y, xr_x, xr_y, uv_peaks, xr_peaks, a, b, suptitle="get_modeled_peaks_impl: return")

    return np.asarray(uv_peaks), np.asarray(xr_peaks)

def get_proportions(x, peaks):
    areas = []
    for h, mu, sigma, tau in peaks:
        y = egh(x, h, mu, sigma, tau)
        areas.append(np.sum(y))
    return np.array(areas)/np.sum(areas)

def get_peak_region_ends(x, peaks):
    ends = []
    for j in [0, -1]:
        h, mu, sigma, tau = peaks[j]
        sign = -1 if j == 0 else +1
        pos = int(mu + sign*3*(sigma+tau))
        ends.append(min(max(0, pos), len(x)))
    return ends

"""
    this class is experimantal
"""
class PeakRegionLrf:
    def __init__(self, D, E, x, peaks):
        f, t = get_peak_region_ends(x, peaks)
        x_ = x[f:t]
        y_list = []
        for p in peaks:
            y_list.append(egh(x_, *p))
        C = np.array(y_list)
        D_ = D[:,f:t]
        P = D_ @ np.linalg.pinv(C)

        E_ = E[:,f:t]
        D_pinv = np.linalg.pinv(D_)
        W = np.dot(D_pinv, P)
        Pe = np.sqrt(E_**2 @ W**2)

        self.lrf_results = P, Pe
        self.ends = f,t

    def compute_rgs(self, qv):
        from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier
        P, Pe = self.lrf_results
        ret_rgs = []
        for y, ye in zip(P.T, Pe.T):
            sg = SimpleGuinier(np.array([qv, y, ye]).T)
            ret_rgs.append(sg.Rg)
        return ret_rgs
