# coding: utf-8
"""
    Peaks.WindowMapping.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
USE_BASINHOPPING = False
if USE_BASINHOPPING:
    from scipy.optimize import basinhopping
else:
    from scipy.optimize import minimize
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.QuickAnalysis.PeakUtils import recognize_peaks
from molass_legacy.Peaks.ElutionModels import egh

def demo(parent, in_folder):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Elution.CurveUtils import simple_plot

    sp = StandardProcedure()
    sp.load(in_folder)
    sd = sp.get_sd(whole=True)

    uv_curve = sd.get_uv_curve()
    xr_curve = sd.get_xray_curve()


    uv_x = uv_curve.x
    uv_y = uv_curve.y
    uv_spline = uv_curve.spline

    x = xr_curve.x
    y = xr_curve.y
    xr_peaks = recognize_peaks(x, y)

    model_y = np.zeros(len(y))
    for h, mu, sigma, tau in xr_peaks:
        cy = egh(x,  h, mu, sigma, tau)
        model_y += cy

    fx = x[0]
    tx = x[-1]

    def debug_plot(p, add_info=None):
        from DataUtils import get_in_folder

        a, b, fx, tx, scale = p

        plt.push()
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,6))
        fig.suptitle("Debug Plot for %s" % get_in_folder(in_folder=in_folder), fontsize=20)
        ax1.set_title("UV Elution", fontsize=16)
        ax2.set_title("Xray Elution", fontsize=16)

        simple_plot(ax1, uv_curve)
        simple_plot(ax2, xr_curve, color="orange")

        ty = np.zeros(len(y))
        for h, mu, sigma, tau in xr_peaks:
            cy = egh(x,  h, mu, sigma, tau)
            ty += cy
            ax2.plot(x, cy, ':')

        ax2.plot(x, ty, ':')

        ymin1, ymax1 = ax1.get_ylim()
        ax1.set_ylim(ymin1, ymax1)

        ymin2, ymax2 = ax2.get_ylim()
        ax2.set_ylim(ymin2, ymax2)
        for x_ in [fx, tx]:
            uv_x_ = a*x_ + b
            ax1.plot([uv_x_, uv_x_], [ymin1, ymax1], ':', color="gray")
            ax2.plot([x_, x_], [ymin2, ymax2], ':', color="gray")

        if add_info is not None:
            a, b, jfx, jtx, ifx, itx, residual, range_score, order_penalty, deviation_penalty, xr_range_penalty, uv_range_penalty, wx, wy, wbase = add_info

            print("a, b=", a, b)
            print("(jfx, jtx), (ifx, itx)=", (jfx, jtx), (ifx, itx))
            ax1.plot(wx, wy)
            ax1.plot(wx, wbase, color="red")

            score_list = [residual, range_score, order_penalty, deviation_penalty, xr_range_penalty, uv_range_penalty]
            score_names = "residual, range_score, order_penalty, deviation_penalty, xr_range_penalty, uv_range_penalty"

            fv = np.sum(score_list)
            ax3.set_title("fv=%.3g" % fv, fontsize=16)

            ax3.yaxis.tick_right()
            ax3.yaxis.set_label_position("right")
            y_pos = np.arange(len(score_list))
            ax3.barh(y_pos, score_list)
            ax3.set_yticks(y_pos)
            ax3.invert_yaxis() 
            ax3.set_yticklabels(score_names.split(", "))

        fig.tight_layout()
        ret = plt.show()
        plt.pop()
        return ret

    a_init = len(uv_x)/len(x)
    b_init = 0
    scale_init = np.max(uv_y)/np.max(y)
    wh = np.where(model_y > xr_curve.max_y*0.01)[0]
    init_params = (a_init, b_init, x[wh[0]], x[wh[-1]], scale_init)

    VERY_LARGE_VALUE = 1e6
    min_a = a_init*0.7
    max_a = a_init*1.4
    max_x = x[-1]
    min_width = max_x*0.2
    min_s = scale_init*0.7
    max_s = scale_init*1.4

    debug = True

    def objective(p, plot=False):
        nonlocal debug

        a, b, fx, tx, scale = p

        if fx < 0 or tx > max_x or tx - fx < min_width or a < min_a or a > max_a or scale < min_s or scale > max_s:
            jfx, jtx, ifx, itx = 0, 0, 0, 0
            residual = VERY_LARGE_VALUE
            range_score = VERY_LARGE_VALUE
            order_penalty = VERY_LARGE_VALUE
            deviation_penalty = VERY_LARGE_VALUE
            xr_range_penalty = VERY_LARGE_VALUE
            uv_range_penalty = VERY_LARGE_VALUE
            wx = wy = wbase = np.zeros(len(x))
        else:
            ifx = max(0, int(np.round(fx)))
            itx = min(len(x)-1, int(np.round(tx)))
            wx = a*x[ifx:itx] + b
            wy = uv_spline(wx)
            n = len(wy)
            wbase = wy[0] + np.arange(n)*1/(n-1)*(wy[-1] - wy[0])
            wy_ = wy - wbase - scale*y[ifx:itx]

            residual = np.sum(wy_**2)
            # range_score = 1e1*(y[ifx]+y[itx])**2/(tx - fx)**2
            jfx = max(0, int(np.round(a*fx + b)))
            jtx = min(len(uv_x) - 1, int(np.round(a*tx + b)))

            range_score = 1/np.sum(model_y[ifx:itx])
            order_penalty = VERY_LARGE_VALUE*(min(0, tx - fx)**2 + min(0, a)**2)
            deviation_penalty = 0
            xr_range_penalty = VERY_LARGE_VALUE*(min(0, fx - x[0])**2 + max(0, tx - x[-1])**2)
            uv_range_penalty = VERY_LARGE_VALUE*(min(0, wx[0] - uv_x[0])**2 + max(0, wx[-1] - uv_x[-1])**2)

        fv = residual + range_score + order_penalty + deviation_penalty + xr_range_penalty + uv_range_penalty

        if debug or plot:
            print("residual=", residual)
            print("range_score=", range_score)
            print("xr_range_penalty=", xr_range_penalty)
            print("uv_range_penalty=", uv_range_penalty)
            debug = debug_plot(p, add_info=(a, b, jfx, jtx, ifx, itx, residual, range_score, order_penalty, deviation_penalty, xr_range_penalty, uv_range_penalty, wx, wy, wbase))

        return fv

    if USE_BASINHOPPING:
        minimizer_kwargs = {"method": "Nelder-Mead"}
        ret = basinhopping(objective, init_params, niter=20, minimizer_kwargs=minimizer_kwargs)
    else:
        ret = minimize(objective, init_params, method="Nelder-Mead")

    objective(ret.x, plot=True)
