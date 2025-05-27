# coding: utf-8
"""
    SecTheory.StochasticModels.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.special import iv
from scipy.optimize import basinhopping
import molass_legacy.KekLib.DebugPlot as plt

def stochastic(x, h, tpi, npi, scale, t0):
    x_ = x*scale - t0
    return h * iv(1, np.sqrt(4*npi*x_/tpi)) * np.sqrt(npi/(x_*tpi)) * np.exp(-x_/tpi-npi)

def demo(parent, in_folder, trimming=True, correction=False, p0=None):
    from scipy.optimize import curve_fit
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Baseline.BaselineUtils import get_corrected_sd_impl
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    from molass_legacy.Elution.CurveUtils import simple_plot
    from molass_legacy.Peaks.ElutionModels import egh
    from LPM import get_corrected
    from DataUtils import get_in_folder

    sp = StandardProcedure()
    sd = sp.load_old_way(in_folder)

    pre_recog = PreliminaryRecognition(sd)
    if trimming:
        sd_ = sd._get_analysis_copy_impl(pre_recog)
    else:
        sd_ = sd.get_copy()

    if correction:
        v2_copy = get_corrected_sd_impl(sd_, sd, pre_recog)
    else:
        v2_copy = sd_

    D, _, wv, ecurve = v2_copy.get_uv_data_separate_ly()

    ex = ecurve.x[10:-10]
    ey = get_corrected(ecurve.y[10:-10])

    p_init = ecurve.get_emg_peaks()[ecurve.primary_peak_no].get_params()
    popt1, pcov1 = curve_fit(egh, ex, ey, p_init)

    if p0 is None:
        # p0 = (400, 25, 25, 1, 0)
        # p0 = (274.72277122, 18.89459478, 49.19590951, 1.3624587, 13.62451978)
        p0 = (275.1072381, 14.57463207, 63.54627583, 1.35989494, 13.59206267)

    if True:
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("init debug")
            ax.plot(ex, ey)
            ax.plot(ex, stochastic(ex, *p0))
            plt.show()
    if True:
        def obj_func(p):
            y = stochastic(ex, *p)
            ret = np.sum((y - ey)**2)
            if np.isfinite(ret):
                return ret
            else:
                return 1e10
        for k in range(1):
            print([k], "---- basinhopping")
            result = basinhopping(obj_func, p0, stepsize=10)
            p0 = result.x
        popt2 = result.x
    else:
        popt2, pcov2 = curve_fit(stochastic, ex, ey, p0)

    print("popt2=", popt2)
    with plt.Dp():
        in_folder = get_in_folder(in_folder)
        fig, ax = plt.subplots()
        ax.set_title("Stochastic Model Trial for %s" % in_folder, fontsize=20)
        # simple_plot(ax, ecurve, legend=False)
        ax.plot(ex, ey, label="data")
        ax.plot(ex, egh(ex, *popt1), ":", lw=2, label="egh")
        ax.plot(ex, stochastic(ex, *popt2), ":", lw=2, label="stochastic model")
        ax.legend(fontsize=16)
        fig.tight_layout()
        plt.show()
