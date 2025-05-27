"""
    SecTheory.Felinger2005.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import basinhopping
from datetime import datetime
import molass_legacy.KekLib.DebugPlot as plt
from .SecCF import moving_zone_phi, two_site_phi
from .SecPDF import FftInvPdf, c

class MonoPoreModel:
    def __init__(self):
        self.pdf = c

    def fit(self, x, y, init_params=None):
        h_ = 100
        n1_ = len(x)//2
        t1_ = 10
        def obj_func(p):
            h, n1, t1 = p
            return np.sum((h*self.pdf(x, n1, t1) - y)**2)

        if init_params is None:
            init_params = [20, 60, 3]

        # ret = basinhopping(obj_func, (h_, n_, p1_, t1_, t2_))
        ret = basinhopping(obj_func, init_params)
        # ret = basinhopping(obj_func, init_params)
        return ret

    def __call__(self, x, h, n1, t1):
        return h*self.pdf(x, n1, t1)

class MonoPoreMovingZoneModel:
    def __init__(self):
        self.pdf = FftInvPdf(moving_zone_phi)

    def fit(self, x, y, init_params=None):
        def obj_func(p):
            h, n1, t1, N0, t0 = p
            # neg_penalty = min(0, n1)**2 + min(0, t1)**2 + min(0, N0)**2 + min(0, t0)**2
            neg_penalty = min(0, n1)**2 + min(0, N0)**2
            return np.sum((h*self.pdf(x, n1, t1, N0, t0) - y)**2) + neg_penalty*1e8

        if init_params is None:
            init_params = [20, 240, 1, 100, 10]

        # ret = basinhopping(obj_func, (h_, n_, p1_, t1_, t2_))
        ret = basinhopping(obj_func, init_params)
        # ret = basinhopping(obj_func, init_params)
        return ret

    def __call__(self, x, h, n1, t1, N0, t0):
        return h*self.pdf(x, n1, t1, N0, t0)

def demo():
    from .SecCF import simple_phi, moving_zone_phi
    from .SecPDF import c, FftInvPdf

    t = np.arange(300)

    npi = 10
    tpi = 10
    simple_pdf = FftInvPdf(simple_phi)

    moving_pdf = FftInvPdf(moving_zone_phi)
    N0 = 6
    t0 = 6

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        fig.suptitle("Proof of Numerical Inversion", fontsize=20)
        ax1.set_title("Simple", fontsize=16)
        ax1.plot(t, c(t, npi, tpi), label="formula")
        ax1.plot(t, simple_pdf(t, npi, tpi), label="numerical")
        ax1.legend()
        ax2.set_title("Moving Zone considered", fontsize=16)
        ax2.plot(t, moving_pdf(t, npi, tpi, N0, t0), label="numerical")
        ax2.legend()
        fig.tight_layout()
        plt.show()

def real_demo(in_folder, trimming=True, correction=True):
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


    if False:
        model = MonoPoreModel()
        # init_params = [10, 150, 0.99, 0.1, 1]
        # init_params = [21.09085630844535, 57.05478375607099, 0.8134837337129833, 2.8055854804291713, 2.8055891557224237]
        # init_params = [21.09085438503959, 57.05479161605151, 0.4742816953836163, 2.8055865820081425, 2.805584983307841]
        # init_params = [21.0908540732968, 57.05479955016449, 0.8037212374299743, 2.8055851256770192, 2.805586333848221]
        # init_params = [21.0908535, 57.05480977, 2.80558488]
        init_params = [21.09085432546422, 57.05479510560458, 2.8055856104192967]
    elif in_folder.find("OA") > 0:
        model = MonoPoreMovingZoneModel()
        # init_params = [20, 240, 1, 100, 10]
        init_params = [21.279552638748612, 320.1301176968762, 1.8177109143998595e-07, 27.459933033640482, 157.87215347747625]
    else:
        assert False

    fh = open("params.csv", "w")

    ok_params = []
    min_value = None

    for k in range(10000):
        ret = model.fit(ex, ey, init_params=init_params)
        params = ret.x
        print("ret.fun=", ret.fun)

        if np.isnan(ret.fun):
            print("******************************** bad params =", params)
            init_params = ok_params[-1]
            continue

        if k % 1000 == 0:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), [k], "---------------------------------------------")

        # if k % 100 == 0:
        if k == 0:
            with plt.Dp():
                in_folder = get_in_folder(in_folder)
                fig, ax = plt.subplots()
                ax.set_title("Two-site Stochastic Model Fitting for %s" % in_folder, fontsize=20)
                # simple_plot(ax, ecurve, legend=False)
                ax.plot(ex, ey, label="data")
                ax.plot(ex, egh(ex, *popt1), ":", lw=3, label="egh")
                ax.plot(ex, model(ex, *params), ":", lw=3, label="stochastic model")
                ax.legend(fontsize=16)
                fig.tight_layout()
                plt.show()

        if min_value is None or ret.fun < min_value:
            print([k], "------------------ ret.fun=", ret.fun)
            print("params=", params)
            fh.write("[%d] fv=%s\n" % (k, str(ret.fun)))
            fh.write("params=" + ",".join([str(p) for p in params]) + "\n")
            fh.flush()
            init_params = params
            min_value = ret.fun
            ok_params.append(params)

    fh.close()
