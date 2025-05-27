"""
    SecTheory.CavazziniProof.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from datetime import datetime
import molass_legacy.KekLib.DebugPlot as plt
from .SecCF import two_site_phi
from .SecPDF import FftInvPdf

class TwoSiteSpikeModel:
    def __init__(self, m):
        self.m = m
        self.pdf = FftInvPdf(two_site_phi)

    def __call__(self, x, n, p1, tratio):
        t1 = self.m/n
        p2 = 1 - p1
        t2 = tratio*t1
        return self.pdf(x, n, p1, p2, t1, t2)

class TwoSiteModel:
    def __init__(self):
        self.pdf = FftInvPdf(two_site_phi)

    def fit(self, x, y, init_params=None):
        h_ = 100
        n_ = len(x)//2
        p1_ = 0.99
        t1_ = 1
        t2_ = 10*t1_
        def obj_func(p):
            h, n, p1, t1, t2 = p
            neg_penalty = min(0, p1)**2 + (max(1, p1) - 1)**2 + min(0, t1)**2 + min(0, t2)**2
            return np.sum((h*self.pdf(x, n, p1, 1 - p1, t1, t2) - y)**2) + neg_penalty*1e8

        # init_params = [2.01175632e+02, 6.53814668e+02, 9.24384125e-01, 1.49033153e-01,  1.21771676e+01]
        # init_params = [100, 500, 0.99, 0.1,  1]
        # init_params = [201.9764699, 530.36354375,  0.96139286, 0.61745049, 18.6660736]
        if init_params is None:
            init_params = [200, 500, 0.99, 0.5, 5]

        # ret = basinhopping(obj_func, (h_, n_, p1_, t1_, t2_))
        ret = basinhopping(obj_func, init_params)
        # ret = basinhopping(obj_func, init_params)
        return ret

    def __call__(self, x, h, n, p1, t1, t2):
        return h*self.pdf(x, n, p1, 1 - p1, t1, t2)

class TwoSiteAlphaModel:
    def __init__(self):
        self.pdf = FftInvPdf(two_site_phi)

    def fit(self, x, y, init_params=None):
        def obj_func(p):
            h, n, p1, alpha = p
            neg_penalty = min(0, p1)**2 + (max(1, p1) - 1)**2 + min(0, alpha)**2
            return np.sum((h*self.pdf(x, n, p1, 1 - p1, 1, alpha) - y)**2) + neg_penalty*1e8

        # init_params = [2.01175632e+02, 6.53814668e+02, 9.24384125e-01, 1.49033153e-01,  1.21771676e+01]
        # init_params = [100, 500, 0.99, 0.1,  1]
        # init_params = [201.9764699, 530.36354375,  0.96139286, 0.61745049, 18.6660736]
        if init_params is None:
            init_params = [211, 574, 0.995, 65.6]

        # ret = basinhopping(obj_func, (h_, n_, p1_, t1_, t2_))
        ret = basinhopping(obj_func, init_params)
        # ret = basinhopping(obj_func, init_params)
        return ret

    def __call__(self, x, h, n, p1, alpha):
        return h*self.pdf(x, n, p1, 1 - p1, 1, alpha)


def demo0():
    t = np.arange(300)

    two_site_pdf = FftInvPdf(two_site_phi)

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_title("Two Site Model", fontsize=20)
        # n, p1, p2, t1, t2 = 100, 0.99, 0.01, 0.1, 1
        p1, p2 = 0.99, 0.01
        t1 = 0.1
        t2 = 1
        for n in [100, 200, 400, 800, 1000, 2000]:
            # tR = n*(p1*t1 + p2*t2)
            ax.plot(t, two_site_pdf(t, n, p1, p2, t1, t2), label="n=%d" % n)
        # xmin, xmax = ax.get_xlim()
        # ax.set_xlim(0, 3)
        ax.legend()
        fig.tight_layout()
        plt.show()

def demo1():
    t = np.arange(300)

    model = TwoSiteSpikeModel(150)

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_title("Two Site Model", fontsize=10)
        # n, p1, p2, t1, t2 = 100, 0.99, 0.01, 0.1, 1
        p1 = 0.99
        p2 = 1 - p1
        for n in [100, 200, 400, 800, 1000, 2000]:
            ax.plot(t, model(t, n, p1, 10), label="n=%d" % n)
        ax.legend()
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
        model = TwoSiteModel()
        # init_params = [10, 150, 0.99, 0.1, 1]
        # init_params = [21.09085630844535, 57.05478375607099, 0.8134837337129833, 2.8055854804291713, 2.8055891557224237]
        # init_params = [21.09085438503959, 57.05479161605151, 0.4742816953836163, 2.8055865820081425, 2.805584983307841]
        init_params = [21.0908540732968, 57.05479955016449, 0.8037212374299743, 2.8055851256770192, 2.805586333848221]
    else:
        model = TwoSiteModel()
        # init_params = [200, 500, 0.99, 0.5, 5]
        # init_params = [212.06604355, 496.60195744, 0.99516111, 1.15354905,  67.06323863]
        # params= [211.98227569 497.36669375   0.99517607   1.15150084  67.23597666]
        # params= [212.03260787 500.283879     0.99520864   1.14479576  67.35694569]
        # params= [211.76525286 501.38064212   0.99517714   1.14140277  66.72618185]
        # params= [211.93850687 503.1646396    0.99521843   1.1377971   67.13311391]
        # init_params= [211.82262009, 506.42648153, 0.99522858, 1.12996051, 66.8657622]
        # init_params= [211.76262156348102, 522.2824122494156, 0.9953431949768717, 1.0945862185945257, 66.67018690354158]
        # init_params= [211.51931787620344, 542.0343851514614, 0.9954524856411976, 1.0529220645070299, 66.03641538852862]
        # init_params= [211.5125903865166, 558.5731370008768, 0.9955732039827447, 1.0211441250646398, 66.03620025776257]
        # init_params= [211.32184877429745, 566.583019797747, 0.9955967593428952, 1.0057396934143517, 65.56377332774754]
        # init_params = [211.53926986961773, 580.567581537651, 0.9957255508661492, 0.9817207281960352, 66.0832372341951]
        # init_params = [211.23380012691356, 589.007836848129, 0.9957328806835914, 0.9663319382212798, 65.30489077398083]
        # init_params = [211.1730691340855, 605.1641584047355, 0.995826281975228, 0.9398468190500091, 65.15391941190406]
        # init_params = [211.01281199499408, 614.712295404362, 0.9958630063530421, 0.9245231583976347, 64.78366069670133]
        # init_params = [210.80233635855276, 664.0508285684691, 0.996111858418305, 0.8539738842429706, 64.25952300122583]
        # init_params = [210.519377628442, 707.2090960646246, 0.9962884635747533, 0.800127310961128, 63.541234776791946]
        # init_params = [210.29024437581631, 728.897932727702, 0.9963635408056852, 0.7754107393393052, 63.047159820670586]
        # bad init_params = [210.28559177527205, 728.4806403663978, 0.9963682614399137, 0.7760005009963438, 63.17116889910384]
        # init_params = [210.2432129644718, 736.1089550848279, 0.9964039886713507, 0.7678615697396018, 63.17276056221165]
        # init_params = [210.31577022660957, 736.8047259033731, 0.9964016280718834, 0.7669956193079287, 63.09645579362846]
        # init_params = [210.31861692248353, 738.8852507261381, 0.9964109159558705, 0.7648006217330974, 63.09806054328252]
        # bad init_params = [210.340091473509, 739.4386070044351, 0.996413292144056, 0.7642248442437305, 63.10330054692467]
        # init_params = [210.28932436757205, 739.3193830268104, 0.9964112448124383, 0.7643298019955326, 63.05725223453272]
        # bad init_params = [210.3086178076916, 742.9637121509476, 0.9964273491334341, 0.7604953626068545, 63.07460696639221]
        # init_params = [210.34811156158634, 743.3092486833381, 0.9964326679615353, 0.7602056546113496, 63.1493213914855]
        # init_params = [210.2747478064457, 747.6041965051638, 0.9964439548938019, 0.7556281120973185, 62.99773805323856]
        init_params = [210.2452597184559, 749.4992507845662, 0.9964492328705874, 0.7536282577914788, 62.939887204560435]

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

def real_demo1(in_folder, trimming=True, correction=True):
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

    model = TwoSiteAlphaModel()

    fh = open("params.csv", "w")

    init_params = None
    min_value = None

    for k in range(1000):
        ret = model.fit(ex, ey, init_params=init_params)
        params = ret.x
        # print("params=", params)

        if k % 100 == 0:
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

    fh.close()

def fitting_demo_impl(in_folder, stochastic_model, init_params):
    from scipy.optimize import curve_fit
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Baseline.BaselineUtils import get_corrected_sd_impl
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    from molass_legacy.Peaks.ElutionModels import egh
    from LPM import get_corrected
    from DataUtils import get_in_folder
    from SecSaxs.DataTreatment import DataTreatment
    from .StochasticSolver import CfDomain

    sp = StandardProcedure()
    sd = sp.load_old_way(in_folder)

    pre_recog = PreliminaryRecognition(sd)
    treat = DataTreatment(route="v2", trimming=2, correction=1)
    sd_copy = treat.get_treated_sd(sd, pre_recog)

    D, E, qv, ecurve = sd_copy.get_xr_data_separate_ly()

    fig, ax = plt.subplots()

    x = ecurve.x
    y = ecurve.y

    solver = CfDomain(x, y)
    spline = solver.fit(stochastic_model, init_params=init_params)

    ax.plot(x, y, label="data")
    ax.plot(x, spline(x), label="model")

    ax.legend()

    fig.tight_layout()
    plt.show()

def fitting_demo_single_site(in_folder, init_params):
    from .Cavazzini1999 import single_site_cf
    fitting_demo_impl(in_folder, single_site_cf, init_params)

def fitting_demo_two_site(in_folder, init_params):
    from .Cavazzini1999 import two_site_cf
    fitting_demo_impl(in_folder, two_site_cf, init_params)

def fitting_demo_three_site(in_folder, init_params):
    from .Cavazzini1999 import three_site_cf
    fitting_demo_impl(in_folder, three_site_cf, init_params)

def fitting_demo_uniform_adsorption_energy(in_folder, init_params):
    from .Cavazzini1999 import uniform_adsorption_energy
    fitting_demo_impl(in_folder, uniform_adsorption_energy, init_params)

def fitting_demo_uniform_mean_sojourn_time(in_folder, init_params):
    from .Cavazzini1999 import uniform_mean_sojourn_time
    fitting_demo_impl(in_folder, uniform_mean_sojourn_time, init_params)
