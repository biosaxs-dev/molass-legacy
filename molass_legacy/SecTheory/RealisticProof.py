# coding: utf-8
"""
    SecTheory.RealisticProof.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.special import iv
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt
from .SecCF import simple_phi
from .SecPDF import FftInvPdf

def single_pore_cf(w, rg, rp, x0, nperm, tperm, me, mp):
    if rg > rp:
        np_ = 0
        tp_ = 0
    else:
        rho = rg/rp
        np_ = nperm*(1 - rho)**me
        tp_ = tperm*(1 - rho)**mp
    time_shift = np.exp(-2*np.pi*1j*x0*w)
    return time_shift*simple_phi(w, np_, tp_)

class RealisticCf:
    def __init__(self, n_species, model_cf):
        self.n_species = n_species
        self.model_cf = model_cf

    def split_params(self, flat_params):
        n = self.n_species*2
        rg_params = flat_params[0:n].reshape((self.n_species,2))
        rp_params = flat_params[n:]
        return rg_params, rp_params

    def __call__(self, w, params):
        rg_params, rp_params = self.split_params(params)
        z = 0
        for rg, weight in rg_params:
            z += weight*self.model_cf(w, rg, *rp_params)
        return z

    def objective_func(self, w, params, cft):
        rg_params, rp_params = self.split_params(params)
        z = 0
        penalty = 0
        for rg, weight in rg_params:
            z += weight*self.model_cf(w, rg, *rp_params)
            penalty += min(0, rg)**2 + max(rg - 200, 0)**2          # rg
            penalty += max(rp_params[0] - 1000, 0)**2               # rp
            x0 = rp_params[1]
            penalty += min(x0, -100)**2 + max(100 - x0, 0)**2       # x0
            tperm = rp_params[3]
            penalty += min(tperm, 0)**2 + max(1 - tperm, 0)**2      # tperm
            for v in rp_params[-2:]:
                penalty += min(0, v)**2 + max(v - 3, 0)**2          # me, mp
            penalty += min(0, weight)**2
        return np.sum(np.abs(z - cft)) + penalty*1e8

    def print_params(self, params):
        rg_params, rp_params = self.split_params(params)
        for k, (rg, weight) in enumerate(rg_params):
            print([k], "%.3g, %.3g" % (rg, weight))
        for name, value in zip(["rp", "x0", "nperm", "tperm", "me", "mp"], rp_params):
            print("%s = %.3g" % (name, value))

def fitting_demo(in_folder, niter=1):
    from scipy.optimize import curve_fit
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Baseline.BaselineUtils import get_corrected_sd_impl
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    from molass_legacy.Peaks.ElutionModels import egh
    from LPM import get_corrected
    from DataUtils import get_in_folder
    from SecSaxs.DataTreatment import DataTreatment
    from .RealisticSolver import RealisticCfDomain

    sp = StandardProcedure()
    sd = sp.load_old_way(in_folder)

    pre_recog = PreliminaryRecognition(sd)
    treat = DataTreatment(route="v2", trimming=2, correction=1)
    sd_copy = treat.get_treated_sd(sd, pre_recog)

    D, E, qv, ecurve = sd_copy.get_xr_data_separate_ly()

    fig, ax = plt.subplots()

    x = ecurve.x
    y = ecurve.y

    real_cf = RealisticCf(3, single_pore_cf)
    domain = RealisticCfDomain(x, y, real_cf)
    area = domain.area

    rg_params = np.array([(rg, area*w) for rg, w in zip((70, 60, 50), (0.1, 0.8, 0.1))]).flatten()
    #            rp, nperm, tperm, me, mp
    rp_params = np.array([70, 2000, 1, 1, 1])
    init_params = np.concatenate([rg_params, rp_params])

    if False:
        init_params= np.array([
            9.00000000e+01, -3.03669344e-01,
            6.96923790e+01,  6.50125052e-02,
            6.90057185e+01,  6.46371911e-01,
            7.05385931e+01,  2.64632266e+03,   5.05682855e-01,  3.45833675e-01,  4.90885787e-02])

    for k in range(niter):
        opt_result = domain.fit(init_params=init_params)
        init_params = opt_result.x

    real_cf.print_params(init_params)

    spline = domain.get_spline(init_params)

    ax.plot(x, y, label="data")
    ax.plot(x, spline(x), label="model")
    for k, spline in enumerate(domain.get_component_splines(init_params)):
        ax.plot(x, spline(x), label="component=%d" % k)

    ax.legend()

    fig.tight_layout()
    plt.show()
