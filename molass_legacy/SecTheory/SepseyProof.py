# coding: utf-8
"""
    SecTheory.SepseyProof.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.special import iv
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt

def fitting_demo_impl(in_folder, stochastic_model, init_params, niter):
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

    for k in range(niter):
        opt_result = solver.fit_impl(stochastic_model, init_params=init_params)
        init_params = opt_result.x

    spline = solver.get_spline(init_params)

    ax.plot(x, y, label="data")
    ax.plot(x, spline(x), label="model")

    ax.legend()

    fig.tight_layout()
    plt.show()

def fitting_demo_lognormal_pore_cf(in_folder, init_params, niter=1):
    from .Sepsey2014 import lognormal_pore_cf
    fitting_demo_impl(in_folder, lognormal_pore_cf, init_params, niter)
