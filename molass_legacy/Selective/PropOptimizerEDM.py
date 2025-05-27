"""
    Selective.PropOptimizerEDM.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from time import sleep, time
from molass_legacy._MOLASS.SerialSettings import set_setting
from molass_legacy.Models.RateTheory.EDM import guess_multiple_impl, edm_impl
from molass_legacy.KekLib.OurTkinter import Tk
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from .PropOptimizerImpl import compute_range_rgs, PROP_MIN_VALUE, RangeRgComputer

VERY_SMALL_RG = 1

def guess_init_params(x, y):
    num_peaks = 2
    params_array = np.array(guess_multiple_impl(x, y, num_peaks))
    return params_array

def compute_props(x, params_array):
    areas = []
    for params in params_array:
        cy = edm_impl(x, *params)
        areas.append(np.sum(cy))
    props = np.array(areas)/np.sum(areas)
    return props

def compute_cy_list(x, params_array):
    cy_list = []
    for params in params_array:
        cy = edm_impl(x, *params)
        cy_list.append(cy)
    return cy_list

def optimize_to_props(x, y, peaks, props, use_basinhopping=False):
    assert len(peaks) == len(props)
    peaks = np.asarray(peaks)
    props = np.asarray(props)

    def objective(p):
        ty = np.zeros(len(y))
        p_ = p.reshape(peaks.shape)
        areas = []
        try:
            cy_list = []
            for params in p_:
                cy = edm_impl(x, *params)
                cy_list.append(cy)
                areas.append(np.sum(cy))
                ty += cy
            props_ = np.array(areas)/np.sum(areas)
            area_dev = np.sum((ty - y)**2)
            prop_dev = max(PROP_MIN_VALUE, np.sum((props_ - props)**2))
            fv = np.log(area_dev) + np.log(prop_dev) - np.log(np.std(p_[:,1]))
        except:
            fv = np.inf
        return fv

    init_params = peaks.flatten()
    if use_basinhopping:
        ret = basinhopping(objective, init_params, minimizer_kwargs=dict(method="Nelder-Mead"))
    else:
        ret = minimize(objective, init_params, method="Nelder-Mead")
    return ret

def compute_optimal_proportion(progress_queue, job_args):
    for i in range(10):
        print([i])
        sleep(1)
        progress_queue.put([i])
    progress_queue.put([-1])
