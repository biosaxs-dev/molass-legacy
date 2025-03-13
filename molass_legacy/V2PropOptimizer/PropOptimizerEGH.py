"""
    V2PropOptimizer.PropOptimizerEGH.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from time import sleep
from molass_legacy.Peaks.ElutionModels import egh
from molass_legacy.QuickAnalysis.ModeledPeaks import recognize_peaks
from .PropOptimizer import PROP_MIN_VALUE

def guess_init_params(x, y):
    peaks = recognize_peaks(x, y, num_peaks=2)
    return np.asarray(peaks)

def compute_props(x, peaks):
    areas = []
    for h, m, s, t in peaks:
        cy = egh(x, h, m, s, t)
        areas.append(np.sum(cy))
    props = np.array(areas)/np.sum(areas)
    return props

def compute_cy_list(x, peaks):
    cy_list = []
    for h, m, s, t in peaks:
        cy = egh(x, h, m, s, t)
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
                cy = egh(x, *params)
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
