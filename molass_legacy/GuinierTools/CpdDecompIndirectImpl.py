"""
    GuinierTools.CpdDecompIndirectImpl.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
import ruptures as rpt
import molass_legacy.KekLib.DebugPlot as plt
from GuinierTools.CpdDecompUtils import compute_end_points

VERY_LARGE_VALUE = 1e8
MAX_TAU_RARIO = 1.0

def imporove_decomposition(x, y, model, peaks, end_points):

    new_trs = (end_points[0:-1] + end_points[1:])/2
    min_scale = np.average(peaks[:,0])*0.2
    max_sigma = np.max(peaks[:,2])*1.5

    def near_trs_objective(p):
        cy_list = []
        temp_peaks = p.reshape(peaks.shape)
        for i, params in enumerate(temp_peaks):
            cy = model(x, params)
            cy_list.append(cy)
        ty = np.sum(cy_list, axis=0)
        negetive_penalty = min(0, np.min(temp_peaks[:,0]) - min_scale)**2 * VERY_LARGE_VALUE
        sigma_penalty = max(0, np.max(temp_peaks[:,2]) - max_sigma)**2 * VERY_LARGE_VALUE
        tau_penalty = max(0, np.max(np.abs(temp_peaks[:,3]/temp_peaks[:,2])) - MAX_TAU_RARIO)**2 * VERY_LARGE_VALUE
        return np.log(np.sum((y - ty)**2)) + 1.5*np.log(np.sum((temp_peaks[:,1] - new_trs)**2)) + negetive_penalty + sigma_penalty + tau_penalty

    ret = minimize(near_trs_objective, peaks.flatten(), method='Nelder-Mead')

    return ret.x.reshape(peaks.shape)