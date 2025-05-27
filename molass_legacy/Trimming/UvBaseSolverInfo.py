"""
    UvBaseSolverInfo.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import curve_fit
import molass_legacy.KekLib.DebugPlot as plt
from .Sigmoid import ex_sigmoid
from molass_legacy.Peaks.EghSupples import d_egh

TRY_SPECIAL_SIGMOID = False

def make_info_lists(fc, debug=False):
    egh_param_list = fc.a_curve.get_model_param_list()
    sigmoid_param_list = guess_better_sigmoid_params(fc, debug=debug)
    d_egh_param_list = guess_distortion_params(fc, egh_param_list, sigmoid_param_list, debug=debug)
    return egh_param_list, sigmoid_param_list, d_egh_param_list

def guess_better_sigmoid_params(fc, debug=False):
    sigmoid_param_list = [popt for popt in fc.popts if popt is not None]

    if len(sigmoid_param_list) == 0 and TRY_SPECIAL_SIGMOID:
        # as in 20231214/PgpApo
        try:
            if debug:
                from importlib import reload
                import Trimming.SpecialSigmoid
                reload(Trimming.SpecialSigmoid)
            from .SpecialSigmoid import guess_special_sigmoid_params
            sigmoid_param_list = guess_special_sigmoid_params(fc, debug=debug)
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(None, "guess_special_sigmoid_params failed: ")

    num_sigmoids = len(sigmoid_param_list)
    assert num_sigmoids > 0                 # num_sigmoids == 0 case must fail
    num_sigmoid_params = 6

    x = fc.a_curve2.x
    y = fc.a_curve2.y

    def sigmoid_chain_func(x, *p):
        cy = np.zeros(len(x))
        start = 0
        for k in range(num_sigmoids):
            stop = start+num_sigmoid_params
            cy += ex_sigmoid(x, *p[start:stop])
            start = stop
        return cy

    p0 = np.array(sigmoid_param_list).flatten()

    if debug:
        print("sigmoid_param_list=", sigmoid_param_list)
        print("p0=", p0)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("guess_better_sigmoid_params before")
            ax.plot(x, y)
            ax.plot(x, sigmoid_chain_func(x, *p0))
            plt.show()

    popt, pcov = curve_fit(sigmoid_chain_func, x, y, p0)

    if debug:
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("guess_better_sigmoid_params after")
            ax.plot(x, y)
            ax.plot(x, sigmoid_chain_func(x, *popt))
            plt.show()

    ret_list = []
    start = 0
    for k in range(num_sigmoids):
        stop = start + num_sigmoid_params
        ret_list.append(popt[start:stop].copy())
        start = stop
    return ret_list

def guess_distortion_params(fc, egh_param_list, sigmoid_param_list, debug=False):
    num_peaks = len(egh_param_list)
    num_egh_params = 4

    x = fc.a_curve2.x
    y = fc.a_curve2.y

    y_ = y.copy()
    for params in sigmoid_param_list:
        y_ -= ex_sigmoid(x, *params)

    if debug:
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("guess_distortion_params before")
            ax.plot(x, y)
            ax.plot(x, y_)
            plt.show()

    egh_param_list_ = [prm.copy()for prm in egh_param_list]
    def distortion_curve_func(x, *p):
        cy = np.zeros(len(x))
        for k in range(num_peaks):
            params = egh_param_list_[k]
            params[0] = p[k]
            cy += d_egh(x, *params)
        return cy

    p0_list = []
    for params in egh_param_list:
        # change H
        p0_list.append(-params[0]*0.5)      # it should be of opposite sign

    p0 = np.array(p0_list)
    popt, pcov = curve_fit(distortion_curve_func, x, y_, p0)

    if debug:
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("guess_distortion_params after")
            ax.plot(x, y)
            ax.plot(x, y_)
            ax.plot(x, distortion_curve_func(x, *popt))
            plt.show()

    return popt
