"""
    V2PropOptimizer.PropOptimizerEDM.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from time import sleep, time
from molass_legacy._MOLASS.SerialSettings import set_setting
from molass_legacy.Models.RateTheory.EDM import guess_multiple, edm_impl
from molass_legacy.KekLib.OurTkinter import Tk
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from .PropOptimizer import compute_range_rgs, PROP_MIN_VALUE, RangeRgComputer

VERY_SMALL_RG = 1

def guess_init_params(x, y):
    num_peaks = 2
    params_array = np.array(guess_multiple(x, y, num_peaks))
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
    print("compute_optimal_proportion EDM")

    progress = 0

    root = Tk.Tk()

    v2_optimizer = job_args.v2_optimizer
    xr_curve = v2_optimizer.xr_curve
    x = xr_curve.x
    y = xr_curve.y
    num_peaks = 2

    params_array = np.array(guess_multiple(x, y, num_peaks))

    def plot_model(params_array, title, rgs=None):
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title(title)
            ax.plot(x, y, color="orange")
            cy_list = []
            areas = []
            for params in params_array:
                cy = edm_impl(x, *params)
                cy_list.append(cy)
                areas.append(np.sum(cy))
                ax.plot(x, cy, ":")
            ty = np.sum(cy_list, axis=0)
            ax.plot(x, ty, ":", color="red")
            props = areas/np.sum(areas)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            tx = xmin*0.8 + xmax*0.2
            ty = ymin*0.3 + ymax*0.7
            ax.text(tx, ty, "props=%.3g, %.3g" % tuple(props))

            if rgs is not None:
                ty = ymin*0.5 + ymax*0.5
                ax.text(tx, ty, "rgs=%.3g, %.3g" % tuple(rgs))

            fig.tight_layout()
            plt.show()

    progress += 1
    progress_queue.put([progress])
    plot_model(params_array, "after guess_multiple")

    p = job_args.init_prop
    init_pv = np.array((p, 1-p))

    def objective_prop(params_vector, target_pv, return_full=False):
        cy_list = []
        areas = []
        for params in params_vector.reshape(params_array.shape):
            cy = edm_impl(x, *params)
            cy_list.append(cy)
            areas.append(np.sum(cy))
        ty = np.sum(cy_list, axis=0)
        props = np.array(areas)/np.sum(areas)
        fv = np.log(np.sum((ty - y)**2))*0.4 + np.log(max(PROP_MIN_VALUE, np.sum((props - target_pv)**2)))*0.6
        if return_full:
            return fv, cy_list
        else:
            return fv

    prop_ret = minimize(lambda x: objective_prop(x, init_pv), params_array.flatten(), method="Nelder-Mead")

    progress += 1
    progress_queue.put([progress])
    plot_model(prop_ret.x.reshape(params_array.shape), "after minimize objective_prop")

    ecurve = ElutionCurve(y)
    paired_ranges = ecurve.get_default_paired_ranges()
    print("paired_ranges=", paired_ranges)
    qv = v2_optimizer.qvector
    D = v2_optimizer.xrD
    E = v2_optimizer.xrE

    init_params = np.concatenate([[job_args.init_prop], prop_ret.x])

    # maxiter = len(init_params)*200
    maxiter = len(init_params)*20
    call_count = 0
    temp_progress = None

    rg_computer = RangeRgComputer(qv, D, E, paired_ranges)

    def objective_rdr(params, return_details=False):
        nonlocal call_count, temp_progress

        call_count += 1
        if call_count % 100 == 0:
            print("called", call_count)
            ratio = call_count/maxiter
            temp_progress = progress + int((10 - progress)*ratio)
            progress_queue.put([temp_progress])

        p = params[0]
        target_pv = np.array((p, 1-p))
        prop_fv, cy_list = objective_prop(params[1:], target_pv, return_full=True)
        C = np.array(cy_list)
        rgs = rg_computer.compute(C)

        denom = rgs[0] + rgs[1]
        if denom < VERY_SMALL_RG:
            rdr = 1
        else:
            rdr = abs(rgs[0] - rgs[1])/denom

        if return_details:
            return np.log(rdr), prop_fv, rgs
        else:
            return np.log(rdr)*0.25 + prop_fv*0.75

    t0 = time()
    rdr_ret = minimize(objective_rdr, init_params, method="Nelder-Mead", options=dict(maxiter=maxiter))
    print("it took", time() - t0)

    progress = temp_progress
    progress += 1
    progress_queue.put([progress])

    fv1, fv2, rgs = objective_rdr(rdr_ret.x, return_details=True)
    print("fv1=", fv1, "fv2=", fv2, "rgs=", rgs)
    plot_model(rdr_ret.x[1:].reshape(params_array.shape), "after minimize objective_rdr", rgs=rgs)

    progress_queue.put([-1])
