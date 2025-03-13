"""
    GuinierTools.CpdDecompDirect.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting
from GuinierTools.RgCurveUtils import get_connected_curve_info
from molass_legacy.Models.Stochastic.DispersiveMonopore import guess_params_using_moments

MIN_PROP = 0.03

def compute_proportions(x, y, end_points):
    areas = []
    for i in range(len(end_points)-1):
        j = bisect_right(x, end_points[i])
        k = bisect_right(x, end_points[i+1])
        area = np.sum(y[j:k])
        areas.append(area)
    areas = np.array(areas)
    props = areas/np.sum(areas)
    props[props < MIN_PROP] = MIN_PROP
    return props/np.sum(props)

def cpd_direct_impl(editor, debug=False):
    from importlib import reload
    import GuinierTools.CpdDecompUtils
    reload(GuinierTools.CpdDecompUtils)
    from GuinierTools.CpdDecompUtils import compute_end_points
    from Estimators.SdmEstimatorImpl import guess_exec_spec, edit_to_full_sdmparams

    print("cpd_direct_impl")

    x, y, peaks, baseline = editor.xr_draw_info
    rg_curve = editor.dsets[1]

    x_, y_, rgv, qualiteis, valid_bools = get_connected_curve_info(rg_curve)

    nc = editor.get_n_components() - 1
    end_points = compute_end_points(nc, x_, rgv)
    print("end_points", end_points)
    trs = (end_points[0:-1] + end_points[1:])/2
    rg_params = []
    moments_list = []
    width_scale = 0.9     # 
    for i, tr in enumerate(trs):
        j = bisect_right(x_, tr)
        rg_params.append(rgv[j])
        M1 = tr
        M2 = ((tr - end_points[i])*width_scale)**2
        M3 = 0
        moments_list.append((M1, M2, M3))
    
    if debug:
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("cpd_direct_impl: moments")
            ax.plot(x, y)
            for M in moments_list:
                ax.axvline(x=M[0], color='green')
                s = np.sqrt(M[1])
                ax.axvspan(M[0]-s, M[0]+s, color='green', alpha=0.2)
            fig.tight_layout()
            ret = plt.show()
            if not ret:
                return

    rg_params = np.array(rg_params)
    qualities = np.ones(nc)
    props = compute_proportions(x, y, end_points)
    print("props=", props)

    in_folder = get_setting("in_folder")
    if in_folder.find("20240730") > 0 and nc == 4:
        unreliable_indeces = [3]
    else:
        unreliable_indeces = []

    # exec_spec = guess_exec_spec(peak_rgs, props, qualities)
    exec_spec = {
        "unreliable_indeces": unreliable_indeces,
        "poresize_bounds": (70, 80),
    }

    ret = guess_params_using_moments(x, y, moments_list, rg_params, qualities, props, exec_spec=exec_spec, debug=False)
    if ret is None:
        return
    
    sdm_params, corrected_rgs, bounds = ret
    optimizer = editor.optimizer

    init_params = edit_to_full_sdmparams(editor, sdm_params, corrected_rgs, optimizer.uv_curve, debug=False)
    if init_params is None:
        return
    
    if True:
        with plt.Dp(button_spec=["OK", "Cancel"]):
            fig, ax = plt.subplots()
            ax.set_title("cpd_direct_impl: result")
            axt = ax.twinx()
            axt.grid(False)
            axes = (None, ax, None, axt)
            optimizer.objective_func(init_params, plot=True, axis_info=(fig, axes))
            fig.tight_layout()
            ret = plt.show()
            if not ret:
                return

    editor.draw_scores(init_params, create_new_optimizer=False)