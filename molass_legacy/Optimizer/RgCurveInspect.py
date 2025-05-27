"""
    Optimizer.RgCurveInspect.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from .RatioInterpretIllust import MODEL_NAME_DICT
from .FvScoreConverter import convert_score

def rg_curve_inspect(caller):
    from importlib import reload
    import Optimizer.FuncReloadUtils
    reload(Optimizer.FuncReloadUtils)
    from molass_legacy.Optimizer.FuncReloadUtils import reload_optimizer

    print("rg-curve inspect")
    js_canvas = caller.canvas
    optimizer = reload_optimizer(js_canvas.fullopt)
    params = js_canvas.get_current_params()
    lrf_info = optimizer.objective_func(params, return_lrf_info=True, debug=True)   # this sets optimizer.separate_params to update rg_params

    optimizer.update_guinier_region(params=params, debug=True)     # in order to reload GuinierDeviation, which is currently under revision

    job_name = js_canvas.dialog.get_job_info()[0]
    in_folder = get_in_folder()

    Pxr = lrf_info.matrices[0]
    Cxr = lrf_info.matrices[1]
    rg_params = optimizer.separate_params[2]
    
    Guinier_deviation = optimizer.get_guinier_deviation(Pxr, Cxr, rg_params, debug=True)
    if Guinier_deviation is None:
        return
    print("Guinier_deviation=%.3g" % Guinier_deviation)

    fv = optimizer.objective_func(params)
    sv = convert_score(fv)

    with plt.Dp():
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 5))
        ax1.set_title("UV Decomoposition", fontsize=16)
        ax2.set_title("Xray Decomposition", fontsize=16)
        ax3.set_title("Objecive Function Scores in SV=%.3g" % sv, fontsize=16)
        axt = ax2.twinx()
        axt.grid(False)

        optimizer.objective_func(params, plot=True, axis_info=(fig, (ax1, ax2, ax3, axt)))

        fig.tight_layout()
        plt.show()