"""
    Optimizer.ScaleAdjustInspect.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from .RatioInterpretIllust import MODEL_NAME_DICT
from .FvScoreConverter import convert_score

USE_BH = True

def scale_adjust_inspect(caller):
    print("scale_adjust_inspect")
    js_canvas = caller.canvas

    optimizer = js_canvas.fullopt
    func_code = optimizer.get_name()
    model_name = MODEL_NAME_DICT[func_code]
    assert model_name == "SDM"
    params = js_canvas.get_current_params()
    job_name = js_canvas.dialog.get_job_info()[0]
    in_folder = get_in_folder()

    scale_indeces_list = [[0,1,2], [8,9,10]]  # 0,1,2: xr scales, 8,9,10: uv scales
    
    if USE_BH:
        temp_params = optimizer.to_norm_params(params)
    else:
        temp_params = params.copy()

    for scale_indeces in scale_indeces_list:
        print("optimizing with scale_indeces", scale_indeces)
        def scale_objective(p):
            temp_params[scale_indeces] = p
            if USE_BH:
                return optimizer.objective_func_wrapper(temp_params)
            else:
                return optimizer.objective_func(temp_params)

        if USE_BH:
            minimizer_kwargs = dict(method='Nelder-Mead')
            res = basinhopping(scale_objective, temp_params[scale_indeces], niter=20, minimizer_kwargs=minimizer_kwargs)
        else:
            res = minimize(scale_objective, temp_params[scale_indeces], method="Nelder-Mead")
        temp_params[scale_indeces] = res.x

    if USE_BH:
        temp_params = optimizer.to_real_params(temp_params)

    with plt.Dp():
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
        ax1 = axes[0,0]
        ax2 = axes[0,1]
        ax2t = ax2.twinx()
        ax2t.grid(False)
        ax3 = axes[0,2]
        ax4 = axes[1,0]
        ax5 = axes[1,1]
        ax5t = ax5.twinx()
        ax5t.grid(False)
        ax6 = axes[1,2]
        fig.suptitle("Scale Adjust Inspection at %s with=%s of %s" % (job_name, model_name, in_folder), fontsize=20)
        fv1 = optimizer.objective_func(params, plot=True, axis_info=(fig, (ax1, ax2, ax3, ax2t)))
        fv2 = optimizer.objective_func(temp_params, plot=True, axis_info=(fig, (ax4, ax5, ax6, ax5t)))
        ax1.set_title("Current Result UV", fontsize=16)
        ax2.set_title("Current Result XR", fontsize=16)
        ax3.set_title("Current Result SV=%.3g" % convert_score(fv1), fontsize=16)
        ax4.set_title("Scale Adjusted UV", fontsize=16)
        ax5.set_title("Scale Adjusted XR", fontsize=16)
        ax6.set_title("Scale Adjusted SV=%.3g" % convert_score(fv2), fontsize=16)
        fig.tight_layout()
        plt.show()