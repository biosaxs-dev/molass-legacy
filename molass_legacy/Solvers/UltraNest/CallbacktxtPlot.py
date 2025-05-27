"""
    Solvers.UltraNesst.CallbacktxtPlot.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Optimizer.FvScoreConverter import convert_score

def plot_callback_txt_impl(caller):
    from importlib import reload
    import Optimizer.FuncReloadUtils
    reload(Optimizer.FuncReloadUtils)
    from molass_legacy.Optimizer.FuncReloadUtils import reload_optimizer
    print("plot_callback_txt_impl")
    caller.update_information()
    canvas = caller.canvas
    optimizer = canvas.fullopt
    fv_array, x_array = canvas.demo_info[0:2]
    index = 2
    temp_params = x_array[index]
    fv = optimizer.objective_func(temp_params)
    sv = convert_score(fv)
    with plt.Dp(button_spec=["OK", "Cancel"]):
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 5))
        fig.suptitle("plot_callback_txt_impl: index=%d" % index, fontsize=20)
        axt = ax2.twinx()
        axt.grid(False)
        ax1.set_title("UV Decomposition", fontsize=16)
        ax2.set_title("Xray Decomposition", fontsize=16)
        ax3.set_title("Objective Function Scores in SV=%.3g" % sv, fontsize=16)
        optimizer.objective_func(temp_params, plot=True, axis_info=(fig, (ax1, ax2, ax3, axt)))
        fig.tight_layout()
        plt.show()