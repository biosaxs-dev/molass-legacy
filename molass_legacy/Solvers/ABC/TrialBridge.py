"""
    Solvers.ABC.TrialBridge.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Optimizer.FvScoreConverter import convert_score

def abc_trial_impl(caller):
    from importlib import reload
    import Optimizer.FuncReloadUtils
    reload(Optimizer.FuncReloadUtils)
    from molass_legacy.Optimizer.FuncReloadUtils import reload_optimizer
    print("abc_trial_impl")
    canvas = caller.canvas
    optimizer = canvas.fullopt
    init_params = canvas.get_current_params()
    temp_optimizer = reload_optimizer(optimizer, init_params=init_params)
    separete_params = temp_optimizer.split_params_simple(init_params)
    print("separete_params: ", separete_params)
    ret = temp_optimizer.solve(init_params, method="pyabc", niter=10)
    ret_params = temp_optimizer.to_real_params(ret.x)
    fv = temp_optimizer.objective_func(ret_params)
    sv = convert_score(fv)
    with plt.Dp(button_spec=["OK", "Cancel"]):
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 5))
        fig.suptitle("abc_trial_impl", fontsize=20)
        axt = ax2.twinx()
        axt.grid(False)
        ax1.set_title("UV Decomposition", fontsize=16)
        ax2.set_title("Xray Decomposition", fontsize=16)
        ax3.set_title("Objective Function Scores in SV=%.3g" % sv, fontsize=16)
        temp_optimizer.objective_func(ret_params, plot=True, axis_info=(fig, (ax1, ax2, ax3, axt)))
        fig.tight_layout()
        plt.show()