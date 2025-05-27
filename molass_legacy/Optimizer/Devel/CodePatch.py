"""
    Optimizer.Devel.CodePatch.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

def code_patch(optimizer, work_params, current_folder):
    from importlib import reload
    import Optimizer.FuncReloadUtils
    reload(Optimizer.FuncReloadUtils)
    from molass_legacy.Optimizer.FuncReloadUtils import reload_optimizer

    temp_optimizer = reload_optimizer(optimizer, init_params=work_params)

    restart_params_file = os.path.join(current_folder, "restart_params.txt")

    def save_params():
        np.savetxt(restart_params_file, work_params)

    def load_params():
        temp_params = np.loadtxt(restart_params_file)
        print("load_params: temp_params=", temp_params)
        for i, val in enumerate(temp_params):
            work_params[i] = val

    extra_button_specs = [
        ("save_params", save_params),
        ("load_params", load_params),
    ]

    with plt.Dp(button_spec=["OK", "Cancel"], extra_button_specs=extra_button_specs):
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))
        fig.suptitle("Code Patch Preview")

        axt = ax2.twinx()
        axt.grid(False)
        axes = (ax1, ax2, ax3, axt)
        temp_optimizer.objective_func(work_params, plot=True, axis_info=(fig, axes))

        fig.tight_layout()
        plt.show()