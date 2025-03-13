"""
    Optimizer.Devel.BoundsInspect.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

def bounds_inspect_impl(caller):
    canvas = caller.canvas
    optimizer = canvas.fullopt
    init_params = canvas.get_current_params()
    real_bounds = np.array(optimizer.get_param_bounds(init_params))
    print("real_bounds: ", real_bounds)

    with plt.Dp(button_spec=["OK", "Cancel"]):
        fig, ax = plt.subplots()
        fig.suptitle("Bounds Inspect")
        fig.tight_layout()
        plt.show()