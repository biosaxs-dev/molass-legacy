"""
    SecTheory.EdmSpike.py

    EDM - Equilibrium Dispersive Model - Spike

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
import io
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import seaborn
seaborn.set()

import molass_legacy.KekLib.DebugPlot as plt
from DataUtils import get_in_folder
from .Edm import guess_multiple_edms

def guess_edm_initial_params(x, y, return_edms=False):

    slice_ = slice(200, 300)
    edms_ = guess_multiple_edms(x[slice_], y[slice_], 1)
    edm1 = edms_[0]

    slice_ = slice(300, 350)
    edms_ = guess_multiple_edms(x[slice_], y[slice_], 1)
    edm2 = edms_[0]

    y_ = y - edm1(x) - edm2(x)
    y_[y_ < 0] = 0

    slice_ = slice(100, 200)
    edms_ = guess_multiple_edms(x[slice_], y_[slice_], 1)
    edm0 = edms_[0]
    edms = [edm0, edm1, edm2]

    if return_edms:
        return edms

    xr_params = []
    for edm in edms:
        xr_params.append(edm.get_comp_params())

    return np.array(xr_params)

def spike(in_folder, sd, num_components):
    from .Edm import edm_func

    print("spike_impl", in_folder, num_components)

    D, E, qv, xr_curve = sd.get_xr_data_separate_ly()

    x = xr_curve.x
    y = xr_curve.y
    params = guess_edm_initial_params(x, y, return_edms=False)

    u = 0.5

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_title("EDM initial fitting to %s" % get_in_folder(in_folder), fontsize=20)
        ax.plot(x, y, color="orange")

        ty = np.zeros(len(y))
        for k, p in enumerate(params):
            print([k], *p)
            cy = edm_func(x, u, *p)
            ty += cy
            ax.plot(x, cy, ":", label="component-%d" % (k+1))

        ax.plot(x, ty, ":", color="red", lw=2, label="component total")

        ax.legend(loc="upper left", fontsize=16)

        fig.tight_layout()
        plt.show()
