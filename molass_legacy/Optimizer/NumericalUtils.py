"""
    NumericalUtils.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""

import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

OUTLIER_SCALE = 2
OUTLIER_IGNORE_LIMIT = 1e-6

def safe_ratios_debug_plot(x, y, xr_ty, xr_cy_list, rg_curve, rg_params):
    return

    with plt.Dp():
        fig, ax = plt.subplots()
        axt = ax.twinx()
        axt.grid(False)

        ax.plot(x, y, color="orange")
        for cy in xr_cy_list[:-1]:
            ax.plot(x, cy, ":")

        ax.plot(x, ty, ":", color="red")

        # axt.plot(ratios, "-", color="C1")
        fig.tight_layout()
        plt.show()

def safe_ratios(ones, cy, ty, debug=False):
    ratios = cy/ty
    ratios[ty==0] = 1

    if debug:
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("safe_ratios entry")
            axt = ax.twinx()
            axt.grid(False)

            ax.plot(ty, ":")
            ax.plot(cy, ":")
            axt.plot(ratios, "-", color="C1")
            fig.tight_layout()
            plt.show()

    outliers = np.where(np.abs(ratios) > ones*OUTLIER_SCALE)[0]
    if len(outliers) == 0:
        return ratios

    # outliers appear only where both cy and ty are very small
    outlier_height = np.max(np.abs(ty[outliers]))
    if outlier_height < OUTLIER_IGNORE_LIMIT:
        widened = ty < OUTLIER_IGNORE_LIMIT
    else:
        widened = None

    if debug:
        print("outlier_height=", outlier_height)
        ratios_orig = ratios.copy()
        if widened is None:
            ratios[outliers] = 1
        else:
            ratios[widened] = 1

        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("safe_ratios outliers")

            axt = ax.twinx()
            axt.grid(False)

            ax.plot(cy)
            ax.plot(ty)

            axt.plot(ratios_orig)
            axt.plot(ratios)
            plt.show()
    else:
        if widened is None:
            ratios[outliers] = 1
        else:
            ratios[widened] = 1

    return ratios
