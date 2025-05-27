"""
    SecTheory.T0UpperBound.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt
from GeometryUtils import rotated_argmin
from molass_legacy._MOLASS.SerialSettings import get_setting

SMALL_RATIO = 0.005
SMALL_ANGLE = -np.pi/24

def estimate_t0upper_bound(ecurve, debug=False):
    x = ecurve.x
    y = ecurve.y

    m = ecurve.peak_info[0][1]
    y_ = y[0:m]
    i = rotated_argmin(SMALL_ANGLE, y_)

    cy = np.cumsum(y)
    j = bisect_right(cy, cy[-1]*SMALL_RATIO)

    k = min(i, j)

    if debug:
        from PlotUtils import align_yaxis_np
        from DataUtils import get_in_folder

        in_folder = get_in_folder()

        with plt.Dp():
            fig, ax1 = plt.subplots()
            axt = ax1.twinx()
            axt.grid(False)
            fig.suptitle("T0Upper Bound Debug Plot for %s" % in_folder, fontsize=20)
            ax1.plot(x, y)
            ax1.plot(x[m], y[m], "o", color="green")
            ax1.plot(x[i], y[i], "o", color="red")
            ymin, ymax = ax1.get_ylim()
            ax1.set_ylim()
            ax1.plot(x[[k,k]], [ymin, ymax], color="yellow")
            axt.plot(x, cy, color="C1")
            axt.plot(x[j], cy[j], "o", color="cyan")
            align_yaxis_np(ax1, axt)
            fig.tight_layout()
            plt.show()

    return int(x[k])    # int(...) is meant to avoid str(...) to become like 'np.int64(...)' in a larger RAM environment
