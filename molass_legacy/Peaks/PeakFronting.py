"""
    Peaks.PeakFronting.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

def has_fronting_peak(ecurve, debug=False):
    x = ecurve.x
    y = ecurve.y
    j = ecurve.max_j
    hy = ecurve.max_y/2
    hyw = np.where(y > hy)[0]
    gap = np.where(hyw[1:] - hyw[:-1] > 1)[0]
    if len(gap) > 0:
        # task: retry to find a range with no gap
        ret_judge = False
    else:
        lenL = j - hyw[0]
        lenR = hyw[-1] - j
        ret_judge = lenL > lenR

    if debug:
        print("gap=", gap)
        if len(gap) == 0:
            print("lenL, lenR=", lenL, lenR)
        print("ret_judge=", ret_judge)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("Fronting Peak Detection")
            ax.plot(x, y, label="data")
            ax.plot(x[hyw], y[hyw], 'o', label="upper half")
            fig.tight_layout()
            plt.show()

    return ret_judge