"""
    Trimming.SigmoidGuessX0.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import ruptures as rpt
import molass_legacy.KekLib.DebugPlot as plt

def guess_x0_impl(x, y, debug=False):

    algo = rpt.Dynp(model="l2").fit(y)
    breakpoints = algo.predict(n_bkps=4)
    i = breakpoints[0]

    if debug:
        print("breakpoints=", breakpoints)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("guess_x0_impl")
            ax.plot(x, y, label="data")
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            x0 = x[i]
            ax.plot([x0, x0], [ymin, ymax], ":", lw=2, color="red", label="x0")
            for j in breakpoints[:-1]:
                ax.axvline(x=x[j], color='yellow')
            fig.tight_layout()
            plt.show()

    return i