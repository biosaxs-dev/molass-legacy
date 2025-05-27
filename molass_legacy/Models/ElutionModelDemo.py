"""

    ElutionModelDemo.py

    Copyright (c) 2017-2023, SAXS Team, KEK-PF

"""
import numpy as np
from matplotlib.gridspec import GridSpec
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Mapping.SingleComponent import PEAK_EVAL_RANGE_RATIO

def demo(caller):
    print("ElutionModelDemo")
    editor = caller.editor
    opt_recs = editor.opt_recs
    print(len(opt_recs))

    rec = opt_recs[1]

    x = editor.x
    y = editor.y

    x_ = x - x[0]
    cy = rec.evaluator(x_)

    m = np.argmax(cy)
    xL = x[0:m]
    cyL = cy[0:m]
    xR = x[m:]
    cyR = cy[m:]

    with plt.Dp():
        fig = plt.figure(figsize=(12,6))
        fig.suptitle("EGHA, EMGA Inverse Spline Demo", fontsize=20)
        gs = GridSpec(7,2)
        ax0 = fig.add_subplot(gs[:,0])
        ax1 = fig.add_subplot(gs[0:4,1])
        ax2 = fig.add_subplot(gs[4:,1])

        ax0.set_title("Attending Element", fontsize=16)
        ax1.set_title("Left Side Inverse", fontsize=16)
        ax2.set_title("RIght Side Inverse", fontsize=16)

        ax0.plot(x, y)
        ax0.plot(xL, cyL, ":", lw=3)
        ax0.plot(xR, cyR, ":", lw=3)

        xmin, xmax = ax0.get_xlim()
        ax0.set_xlim(xmin, xmax)
        yf = cy[m]*PEAK_EVAL_RANGE_RATIO
        ax0.plot([xmin, xmax], [yf, yf], color="red")

        ax1.plot(cyL, xL, ":", lw=3, color="C1")
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax)
        ax1.plot([yf, yf], [ymin, ymax], color="red")

        ax2.plot(-cyR, xR, ":", lw=3, color="C2")
        ymin, ymax = ax2.get_ylim()
        ax2.set_ylim(ymin, ymax)
        ax2.plot([-yf, -yf], [ymin, ymax], color="red")

        fig.tight_layout()
        plt.show()
