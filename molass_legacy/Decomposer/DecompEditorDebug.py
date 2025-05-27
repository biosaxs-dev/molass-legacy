"""
    DecompEditorDebug.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

def debug_plot_impl(frame, dialog):

    x = frame.x
    fx = frame.fx
    y = frame.y
    uv_y = frame.uv_y
    j0 = frame.xr_j0

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

        ty = np.zeros(len(x))
        ax1.plot(x, uv_y)
        for k, rec in enumerate(frame.opt_recs_uv):
            # print([k], "str(rec)=", str(rec))
            func = rec[1]
            cy = func(fx)
            ax1.plot(x, cy, ":")
            ty += cy

        ax1.plot(x, ty, ":", color="red", lw=2)

        ax2.plot(x, y)

        ty = np.zeros(len(x))
        for k, rec in enumerate(frame.opt_recs):
            print([k], "str(rec)=", str(rec))
            # func = rec[1]
            func = rec.evaluator
            cy = func(fx, debug=True)
            ax2.plot(x, cy, ":")
            ty += cy

        ax2.plot(x, ty, ":", color="red", lw=2)

        fig.tight_layout()
        plt.show()
