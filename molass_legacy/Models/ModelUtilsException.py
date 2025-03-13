"""

    Models.ModelUtilsException.py

    Copyright (c) 2024, SAXS Team, KEK-PF

"""
import molass_legacy.KekLib.DebugPlot as plt

def get_range_exceptionally(x, model, params, debug=False):
    print("get_range_exceptionally: params=", params)
    y = model(x, params)
    h, m, s, t_ = params[0:4]
    f = max(x[0], m-2*s)
    t = min(x[-1], m+2*s)

    if debug:
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("get_range_exceptionally")
            ax.plot(x, y)
            ax.axvspan(f, t, alpha=0.5)
            plt.show()

    return f, t