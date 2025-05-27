"""
    DebugCushion.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import molass_legacy.KekLib.DebugPlot as plt

def debug_impl(debug_info):
    print("debug_impl2")
    smoother = debug_info[0]
    x = smoother.x
    y = smoother.y
    # print(x, y)

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.show()
