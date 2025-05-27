# coding: utf-8
"""
    BaseSurfaceDemo.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import molass_legacy.KekLib.DebugPlot as plt
from MatrixData import simple_plot_3d
from MeasuredData import MeasuredData
from DataUtils import get_in_folder

def figure_for_doc(in_folder):
    print(in_folder)
    md = MeasuredData(in_folder)
    # md.plot()

    xd = md.xr

    plt.push()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Non-Corrected Data", fontsize=20)
    simple_plot_3d(ax, xd.data, x=xd.vector, alpha=0.3)
    fig.tight_layout()
    plt.show()
    plt.pop()
