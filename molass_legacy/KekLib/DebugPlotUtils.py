# coding: utf-8
"""

    DebugPlotUtils.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF

"""
from time import sleep
import matplotlib.pyplot as _plt
from OurTkinter import Tk
from TkUtils import adjusted_geometry
import DebugPlot as dplt
from DebugPlot import set_plot_env

root = None

def mpl_setup(use_mpl=False):
    if use_mpl:
        ret_module = _plt
    else:
        global root
        root = Tk.Tk()
        root.geometry( adjusted_geometry( root.geometry() ) )
        root.withdraw()
        set_plot_env( root )
        ret_module = dplt

    return ret_module

def mpl_teardown(use_mpl, auto):
    if use_mpl:
        if auto:
            sleep(0.5)
    else:
        global root
        if auto:
            root.after( 500, root.quit )
        else:
            root.after( 0, root.quit )
        root.mainloop()
        root.destroy()
