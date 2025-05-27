"""
    OurManim3D.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
# import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import molass_legacy.KekLib.DebugPlot as plt
from OurManim import manim_init

def spike():
    manim_init()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(0, 0)
    # ax.set_axis_off()
    ax.text(0.2, 0.2, 0.5, r"$ v $", ha="center", va="center", fontsize=100)
    ax.text(0.5, 0.5, 0.5, r"$ = $", ha="center", va="center", fontsize=100)
    ax.text(0.8, 0.8, 0.5, r"$ u $", ha="center", va="center", fontsize=100)
    plt.show()

class TextGroup3D:
    def __init__(self):
        pass
