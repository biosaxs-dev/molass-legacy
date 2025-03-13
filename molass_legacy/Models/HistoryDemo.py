"""
    HistoryDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

this_dir = os.path.dirname(os.path.abspath( __file__ ))
sys.path.append(this_dir + '/..')
import molass_legacy.KekLib
from molass_legacy.Models.ElutionCurveModels import egh, emg
from SecTheory.Edm import guess_single_edm, edm_func

def demo():
    x = np.arange(300)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,3))
    for ax in ax1, ax2, ax3:
        ax.set_axis_off()

    ax1.plot(x, egh(x, 1, 150, 20, 0), label="Gaussian")
    ax2.plot(x, egh(x, 1, 120, 20, 0), ":", label="Gaussian")
    h = emg(np.array([120]), 1, 105, 18, 20)
    ax2.plot(x, emg(x, 1/h[0], 105, 18, 20), label="EMG")
    ax2.plot(x, egh(x, 1, 120, 20, 20), label="EGH")

    egh_y = egh(x, 1, 120, 20, 20)
    edm = guess_single_edm(x, egh_y)
    params = edm.get_comp_params()
    u = 0.5
    edm_y = edm_func(x, u, *params)

    ax3.plot(x, egh_y, ":", label="EGH", color="C2")
    ax3.plot(x, edm_y, label="EDM", color="C3")

    a, b, e, Dz, cinj = params
    edm_y2 = edm_func(x, u, a/3.5, -3*b, e, Dz, cinj)
    k = np.argmax(edm_y2)
    ax3.plot(x, edm_y2/edm_y2[k], label="fronting EDM", color="C4")

    for ax in ax1, ax2, ax3:
        ax.legend()

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    demo()
