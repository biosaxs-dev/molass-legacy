"""
    Optimizer.FvSynthesizer.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting

NUM_MAJOR_SCORES = get_setting("NUM_MAJOR_SCORES")
# see test_7310_FvScoreSynthesizer.py
VALUE_WEIGHTS7_OLD = [0.35, 0.1, 0.35, 0.1, 0.033, 0.033, 0.034]
VALUE_WEIGHTS7 = [0.25, 0.1, 0.25, 0.1, 0.233, 0.033, 0.034]
VALUE_WEIGHTS8_OLD = [0.35, 0.1, 0.35, 0.1, 0.025, 0.025, 0.025, 0.025]
VALUE_WEIGHTS8 = [0.25, 0.1, 0.25, 0.1, 0.225, 0.025, 0.025, 0.025]

MEAN_WEIGHT = 0.8
STDEV_WEIGHT = 0.2

def synthesize(values, positive_elevate=0):
    value_weights = VALUE_WEIGHTS7 if len(values) == 7 else VALUE_WEIGHTS8
    values = np.asarray(values)+ positive_elevate
    stds = np.std(values, axis=0)
    return MEAN_WEIGHT*np.sqrt(np.sum(value_weights * values**2)) + STDEV_WEIGHT*stds - positive_elevate

def synthesize_demo(values, positive_elevate=0):
    values = np.asarray(values)+ positive_elevate
    stds = np.std(values, axis=0)
    return MEAN_WEIGHT*np.sqrt(np.mean(values**2, axis=0)) + STDEV_WEIGHT*stds - positive_elevate

def trial_demo():
    x = np.linspace(-5, 5, 30)
    y = np.linspace(-5, 5, 30)
    xx, yy = np.meshgrid(x, y)
    zz1 = xx + yy
    zz2 = np.max([xx, yy], axis=0)
    zz3 = np.mean([xx, yy], axis=0) + np.max([xx, yy], axis=0)
    zz4 = synthesize_demo([xx, yy], positive_elevate=5)

    fig, axes = plt.subplots(ncols=4, figsize=(18,6), subplot_kw=dict(projection="3d"))
    fig.suptitle("Comparison of Score Synthesizing Functions", fontsize=20)
    ax1, ax2, ax3, ax4 = axes
    ax1.set_title(r"$z=x+y$")
    ax2.set_title(r"$z=max(x,y)$")
    ax3.set_title(r"$z=mean(x,y)+max(x,y)$")
    ax4.set_title(r"$z=\sqrt{\frac{1}{2}(x^2+y^2)} + std([x,y])$")
    ax1.plot_surface(xx, yy, zz1, cmap="jet")
    ax2.plot_surface(xx, yy, zz2, cmap="jet")
    ax3.plot_surface(xx, yy, zz3, cmap="jet")
    ax4.plot_surface(xx, yy, zz4, cmap="jet")
    zlims = []
    for ax in axes:
        zlims.append(ax.get_zlim())
    zlims = np.array(zlims)
    zmin = np.min(zlims[:,0])
    zmax = np.max(zlims[:,1])
    for ax in axes:
        ax.set_zlim(zmin, zmax)

    fig.tight_layout()
    fig.subplots_adjust(left=0.03, right=0.95, wspace=0.1)
    plt.show()

if __name__ == '__main__':
    # use test_7310_FvScoreSynthesizer.py
    # import seaborn
    # seaborn.set()
    # trial_demo()
    pass
