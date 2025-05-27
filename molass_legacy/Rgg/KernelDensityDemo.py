# coding: utf-8
"""
    Rgg.KernelDensityDemo.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from pomegranate import GaussianKernelDensity
from .KernelDensityFix import UniformKernelDensity
import molass_legacy.KekLib.DebugPlot as plt

def demo():
    x = np.linspace(0, 50, 1000)

    fig, axes = plt.subplots(ncols=2, figsize=(14,7))

    for k, args in enumerate([
        (GaussianKernelDensity([30], bandwidth=1), 'Gaussian Kernel Density'),
        (UniformKernelDensity([30], bandwidth=1), 'Uniform Kernel Density'),
        ]):
        ax = axes[k]
        d, name = args
        ax.set_title(name)
        y = d.probability(x)
        ax.fill_between(x, 0, y, color='c')

    fig.tight_layout()
    plt.show()
