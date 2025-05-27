# coding: utf-8
"""
    UnifiedModel.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from .GaussianMixture import gm_curve
from molass_legacy.KekLib.OurMatplotlib import get_color
import molass_legacy.KekLib.DebugPlot as plt

SHOW_PROPORTIONS = False

def tutorial():

    if SHOW_PROPORTIONS:
        nrows = 2
        fighight = 10
    else:
        nrows = 1
        fighight = 5

    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(18, fighight))
    fig.suptitle("Unified Modelling of SEC-SAXS", fontsize=20)

    u_title_list = [
            "Source Distribution caused from Rg",
            "Apparant Distribution in Absorbance depending on Wavelength",
            "Apparant Distribution in Scattering depending on Angle"
            ]
    d_title_list = [
            "Component Proportions caused from Rg",
            "Component Proportions in Absorbance",
            "Component Proportions in Scattering"
            ]

    if SHOW_PROPORTIONS:
        uaxes = axes[0,:]
        daxes = axes[1,:]
        title_y = 1.02
    else:
        uaxes = axes
        daxes = [None]*3
        title_y = 1.01

    for axu, u_title, axd, u_title in zip(uaxes, u_title_list, daxes, d_title_list):
        axu.set_title(u_title, fontsize=14, y=title_y)
        if axd:
            axd.set_title(u_title, fontsize=14, y=title_y)

    x_length_list = [100, 600, 300]

    params_list = [
            [(100/3, 30/3, 1), (200/3, 20/3, 1)],
            [(200, 60, 0.7), (400, 40, 1.2)],
            [(100, 30, 1.2), (200, 20, 0.5)],
            ]

    min_ymin = np.inf
    max_ymax = -np.inf
    for axu, axd, length, params in zip(uaxes, daxes, x_length_list, params_list):
        x, y, gy = gm_curve(x_length=length, params=params, ret_detail=True)
        scale = length/400
        axu.plot(x, y*scale)
        for k, gy_ in enumerate(gy):
            axu.plot(x, gy_*scale, ':')
            if axd:
                axd.plot(x, gy_/y, ':', color=get_color(k+1))

        ymin, ymax = axu.get_ylim()
        min_ymin = min(min_ymin, ymin)
        max_ymax = max(max_ymax, ymax)

    for axu in uaxes:
        axu.set_ylim(min_ymin, max_ymax)

    fig.tight_layout()
    fig.subplots_adjust(top=0.91 if SHOW_PROPORTIONS else 0.85)

    plt.show()
