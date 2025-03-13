# coding: utf-8
"""
    SaxsCurve.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
import numpy as np
import matplotlib.pyplot as plt

class SaxsCurve:
    def __init__(self, fig, ax, x, in_y, r_y):
        ax.set_title("Scattering Curve", fontsize=20)

        ax.set_xlabel('Q', fontsize=16)
        ax.set_ylabel('Log(I)', fontsize=16)

        scale = 1 if in_y is None else in_y[0]/r_y[0]
        ax.plot(x, np.log10(r_y*scale), label='simulated')
        if in_y is not None:
            ax.plot(x, np.log10(in_y), ':', label='experiment')

        ax.legend(fontsize=16)
