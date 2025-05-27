# coding: utf-8
"""
    Uv2dLpm.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.Baseline.ScatteringBaseline import ScatteringBaseline
import molass_legacy.KekLib.DebugPlot as plt

class Uv2dLpm:
    def __init__(self, a_vector, iy):
        self.iy = iy
        y = a_vector[iy]
        sbl = ScatteringBaseline(y)
        A, B = sbl.solve()

        if False:
            x = np.arange(iy.start, iy.stop)
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(a_vector)
            ax.plot(x, y)
            ax.plot(x, A*x + B)
            fig.tight_layout()
            plt.show()

        self.params = (A, B)

    def get_params(self):
        # return params interpreted as in 3d
        A, B = self.params
        b = B - self.iy.start*A
        return 0, A, b
