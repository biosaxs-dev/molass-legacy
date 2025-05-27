# coding: utf-8
"""
    SimpleSolver.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
import numpy as np
from scipy import optimize, stats
from lmfit import Parameters, minimize
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.SciPyCookbook import smooth
from SimpleUnfolding import SimpleUnfolding

MAX_ITERATION_FMIN_CG = 10000
RT = 1

class SimpleSolver:
    def __init__(self):
        pass

    def solve(self, data, index=None):
        print(data.shape)
        self.data = data
        self.guess_init_C(index)

    def guess_init_C(self, index):
        self.x = x = np.arange(self.data.shape[1])
        self.y = y = self.data[index,:]
        sy = smooth(y)
        self.model = SimpleUnfolding()
        res = self.model.fit(x, sy, G_init=10)     # 0.1, 1, 10
        self.params = res.params

    def plot_guess(self, ax, title=None):
        # print('params=', [(k, p.value) for k, p in self.params.items()])

        x = self.x
        y = self.y
        pf = self.model.compute_Pf(self.params, x)
        yf = self.model.compute_yf(self.params, pf, x)
        yu = self.model.compute_yu(self.params, pf, x)

        axt = ax.twinx()
        axt.grid(False)

        if title is not None:
            ax.set_title(title, fontsize=16)
        ax.set_xlabel('Eno')
        ax.set_ylabel('Proportion')
        axt.set_ylabel('Intensity')

        ax.plot(x, pf, label='$P_F$')
        axt.plot(x, y, color='orange', label='given')
        axt.plot(x, yf, ':', label='$y_F$')
        axt.plot(x, yu, ':', label='$y_U$')
        axt.plot(x, yf+yu, ':', color='orange', label='$y_F+y_U$')

        ax.legend(bbox_to_anchor=(0, 0.7), loc='center left', fontsize=16)
        axt.legend(loc='upper right', fontsize=16)

    def guess_blank_location(self, slice_):
        G = self.params['G']
        m = self.params['m']
        cx = G/m
        xsize = slice_.stop - slice_.start
        pos_ratio = cx/xsize
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.x, self.y)

        # print('pos_ratio=', pos_ratio, 'slope=', slope)

        if pos_ratio < 0.4  or (pos_ratio < 0.6 and slope > 0):
            side = 'right'
        else:
            side = 'left'
        return side

    def get_crossing_point(self):
        G = self.params['G']
        m = self.params['m']
        cp = G/m
        # print('G=', G, 'm=', m, 'cp=', cp)
        return int(cp)

    def get_fit_model(self):
        return SimpleUnfolding(params=self.params)
