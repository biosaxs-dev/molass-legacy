# coding: utf-8
"""
    TwoStateSolver.py

    Copyright (c) 2019,2024, SAXS Team, KEK-PF
"""
import os
import time
import numpy as np
from scipy import optimize, stats
from lmfit import Parameters, minimize
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.SciPyCookbook import smooth
from SimpleUnfolding import proportion_folded
from SvdDenoise import get_denoised_data

class TwoStateCurves:
    def __init__(self, G, m, ab_params):
        self.G = G
        self.m = m
        self.ab_params = ab_params

    def compute_yf(self, x):
        pf = proportion_folded(self.G, self.m, x)
        af, bf = self.ab_params[0:2]
        return af*pf + bf*x*pf

    def compute_yu(self, x):
        pf = proportion_folded(self.G, self.m, x)
        pu = 1 - pf
        au, bu = self.ab_params[2:4]
        return au*pu + bu*x*pu

class TwoStateSolver:
    def __init__(self):
        pass

    def solve(self, data):
        print(data.shape)
        M = get_denoised_data(data, rank=4)
        x = np.arange(M.shape[1])

        def compute_norm(params):
            G, m = params
            # print('G, m=', G, m)
            pf = proportion_folded(G, m, x)
            pu = 1 - pf
            C = np.array([pf, x*pf, pu, x*pu])
            Cpinv = np.linalg.pinv(C)
            P = M@Cpinv
            z = np.linalg.norm(P@C - M)
            return z

        def get_init_params(G):
            G_init = G
            mx = (x[0] + x[-1])/2
            m_init = G_init/mx
            return G_init, m_init

        if True:
            t0 = time.time()
            min_norm = None
            opt_x = None
            # for g in [0, 0.1, 0.2] + list(range(1, 11)):
            # for g in [0, 0.1, 0.2, 1]:
            for g in range(1, 11):
                G_init, m_init = get_init_params(g)
                res= optimize.minimize(compute_norm, [G_init, m_init], bounds=((0,10),(0,1)))
                norm = compute_norm(res.x)
                # print([g], norm, res.x)
                if min_norm is None or norm < min_norm:
                    min_norm = norm
                    opt_x = res.x
            self.params = opt_x
            print("It took", time.time() - t0)
        else:
            from SimulatedAnnealing import SimulatedAnnealing
            G_init, m_init = get_init_params(3)
            anneal = SimulatedAnnealing()
            anneal.minimize( compute_norm, xrange=[[0, 10], [0, 1]], start=[G_init, m_init], seed=1234,
                            xconstaints=None )
            self.params = anneal.xc
        self.M = M
        self.compute_components(x)

    def compute_components(self, x):
        G, m = self.params
        print('G, m=', G, m)
        pf = proportion_folded(G, m, x)
        pu = 1 - pf
        C = np.array([pf, x*pf, pu, x*pu])
        Cpinv = np.linalg.pinv(C)
        P = self.M@Cpinv
        self.C = C
        self.P = P

    def plot_components(self, ax, data, index, title=None, axt=None, start=0):
        draw_coeff=False
        x = np.arange(start, start+data.shape[1])
        y = data[index,:]
        self.x = x
        self.y = y
        C = self.C
        pf  = C[0,:]
        xpf = C[1,:]
        pu  = C[2,:]
        xpu = C[3,:]
        af, bf, au, bu = self.P[index,:]
        yf = af*pf + bf*xpf
        yu = au*pu + bu*xpu

        if axt is None:
            axt = ax.twinx()
        axt.grid(False)

        if title is not None:
            ax.set_title(title, fontsize=16)
        ax.set_xlabel('Eno')
        ax.set_ylabel('Proportion')
        axt.set_ylabel('Intensity')

        G, m = self.params
        ax.plot(x, pf, label='$P_{F;G=%2.g,m=%.2g}$' % (G, m))
        ax.plot(x, pu, label='$P_U$')
        axt.plot(x, y, color='orange', label='given')
        axt.plot(x, yf, ':', label='$y_F$')
        axt.plot(x, yu, ':', label='$y_U$')
        axt.plot(x, yf+yu, ':', color='orange', label='$y_F+y_U$')

        bbox = (0, 0.65) if draw_coeff else None
        ax.legend(bbox_to_anchor=bbox, loc='center left', fontsize=16)
        axt.legend(loc='upper right', fontsize=16)

    def draw_components_with_bands(self, ax, data, index, title=None, start=0, detail=False):
        from matplotlib.patches import Polygon
        from molass_legacy.KekLib.OurMatplotlib import get_color
        from GeometryUtils import polygon_area_centroid

        x = np.arange(start, start+data.shape[1])
        y = data[index,:]
        self.x = x
        self.y = y
        C = self.C
        pf  = C[0,:]
        xpf = C[1,:]
        pu  = C[2,:]
        xpu = C[3,:]
        af, bf, au, bu = self.P[index,:]
        yf = af*pf + bf*xpf
        yu = au*pu + bu*xpu
        ax.set_xlabel('Eno')
        ax.set_ylabel('Proportion')
        ax.plot(x, y, color='orange', label='given')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax)

        poly_points = list(zip( x, yf )) + [(x[-1], 0), (x[0], 0)]
        f_color = get_color(0)
        f_band = Polygon( poly_points, color=f_color, alpha=0.2 )
        ax.add_patch(f_band)
        cp = polygon_area_centroid(f_band)
        ax.text(*cp, "F", color=f_color, ha='center', va='center', fontsize=50)

        if detail:
            poly_points = list(zip( x, yf )) + list( reversed( list( zip( x, yf - bf*xpf ) ) ) )
            color='pink'
            d_band = Polygon( poly_points, color=color, alpha=0.2 )
            ax.add_patch(d_band)
            cp = polygon_area_centroid(d_band)
            ax.text(*cp, "F.d.d.", color=color, ha='center', va='center', fontsize=50)

        poly_points = list(zip( x, yf )) + list( reversed( list( zip( x, yf+yu ) ) ) )
        u_color = get_color(1)
        u_band = Polygon( poly_points, color=u_color, alpha=0.2 )
        ax.add_patch(u_band)
        cp = polygon_area_centroid(u_band)
        ax.text(*cp, "U", color=u_color, ha='center', va='center', fontsize=50)

    def guess_blank_location(self, slice_):
        G, m = self.params
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
        G, m = self.params
        cp = G/m
        # print('G=', G, 'm=', m, 'cp=', cp)
        return int(cp)

    def get_modeled_curves(self, index):
        curves = TwoStateCurves(*self.params, self.P[index,:])
        return curves.compute_yf, curves.compute_yu
