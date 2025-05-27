# coding: utf-8
"""
    ThreeStateDimerSolver.py

    Copyright (c) 2019,2024, SAXS Team, KEK-PF
"""
import os
import time
import numpy as np
from scipy import optimize, stats
from lmfit import Parameters, minimize
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.SciPyCookbook import smooth
from SimpleUnfolding import RT
from SvdDenoise import get_denoised_data

def compute_constant_k(G, m, x):
    return np.exp((-G + m*x)/RT)

def compute_C(params, x):
    G1, m1, G2, m2, Pt = params
    K1 = compute_constant_k(G1, m1, x)
    K2 = compute_constant_k(G2, m2, x)
    Fd = (-K1*K2 + np.sqrt((K1*K2)**2 + 8*(1+K1)*K1*K2*Pt))/(4*Pt*(1+K1))

    C = np.array([2*Pt*Fd**2/(K1*K2), 2*Pt*Fd**2/K2, Fd])
    return C

class ThreeStateCurves:
    def __init__(self, params, ab_params):
        self.params = params
        self.ab_params = ab_params

    def compute_yn(self, x):
        self.C = compute_C(self.params, x)
        an = self.ab_params[0]
        return an*self.C[0,:]

    def compute_yi(self, x):
        ai = self.ab_params[1]
        return ai*self.C[1,:]

    def compute_yu(self, x):
        au = self.ab_params[2]
        return au*self.C[2,:]

class ThreeStateDimerSolver:
    def __init__(self):
        pass

    def solve(self, data):
        print(data.shape)
        M = get_denoised_data(data, rank=3)
        x = np.arange(M.shape[1])

        def compute_norm(params):
            C = compute_C(params, x)
            Cpinv = np.linalg.pinv(C)
            P = M@Cpinv
            z = np.linalg.norm(P@C - M)
            return z

        def get_init_params(G):
            G_init = G
            mx1 = (x[0]*2 + x[-1])/3
            m1_init = G_init/mx1
            mx2 = (x[0] + x[-1]*2)/3
            m2_init = G_init/mx2
            return G_init, m1_init, m2_init

        t0 = time.time()
        min_norm = None
        opt_x = None
        for g in range(0, 6):
            G_init, m1_init, m2_init = get_init_params(g)
            try:
                res= optimize.minimize(compute_norm, [G_init, m1_init, G_init, m2_init, 1], bounds=((0,10),(0,1),(0,10),(0,1),(0,10)))
            except:
                print([g], 'error')
                continue
            norm = compute_norm(res.x)
            print([g], norm, res.x)
            if min_norm is None or norm < min_norm:
                min_norm = norm
                opt_x = res.x
        self.params = opt_x
        print("It took", time.time() - t0)

        self.M = M
        self.compute_components(x)

    def compute_components(self, x):
        C = compute_C(self.params, x)
        Cpinv = np.linalg.pinv(C)
        P = self.M@Cpinv
        self.C = C
        self.P = P

    def plot_components(self, ax, data, index, title=None, axt=None, start=0):
        x = np.arange(data.shape[1]) + start
        y = data[index,:]
        self.x = x
        self.y = y
        C = self.C
        pn = C[0,:]
        pi = C[1,:]
        pu = C[2,:]
        an, ai, au = self.P[index,:]

        yn = an*pn
        yi = ai*pi
        yu = au*pu

        if axt is None:
            axt = ax.twinx()
        axt.grid(False)

        if title is not None:
            ax.set_title(title, fontsize=16)
        ax.set_xlabel('Eno')
        ax.set_ylabel('Proportion')
        axt.set_ylabel('Intensity')

        ax.plot(x, pn, label='$P_n$')
        ax.plot(x, pi, label='$P_i$')
        ax.plot(x, pu, label='$P_u$')

        axt.plot(x, y, color='orange', label='given')
        axt.plot(x, yn, ':', label='$y_n$')
        axt.plot(x, yi, ':', label='$y_i$')
        axt.plot(x, yu, ':', label='$y_u$')
        axt.plot(x, yn+yi+yu, ':', color='orange', label='$y_n+y_i+y_u$')

        ax.legend(bbox_to_anchor=(0, 0.5), loc='center left', fontsize=16)
        axt.legend(loc='upper right', fontsize=16)

    def draw_components_with_bands(self, ax, data, index, title=None, start=0, detail=False):
        """
        detail=False is not used here. It is just for interface compatibility with TwoStateSolver
        """
        from matplotlib.patches import Polygon
        from molass_legacy.KekLib.OurMatplotlib import get_color
        from GeometryUtils import polygon_area_centroid

        x = np.arange(data.shape[1]) + start
        y = data[index,:]
        self.x = x
        self.y = y
        C = self.C
        pn = C[0,:]
        pi = C[1,:]
        pu = C[2,:]
        an, ai, au = self.P[index,:]

        ax.set_xlabel('Eno')
        ax.set_ylabel('Proportion')

        ax.plot(x, y, color='orange', label='given')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax)

        yn = an*pn
        yi = ai*pi
        yu = au*pu

        poly_points = list( zip( x, yn ) ) + [ (x[-1], 0), (x[0], 0) ]
        n_color = get_color(0)
        n_band = Polygon( poly_points, color=n_color, alpha=0.2 )
        ax.add_patch(n_band)
        cp = polygon_area_centroid(n_band)
        ax.text(*cp, "N", color=n_color, ha='center', va='center', fontsize=50)

        yn_yi = yn+yi
        poly_points = list(zip( x, yn )) + list( reversed( list( zip( x, yn_yi ) ) ) )
        i_color = get_color(1)
        i_band = Polygon( poly_points, color=i_color, alpha=0.2 )
        ax.add_patch(i_band)
        cp = polygon_area_centroid(i_band)
        ax.text(*cp, "I", color=i_color, ha='center', va='center', fontsize=50)

        yn_yi_yu = yn_yi + yu
        poly_points = list(zip( x, yn_yi )) + list( reversed( list( zip( x, yn_yi_yu ) ) ) )
        u_color = get_color(2)
        u_band = Polygon( poly_points, color=u_color, alpha=0.2 )
        ax.add_patch(u_band)
        cp = polygon_area_centroid(u_band)
        ax.text(*cp, "U", color=u_color, ha='center', va='center', fontsize=50)

    def get_modeled_curves(self, index):
        curves = ThreeStateCurves(self.params, self.P[index,:])
        return curves.compute_yn, curves.compute_yi, curves.compute_yu

    def get_imermediate_topx(self):
        G2, m2 = self.params[2:4]
        return G2/m2
