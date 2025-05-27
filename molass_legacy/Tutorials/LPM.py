# coding: utf-8
"""

    Tutorials.LPM.py

    Copyright (c) 2020, SAXS Team, KEK-PF

"""
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.DataStructure.LPM import LPM_3d
from MatrixData import simple_plot_3d

"""
    borrowed at https://stackoverrun.com/ja/q/6260216
    See Also https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.FancyArrowPatch.html
"""
class Arrow3D(FancyArrowPatch):
       def __init__(self, xs, ys, zs, *args, **kwargs):
           FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
           self._verts3d = xs, ys, zs

       def draw(self, renderer):
           xs3d, ys3d, zs3d = self._verts3d
           xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
           self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
           FancyArrowPatch.draw(self, renderer)

def demo_3d(xr):
    ecurve_y = xr.e_curve.y
    lpm = LPM_3d(xr.data, ecurve_y=ecurve_y, debug=True)

    q = xr.vector

    print('xr.data.shape=', xr.data.shape, 'q.shape=', q.shape)

    plt.push()
    fig = plt.figure(figsize=(48, 24))
    gs = GridSpec(2,3)
    ax00 = fig.add_subplot(gs[0,0], projection='3d')
    ax01 = fig.add_subplot(gs[0,1], projection='3d')
    ax02 = fig.add_subplot(gs[0,2], projection='3d')
    ax11 = fig.add_subplot(gs[1,1], projection='3d')
    ax12 = fig.add_subplot(gs[1,2], projection='3d')
    axes = [ax00, ax01, ax02, ax11, ax12]

    scale, elev = 101, -3   # for 20200630_11
    # scale, elev = 51, -2.2    # for 20200630_12

    simple_plot_3d(ax00, xr.data, x=q, edgecolors='cyan', zorder=0)

    # kwargs = {'rcount':5, 'ccount':5}
    kwargs = {}
    corrected = lpm.data.copy()
    simple_plot_3d(ax01, corrected, x=q, edgecolors='cyan', zorder=0)

    BP = lpm.adjust_with_mf(xr.e_index, xr.e_curve)
    simple_plot_3d(ax02, lpm.data, x=q, edgecolors='cyan')

    bp_min = np.min(BP)
    bp_max = np.max(BP)
    print(bp_min, bp_max)

    w = -scale
    zmin = bp_min*(1 - w) + bp_max*w
    w = scale
    zmax = bp_min*(1 - w) + bp_max*w

    fig.tight_layout()
    fig.subplots_adjust(top=0.95, hspace=0.1)

    titles = ["Not corrected", "LPM corrected", "BPA corrected", "LPM corrected (Zoomed)", "BPA corrected  (Zoomed)"]

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=50)

    for ax in [ax11, ax12]:
        ax.set_zlim(zmin, zmax)

    simple_plot_3d(ax11, corrected, x=q, edgecolors='cyan', alpha=0.1, zorder=0, **kwargs)
    simple_plot_3d(ax12, lpm.data, x=q, edgecolors='cyan', alpha=0.1, zorder=0, **kwargs)

    for ax in axes[0:3]:
        ax.view_init(20, -30)
    for ax in axes[3:]:
        ax.view_init(elev, -30)

    ZP = np.zeros(BP.shape)

    simple_plot_3d(ax11, ZP, x=q, edgecolors='pink', color='red', alpha=0.3, zorder=1, **kwargs)
    simple_plot_3d(ax11, BP, x=q, edgecolors='green', color='green', alpha=0.3, zorder=1, **kwargs)
    simple_plot_3d(ax12, ZP, x=q, edgecolors='pink', color='red', alpha=0.3, zorder=1, **kwargs)

    je = len(ecurve_y)-1
    arrow_scale = 1
    for x_, y_, z_ in [ ([q[0],q[0]], [0,0], [BP[0,0]*arrow_scale, 0]),
                        ([q[0],q[0]], [je,je], [BP[0,-1]*arrow_scale, 0]),
                        ([q[-1],q[-1]], [0,0], [BP[-1,0]*arrow_scale, 0]),
                        ([q[-1],q[-1]], [je,je], [BP[-1,-1]*arrow_scale, 0]),
                        ]:
        arrow = Arrow3D(x_, y_, z_,
                        mutation_scale=10, lw=5, arrowstyle="-|>", color="r",
                        shrinkA=0, shrinkB=0, zorder=1)
        ax11.add_artist(arrow)

    plt.show()
    plt.pop()
