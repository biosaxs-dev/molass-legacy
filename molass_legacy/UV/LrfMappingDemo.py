"""
    UV.LrfMappingDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt
from  MatrixData import simple_plot_3d
import MatplotlibUtils

def demo(caller):
    print("demo")
    dsets = caller.optinit_info.dsets
    optimizer = caller.fullopt
    params = caller.canvas.get_current_params()
    split_params = optimizer.split_params_simple(params)
    lrf_info = optimizer.objective_func(params, return_lrf_info=True, lrf_debug=False)
    Pxr, Cxr, Puv, Cuv, mapped_UvD = lrf_info.matrices

    uvD = optimizer.uvD
    wv = optimizer.wvector

    assert Cxr.shape[1] == Cuv.shape[1]
    elen = Cxr.shape[1]
    a, b = split_params[3]
    jxr = np.arange(elen) + optimizer.xr_curve.x[0]
    juv = a*jxr + b

    for M in (Pxr, Cxr, Puv, Cuv, mapped_UvD):
        print(M.shape)

    i280 = bisect_right(wv, 280)

    def plot_components(title, ax, ecurve, j_, P, C):
        ax.set_title(title)
        ax.plot(ecurve.x, ecurve.y)
        cy_list = []
        for k, c in enumerate(C):
            cy = c*P[i280,k]
            cy_list.append(cy)
            ax.plot(j_, cy, ":")
        ty = np.sum(cy_list, axis=0)
        ax.plot(j_, ty, ":", color="red")

    def plot_foldedness(title, ax, wv, P):
        ax.set_title(title)
        fy_list = []
        for k, p in enumerate(P.T[0:-1]):
            f = foldedness.compute(p)
            ax.plot(wv, p, ":", label="component-%d, Foldedness=%.1f" % (k+1, f))
        ax.legend()

    uv_curve = optimizer.uv_curve
    xr_curve = optimizer.xr_curve

    wv = optimizer.wvector
    xr_x = xr_curve.x
    uv_x = a*xr_x + b

    with plt.Dp():
        fig = plt.figure(figsize=(12,10))
        ax00 = fig.add_subplot(221)
        ax01 = fig.add_subplot(222)
        ax10 = fig.add_subplot(223, projection="3d")
        ax11 = fig.add_subplot(224, projection="3d")
        fig.suptitle("LRF Mapping Demo", fontsize=20)
        plot_components("UV Decomposition", ax00, uv_curve, juv, Puv, Cuv)
        plot_components("XR Decomposition", ax01, xr_curve, jxr, Pxr, Cxr)
        ax10.set_title("UV 3D View")
        simple_plot_3d(ax10, mapped_UvD, x=wv, y=uv_x)
        ax11.set_title("XR 3D View")
        simple_plot_3d(ax11, optimizer.xrD, x=optimizer.qvector, y=xr_x)
        fig.tight_layout()
        plt.show()

    with plt.Dp():
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=(6,5))
        ax.set_title("UV 3D View with a badly-mapped Peak")
        ax.set_box_aspect(aspect=(1, 4, 1))
        simple_plot_3d(ax, mapped_UvD, x=wv, y=uv_x)
        simple_plot_3d(ax, mapped_UvD * 0.1, x=wv, color="red", alpha=0.1)

        ax.annotate3D('Bug',(300, 100, 0.0),
                      xytext=(40,30),
                      textcoords='offset points',
                      bbox=dict(boxstyle="round", fc="lightyellow"),
                      arrowprops = dict(arrowstyle="-|>",ec='black', fc='white', lw=3))

        fig.tight_layout()
        plt.show()
