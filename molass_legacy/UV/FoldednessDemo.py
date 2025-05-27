"""
    UV.FoldednessDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt
from  MatrixData import simple_plot_3d
from .Foldedness import Foldedness

def reconstruct_optimizer(dsets, optimizer):
    n_components = optimizer.n_components
    qvector = optimizer.qvector
    wvector = optimizer.wvector
    uv_base_curve = optimizer.uv_base_curve
    xr_base_curve = optimizer.xr_base_curve

    new_optimizer = optimizer.__class__(dsets, n_components,
                        qvector=qvector,
                        wvector=wvector,
                        uv_base_curve=uv_base_curve,
                        xr_base_curve=xr_base_curve,
                        debug=True,
                        )

    init_params = optimizer.init_params
    new_optimizer.prepare_for_optimization(init_params)

    return new_optimizer

def demo(caller, show_result_only=True):
    print("demo")
    dsets = caller.optinit_info.dsets
    optimizer = reconstruct_optimizer(dsets, caller.fullopt)
    params = caller.canvas.get_current_params()
    split_params = optimizer.split_params_simple(params)
    lrf_info = optimizer.objective_func(params, return_lrf_info=True, lrf_debug=False)
    Pxr, Cxr, Puv, Cuv, mapped_UvD = lrf_info.matrices

    if not show_result_only:

        with plt.Dp():
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(20,5), subplot_kw=dict(projection="3d"))
            fig.suptitle("3D View of mapped_UvD")
            ax0.set_title("uvD")
            simple_plot_3d(ax0, optimizer.uvD)
            ax1.set_title("uvD_")
            simple_plot_3d(ax1, optimizer.uvD_)
            ax2.set_title("xrD")
            simple_plot_3d(ax2, optimizer.xrD)
            ax3.set_title("mapped_UvD")
            simple_plot_3d(ax3, mapped_UvD)

            fig.tight_layout()
            plt.show()

        def plot_conc_matrix(ax, C):
            cy_list = []
            for k, cy in enumerate(C):
                cy_list.append(cy)
                ax.plot(cy, ":")
            ty = np.sum(cy_list, axis=0)
            ax.plot(ty, ":", color="red")

        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("conc_matrix")
            plot_conc_matrix(ax1, Cuv)
            plot_conc_matrix(ax2, Cxr)
            fig.tight_layout()
            plt.show()

    uvD = optimizer.uvD
    wv = optimizer.wvector

    foldedness = Foldedness(wv)

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

    with plt.Dp():
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
        fig.suptitle("280, 275, 258 debug plot")
        plot_components("UV Decomposition", axes[0,0], optimizer.uv_curve, juv, Puv, Cuv)
        plot_components("XR Decomposition", axes[0,1], optimizer.xr_curve, jxr, Pxr, Cxr)
        plot_foldedness("UV Foldedness", axes[1,0], wv, Puv)
        axes[1,1].set_axis_off()
        fig.tight_layout()
        plt.show()
