"""
    Models.Stochastic.OligoporePlotImpl.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.BasicUtils import Struct
from molass_legacy.Batch.LiteBatch import LiteBatch
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from molass_legacy.Models.Stochastic.OligoporeUtils import plot_oligopore_conformance_impl
from molass_legacy.Models.Stochastic.DatablockUtils import get_block_lrf_src

def plot_results_impl(N, T, pszv, pszp, t0v, tscales, in_folders, block_list, params_list, rg_curves=None):
    num_blocks = len(block_list)

    batch = LiteBatch()

    with plt.Dp(scrollable=True):
        fig, axes = plt.subplots(nrows=num_blocks, ncols=2, figsize=(20, 4*num_blocks))
        i = 0
        if len(tscales) == 0:
            tscales = np.ones(num_blocks)
        if rg_curves is None:
            rg_curves = [None] * num_blocks
        for axes_pair, in_folder, block, t0, tscale, params, rg_curve in zip(axes, in_folders, block_list, t0v, tscales, params_list, rg_curves):
            lrf_src = get_block_lrf_src(batch, in_folder)
            T_ = T*tscale
            t0_ = t0*tscale
            plot_block_state(axes_pair, get_in_folder(in_folder), block, N, T_, t0_, pszv, pszp, lrf_src, params, rg_curve=rg_curve)
            i += 1
            if i > 0:
                # break
                pass

        fig.tight_layout()
        dp = plt.get_dp()
        dp.after(500, lambda: dp.geometry("2000x800"))      # task: do this with auto_resize=True in DebugPlot 
        plt.show()

def rounded_list(fmt, values):
    return ', '.join([fmt % v for v in values])

def plot_block_state(axes_pair, in_folder, block, N, T, t0, pszv, pszp, lrf_src, params, rg_curve=None):
    from importlib import reload
    import RgProcess.RgCurveUtils
    reload(RgProcess.RgCurveUtils)
    from RgProcess.RgCurveUtils import plot_rg_curve
    print("plot_block_state", in_folder)
    ax1, ax2 = axes_pair
    rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = lrf_src.compute_rgs()
    peaks = lrf_src.get_peaks()
    egh_moments_list = lrf_src.get_egh_moments_list()
    model = lrf_src.model
    plot_info = Struct(x=lrf_src.xr_x, y=lrf_src.xr_y, model=model, peaks=peaks)

    ax1t = ax1.twinx()
    ax2t = ax2.twinx()

    if rg_curve is not None:
        for ax in [ax1t, ax2t]:
            plot_rg_curve(ax, rg_curve, with_qualities=True, quality_scale=100)

    def split_params_for_plot(p, num_pszv=2):
        """
        note that these params correspond to those returned by estimate_oligopore_distribution_separately
        """
        N, K, t0 = p[0:3]
        pszv = p[3:3+num_pszv]
        pszp = p[3+num_pszv:3+2*num_pszv]   # last 
        return N, K, t0, pszv, pszp

    plot_params = split_params_for_plot(params)
    plot_oligopore_conformance_impl(ax1, ax1t, *plot_params, plot_info,
                                    rg_info=(peak_rgs, peak_trs, True))

    K = N * T
    plot_oligopore_conformance_impl(ax2, ax2t, N, K, t0, pszv, pszp, plot_info,
                                    rg_info=(peak_rgs, peak_trs, True))
