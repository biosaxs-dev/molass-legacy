"""
    ModelParams.SdmPlotUtils.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Optimizer.TheDebugUtils import convert_score_list, plot_scores
from molass_legacy.GuinierTools.RgCurveUtils import plot_rg_curves
from molass_legacy.Optimizer.FvScoreConverter import convert_score

def plot_objective_state(score_list_pair, fv, xm,
        lrf_info,
        overlap, rg_curve, rg_params,
        score_names,
        fig_info, axis_info,
        func, params,
        avoid_pinv=False,
        debug=False,
        **kwargs
        ):
    ratio_interpret = kwargs.get("ratio_interpret", False)

    if func is None or params is None:
        pass
    else:
        xr_params, xr_baseparams, rg_params_not_used, (a, b), uv_params, uv_baseparams, (c, d), sec_params = func.split_params_simple(params)
        t0, rp, N, me, T, mp = sec_params
        rho = rg_params/rp
        rho[rho > 1] = 1
        model_trs = t0 + N*T*(1 - rho)**(me + mp)
        if debug:
            print("--------------------------------------- sec_params=", sec_params)
            print("--------------------------------------- rg_params=", rg_params)
            print("--------------------------------------- model_trs=", model_trs)

    uv_x = lrf_info.uv_x
    uv_y = lrf_info.uv_y
    uv_cy_list = lrf_info.get_uv_cy_list()
    uv_ty = lrf_info.uv_ty
    x = lrf_info.x
    y = lrf_info.y
    xr_cy_list = lrf_info.get_xr_cy_list()
    xr_ty = lrf_info.xr_ty

    if ratio_interpret:
        from molass_legacy.Optimizer.OptLrfInfo import get_ratio_cy_list
        uv_cy_list = get_ratio_cy_list(uv_y, uv_cy_list)
        uv_ty = uv_y
        xr_cy_list = get_ratio_cy_list(y, xr_cy_list)
        xr_ty = y

    num_components = lrf_info.get_num_substantial_components()
    score_list = convert_score_list(score_list_pair)

    if fig_info is None:
        in_folder, seeds, result = [None]*3
    else:
        in_folder, seeds, result = fig_info

    if axis_info is None:
        plt.push()
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(21,7))
        axt = ax2.twinx()
        axt.grid(False)
    else:
        if len(axis_info) == 2:
            fig, (ax1, ax2, ax3, axt) = axis_info
        else:
            assert False

    if result is not None:
        func_name =  "" if func is None else " " + func.get_name()
        fig.suptitle("Objective Function%s Debug Plot for %s, SV=%.3g" % (func_name, in_folder, convert_score(result.fun)), fontsize=20)

    if ax1 is not None:
        ax1.plot(uv_x, uv_y, color='blue')
        points = lrf_info.estimate_uv_peak_points(elutions=uv_cy_list[:-1])     # [:-1] excludiong baseline
        for k, uv_cy in enumerate(uv_cy_list):
            if k < num_components:
                style = ':'
                label = 'component-%d' % (k+1)
                color = None
            else:
                style = '-'
                label = 'baseline'
                color = 'red'
            ax1.plot(uv_x, uv_cy, style, color=color, label=label)
            if k < len(points):
                ax1.plot(*points[k], "o", color="yellow")
        ax1.plot(uv_x, uv_ty, ':', color='red', lw=2, label='total')
        ax1.legend()

    if ax2 is not None:
        ax2.plot(x, y, color='orange')
        num_drawn_flags = 0
        points = lrf_info.estimate_xr_peak_points(elutions=xr_cy_list[:-1])     # [:-1] excludiong baseline
        for k, xr_cy in enumerate(xr_cy_list):
            if k < num_components:
                style = ':'
                label = 'component-%d' % (k+1)
                color = None
            elif k == num_components:
                style = '-'
                label = 'baseline'
                color = 'red'
            else:
                style = ':'
                label = 'effect of ii' if num_drawn_flags == 0 else None
                color = 'pink'
                num_drawn_flags += 1
            ax2.plot(x, xr_cy, style, color=color, label=label)
            if k < len(points):
                ax2.plot(*points[k], "o", color="yellow")
        ax2.plot(x, xr_ty, ':', color='red', lw=2, label='total')

        if False:
            # consider overlaps later
            y1 = np.zeros(len(x))
            ax2.fill_between(x, y1, overlap, fc='pink', alpha=0.2)

        ax2.legend(loc='upper right')

    if axt is not None:
        assert ax2 is not None
        rg_params_ = lrf_info.get_valid_rgs(rg_params)
        plot_rg_curves(axt, points[:,1], rg_params_, x, xr_cy_list, xr_ty, rg_curve)

    if ax3 is not None:
        plot_scores(ax3, score_list, score_names)
        if avoid_pinv:
            tx = np.average(ax3.get_xlim())
            ty = np.average(ax3.get_ylim())
            ax3.text(tx, ty, "pinv error!", ha="center", va="center", color="red", fontsize=20)

    if axis_info is None:
        fig.tight_layout()
        if result is not None:
            fig.subplots_adjust(top=0.9)
        debug_fv = plt.show()
        plt.pop()
    else:
        debug_fv = None

    return debug_fv
