"""
    TheDebugUtils.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.patches import Rectangle
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Elution.CurveUtils import simple_plot
from molass_legacy.Optimizer.BasicOptimizer import SCORE_DEV_ALLOW, SCORE_DEV_LIMIT
from molass_legacy._MOLASS.SerialSettings import get_setting

def plot_mapped_curve(ax, map_params, x, uv_curve):
    A, B = map_params
    uv_y = uv_curve.spline(A*x + B)
    ax.plot(x, uv_y, color='blue', label='data')
    topx_list = []
    for k, info in enumerate(uv_curve.get_major_peak_info()):
        topx = info[1]
        topx_list.append(topx)
        topy = uv_curve.spline(topx)
        x_ = (topx - B)/A
        label = 'peak tops' if k == 0 else None
        ax.plot(x_, topy, 'o', color='red', label=label)

    for k, btmx in enumerate(uv_curve.get_major_valley_bottoms(topx_list)):
        btmy = uv_curve.spline(btmx)
        x_ = (btmx - B)/A
        label = 'valley bottoms' if k == 0 else None
        ax.plot(x_, btmy, 'o', color='green', label=label)

def plot_decomp_elements(ax, x, opt_recs):
    for k, rec in enumerate(opt_recs):
        fnc = rec[1]
        cy = fnc(x)
        ax.plot(x, cy, ':', label='component-%d' % k)

def plot_decomp_result_titles(fig, ax1, ax2):
    from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
    in_folder = get_in_folder()
    fig.suptitle("Initial Parameters Estimation for %s" % in_folder, fontsize=20)
    ax1.set_title("Mapped UV Decomposition", fontsize=16)
    ax2.set_title("Xray Decomposition", fontsize=16)

def plot_decomp_result(uv_curve, xr_curve, map_params, decomp_result=None, fig_info=None):

    if fig_info is None:
        plt.push()
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,7))
    else:
        fig, (ax1, ax2) = fig_info

    plot_decomp_result_titles(fig, ax1, ax2)

    x = xr_curve.x
    plot_mapped_curve(ax1, map_params, x, uv_curve)
    simple_plot(ax2, xr_curve, color='orange', legend=False)

    if decomp_result is not None:
        plot_decomp_elements(ax1, x, decomp_result.opt_recs_uv)
        plot_decomp_elements(ax2, x, decomp_result.opt_recs)

    for ax in (ax1, ax2):
        ax.legend()

    if fig_info is None:
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.show()
        plt.pop()

def convert_score_list(score_list_pair):
    score_list, penalties = score_list_pair
    return np.concatenate([score_list, penalties])

def plot_scores(ax, score_list, score_names, alpha=1, label=None, invert=True, add_patch=True, debug=True):
    if debug:
        from importlib import reload
        import molass_legacy.Optimizer.FvScoreConverter
        reload(molass_legacy.Optimizer.FvScoreConverter)
    from .FvScoreConverter import convert_score

    converted_list = [convert_score(v) for v in score_list]

    NUM_MAJOR_SCORES = get_setting("NUM_MAJOR_SCORES")
    major_scores = score_list[0:NUM_MAJOR_SCORES]
    score_mean = np.mean(major_scores)
    score_dev = max(SCORE_DEV_LIMIT, SCORE_DEV_ALLOW * np.std(major_scores))
    f = convert_score(score_mean - score_dev)
    t = convert_score(score_mean + score_dev)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    y_pos = np.arange(len(score_list))
    ax.barh(y_pos, converted_list, label=label, alpha=alpha)
    ax.set_yticks(y_pos, score_names)

    if invert:
        ax.invert_yaxis()

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, 105)

    if add_patch:
        ymin, ymax = ax.get_ylim()
        p = Rectangle(
                (f, ymin),      # (x,y)
                t - f,          # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.1,
            )
        ax.add_patch(p)
