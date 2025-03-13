"""
    OptRecsUtils.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

def debug_plot_opt_recs_impl(ax, x, y, opt_recs, color=None, spline=None):
    for k, rec in enumerate(opt_recs):
        func = rec.evaluator
        peak = rec.peak
        ax.plot(x, y, color="orange")
        ax.plot(x, func(x), ":", color=color)
        if spline is not None:
            px = peak.top_x
            py = spline(px)
            ax.plot(px, py, "o", color="yellow")

def debug_plot_opt_recs(ecurve, opt_recs, eval_x=None, title=None):

    x = ecurve.x
    y = ecurve.y
    max_y = ecurve.max_y
    if eval_x is None:
        eval_x = x

    with plt.Dp():
        if title is None:
            title = "OptRecsDebug"
        fig, ax = plt.subplots()
        ax.set_title(title)
        debug_plot_opt_recs_impl(ax, eval_x, y, opt_recs, spline=ecurve.spline)
        print("ratio=",y/max_y)
        fig.tight_layout()
        plt.show()

def compute_area_proportions(x, opt_recs):
    areas = []
    for k, rec in enumerate(opt_recs):
        func = rec.evaluator
        areas.append(np.sum(func(x)))
    areas = np.array(areas)
    return areas/np.sum(areas)

def eoii_correct_opt_recs(frame, opt_recs_, debug=False):
    if True:
        from importlib import reload
        import LRF.EoiiCorrector
        reload(LRF.EoiiCorrector)
        import DecompUtils
        reload(DecompUtils)

    from LRF.EoiiCorrector import EoiiCorrector
    from molass_legacy.DataStructure.AnalysisRangeInfo import convert_to_paired_ranges
    from DecompUtils import make_range_info_impl

    if debug:
        ecurve = frame.dialog.sd.xray_curve
        debug_plot_opt_recs(ecurve, opt_recs_, title="eoii_correct_opt_recs entry")

    """
        this is just a temporary implementation.
        make ranges for EoiiCorrector which only needs paird_range.top_x used in get_pno_map_impl()
        task: simplify this procudure to get ret[0], i.e., cnv_ranges in PreviewData
    """
    temp_paired_ranges = make_range_info_impl(None, frame.control_info, frame.specpanel_list, no_elm_recs=True)
    ret = convert_to_paired_ranges(temp_paired_ranges)

    ec = EoiiCorrector(frame.dialog.sd, ret[0])

    params_list = []
    for k, rec in enumerate(opt_recs_):
        values = rec.evaluator.get_all_param_values()
        if debug:
            print([k], values)
        params_list.append(values)

    func = opt_recs_[0].evaluator.get_func()  # egha or emga
    corrected, new_params_list = ec.correct_params_list(func, params_list, debug=debug)

    if corrected:
        for rec, new_params in zip(opt_recs_, new_params_list):
            rec.evaluator.set_new_params(new_params)        # this will induce a bug. do away with lmfit.

    if debug:
        ecurve = frame.dialog.sd.xray_curve
        debug_plot_opt_recs(ecurve, opt_recs_, title="eoii_correct_opt_recs result")
