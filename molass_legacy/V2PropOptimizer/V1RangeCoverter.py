"""
    V2PropOptimizer.V1RangeCoverter.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.patches import Rectangle
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.BasicUtils import Struct       # as ControlInfoProxy
from molass_legacy.Decomposer.DecompUtils import make_range_info_impl
from molass_legacy.Decomposer.FitRecord import FitRecord
from molass_legacy.Decomposer.ModelEvaluator import ModelEvaluator
from molass_legacy.Decomposer.UnifiedDecompResult import UnifiedDecompResult
from RangeInfo import shift_editor_ranges
from molass_legacy.DataStructure.AnalysisRangeInfo import AnalysisRangeInfo, convert_to_paired_ranges

def convert_to_v1_model_ranges(prop_optimizer, peaks, paired_ranges, debug=False):
    x = prop_optimizer.x
    y = prop_optimizer.y
    model = prop_optimizer.model

    opt_recs = []
    for kno, params in enumerate(peaks):
        # see 
        sign = 1                # almost always, any exceptions?
        chisqr_n = np.nan       # may not be used
        accepts_real_x = True   # as a temporary fix in ExtrapolSolverDialog
        evaluator = ModelEvaluator(model, params, sign=sign, accepts_real_x=accepts_real_x)
        y_ = evaluator(x)
        m = np.argmax(y_)
        peak = Struct(top_x=x[m] - x[0])   # 
        fit_rec = FitRecord(kno, evaluator, chisqr_n, peak)
        opt_recs.append(fit_rec)

    unif_result = UnifiedDecompResult(x=x, y=y, opt_recs=opt_recs)
    control_info = unif_result.get_range_edit_info()
    editor_ranges = control_info.editor_ranges

    ret_ranges = make_range_info_impl(opt_recs, control_info)
    new_paired_ranges, num_ranges = convert_to_paired_ranges(ret_ranges)

    if debug:
        print("peaks.shape=", peaks.shape)
        print("peaks.paired_ranges=", paired_ranges)
        print("editor_ranges=", editor_ranges)
        print("ret_ranges=", ret_ranges)
        print("new_paired_ranges=", new_paired_ranges)

        shifted_ranges = shift_editor_ranges(x[0], editor_ranges)
        cy_list = prop_optimizer.compute_cy_list(peaks)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("convert_to_models_ranges")
            ax.plot(x, y)
            for cy in cy_list:
                ax.plot(x, cy, ":")
            ty = np.sum(cy_list, axis=0)
            ax.plot(x, ty, ":", color="red")
            ymin, ymax = ax.get_ylim()
            for ranges in shifted_ranges:
                for f, t in ranges:
                    p = Rectangle(
                            (f, ymin),      # (x,y)
                            t - f,          # width
                            ymax - ymin,    # height
                            facecolor   = 'cyan',
                            alpha       = 0.2,
                        )
                    ax.add_patch(p)
            fig.tight_layout()
            plt.show()

    return new_paired_ranges, ret_ranges
