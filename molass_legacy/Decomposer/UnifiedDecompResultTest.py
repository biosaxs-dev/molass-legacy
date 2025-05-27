"""
    UnifiedDecompResultTest.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import copy
import logging
from matplotlib.patches import Rectangle
import molass_legacy.KekLib.DebugPlot as plt

def unit_test(caller):
    from importlib import reload
    import UnifiedDecompResult
    reload(UnifiedDecompResult)
    from UnifiedDecompResult import UnifiedDecompResult

    editor = caller.dialog.get_current_frame()  # get editor because caller.editor is not updated by "Change Model"

    old_result = editor.decomp_result
    print("unit_test")
    # result = copy.deepcopy(old_result)
    result = UnifiedDecompResult(
                xray_to_uv=old_result.xray_to_uv,
                x_curve=old_result.x_curve,
                x=old_result.x,
                y=old_result.y,
                opt_recs=old_result.opt_recs,
                max_y_xray=old_result.max_y_xray,
                model_name=old_result.model_name,
                decomposer=old_result.decomposer,
                uv_y=old_result.uv_y,
                opt_recs_uv=old_result.opt_recs_uv,
                max_y_uv=old_result.max_y_uv,
                nresid_uv=old_result.nresid_uv,
                global_flag=old_result.global_flag,
                )

    logger = logging.getLogger(__name__)

    print("model_name=", result.model_name)
    result.remove_unwanted_elements()

    control_info = result.get_range_edit_info(logger=logger, debug=False)
    editor_ranges = control_info.editor_ranges
    print("editor_ranges=", editor_ranges)

    flags = result.identify_ignorable_elements()
    print("flags=", flags)

    with plt.Dp():
        x = result.x
        y = result.y
        fig, ax = plt.subplots()
        ax.set_title("UnifiedDecompResultTest")
        ax.plot(x, y)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax )

        for list_ in editor_ranges:
            for f, t in list_:
                p = Rectangle(
                        (f, ymin),  # (x,y)
                        t - f,   # width
                        ymax - ymin,    # height
                        facecolor   = 'cyan',
                        alpha       = 0.2,
                    )
                ax.add_patch(p)

        fig.tight_layout()
        plt.show()
