"""
    UnifiedDecompResultTest.py

    Copyright (c) 2020,2025, SAXS Team, KEK-PF
"""
import logging
from matplotlib.patches import Rectangle

def plot_decomp_results(results, editor_ranges, use_debug_plot=False):
    """
    Plot the decomposition results using matplotlib.
    If use_debug_plot is True, use molass_legacy.KekLib.DebugPlot for plotting.
    """
    print("plot_decomp_results: editor_ranges=", editor_ranges)
    if use_debug_plot:
        import molass_legacy.KekLib.DebugPlot as plt
    else:
        import matplotlib.pyplot as plt

    def plot_decomp_results_impl():
        num_results = len(results)
        fig, axes = plt.subplots(ncols=num_results, figsize=(6 * num_results, 5))
        if num_results == 1:
            axes = [axes]
        for ax, result in zip(axes, results):
            x = result.x
            y = result.y
            ax.set_title("Decomposition Results")
            ax.plot(x, y)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)

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

            for rec in result.opt_recs:
                func = rec[1]
                cy = func(x)
                ax.plot(x, cy, ":")

            axt = ax.twinx()
            axt.grid(False)
            for rec in result.opt_recs_uv:
                func = rec[1]
                cy = func(x)
                axt.plot(x, cy, "o", alpha=0.5)

        fig.tight_layout()
        plt.show()

    if use_debug_plot:
        with plt.Dp():
            plot_decomp_results_impl()
    else:
        plot_decomp_results_impl()

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
    print("select_matrix=", control_info.select_matrix)
    print("top_x_list=", control_info.top_x_list)

    flags = result.identify_ignorable_elements()
    print("flags=", flags)

    plot_decomp_results([old_result, result], editor_ranges, use_debug_plot=True)
