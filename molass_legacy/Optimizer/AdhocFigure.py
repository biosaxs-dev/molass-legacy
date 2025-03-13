"""
   AdhocFigure.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

MODEL_NAME_DICT = {
    "G0346": "EGH",
    "G1100": "SDM",
    "G2010": "EDM",
}

def show_adhoc_figure_impl(js_canvas):
    print("show_adhoc_figure_impl")
    optimizer = js_canvas.fullopt
    func_code = optimizer.get_name()
    model_name = MODEL_NAME_DICT[func_code]
    params = js_canvas.get_current_params()
    job_name = js_canvas.dialog.get_job_info()[0]
    in_folder = get_in_folder()

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_title("V2 Result at %s with=%s of %s" % (job_name, model_name, in_folder), fontsize=16)
        optimizer.objective_func(params, plot=True, axis_info=(fig, (None, ax, None, None)))
        fig.tight_layout()
        plt.show()