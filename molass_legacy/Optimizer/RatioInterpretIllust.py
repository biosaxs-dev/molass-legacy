"""
    Optimizer.RatioInterpretIllust.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

MODEL_NAME_DICT = {
    "G0346": "EGH",
    "G1100": "SDM",
    "G2010": "EDM",
}

def ratio_interpret_illust(js_canvas):
    print("show_adhoc_figure_impl")
    optimizer = js_canvas.fullopt
    func_code = optimizer.get_name()
    model_name = MODEL_NAME_DICT[func_code]
    params = js_canvas.get_current_params()
    job_name = js_canvas.dialog.get_job_info()[0]
    in_folder = get_in_folder()

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        ax1.set_title("Strict Interpretation", fontsize=16)
        ax2.set_title("Rational Interpretation", fontsize=16)
        fig.suptitle("Interpretation Contrast at %s with=%s of %s" % (job_name, model_name, in_folder), fontsize=20)
        optimizer.objective_func(params, plot=True, axis_info=(fig, (None, ax1, None, None)))
        optimizer.objective_func(params, plot=True, axis_info=(fig, (None, ax2, None, None)), ratio_interpret=True)
        fig.tight_layout()
        plt.show()