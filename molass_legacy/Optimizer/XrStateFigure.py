"""
    Optimizer.XrStateFigure.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.SerialAnalyzer.DataUtils import cut_upper_folders
from .FvScoreConverter import convert_score

DATA_NAME_DICT = {
    "20220210": "C1015s",
    "20220517": "C1015F",
    "20240730": "5913s001",
}

def show_this_figure_impl(self, best=True):
    optimizer = self.fullopt
    modelname = optimizer.get_model_name()
    if best:
        params = self.get_best_params()
    else:
        params = self.get_current_params()
    fv = optimizer.objective_func(params)
    sv = convert_score(fv)
    in_folder = get_setting('in_folder')
    folder = cut_upper_folders(in_folder)
    dataname = DATA_NAME_DICT[folder]
    with plt.Dp(button_spec=["Close"]):
        fig, ax = plt.subplots(figsize=(6,4.5))
        ax.set_title("Decomposition of %s with %s, SV=%.1f" % (dataname, modelname, sv), fontsize=16, y=1.01)
        axt = ax.twinx()
        axt.grid(False)
        axes = (None, ax, None, axt)
        optimizer.objective_func(params, plot=True, axis_info=(fig, axes))
        fig.tight_layout()
        plt.show()