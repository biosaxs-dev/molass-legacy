"""
Optimizer.JobStatusPlot.py
Job status plot for optimization GUI.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .FvScoreConverter import convert_score

def draw_suptitle(self):
    from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
    from molass_legacy.KekLib.BasicUtils import ordinal_str
    from molass_legacy.Optimizer.OptimizerUtils import get_model_name, get_method_name
    job_name = "000"
    in_folder = get_in_folder()
    model_name = get_model_name(self.func_code)
    text = "Job %s State at %s local minimum on %s with model=%s method=%s" % (
        job_name, ordinal_str(self.curr_index), in_folder, model_name, get_method_name())
    if self.suptitle is None or True:
        self.suptitle = self.fig.suptitle(text, fontsize=20)
    else:
        self.suptitle.set_text(text)

def plot_job_state(self, params=None, plot_info=None):
    import seaborn
    seaborn.set_theme()
    from importlib import reload
    import molass_legacy.Optimizer.ProgressChart
    reload(molass_legacy.Optimizer.ProgressChart)
    from molass_legacy.Optimizer.ProgressChart import draw_progress

    if params is None:
        assert plot_info is not None, "Either params or plot_info must be provided."
        x_array = plot_info[-1]
        if len(x_array) == 0:
            return
        fv = plot_info[0]
        k = np.argmin(fv[:,1])
        self.curr_index = k
        params = x_array[k]
        fv_ = fv[k,1]
    else:
        self.curr_index = 0
        fv_ = self.optimizer.objective_func(params, plot=False)

    self.fig = fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(33, 15)
    axes = []
    for j in range(3):
        j_ = j*5
        ax = fig.add_subplot(gs[0:16,j_:j_+5])
        axes.append(ax)

    axt = axes[1].twinx()
    axt.grid(False)
    axes.append(axt)
    self.axes = axes
    self.prog_ax = fig.add_subplot(gs[17:21,2:])
    peak_ax = fig.add_subplot(gs[21:25,2:])
    rg_ax = fig.add_subplot(gs[25:29,2:])
    map_ax = fig.add_subplot(gs[29:33,2:])
    self.prog_axes = [self.prog_ax, peak_ax, rg_ax, map_ax]
    for ax in self.prog_axes[0:3]:
        ax.set_xticklabels([])
    self.prog_title_axes = [fig.add_subplot(gs[17+i*4:21+i*4,0:2]) for i in range(0,4)]
    prog_titles = ["Function SV", "Peak Top Positions", "Rg Values", "Mapped Range"]
    for ax, title in zip(self.prog_title_axes, prog_titles):
        ax.set_axis_off()
        ax.text(-0.3, 0.5, title, fontsize=16)

    draw_suptitle(self)
    sv = convert_score(fv_)
    for ax in self.axes:
        ax.cla()
    ax1, ax2, ax3, axt = self.axes
    axt.grid(False)
    ax1.set_title("UV Decomposition", fontsize=16)
    ax2.set_title("Xray Decomposition", fontsize=16)
    ax3.set_title("Objective Function Scores in SV=%.3g" % sv, fontsize=16)

    axis_info = (fig, (*axes,))
    self.optimizer.objective_func(params, plot=True, axis_info=axis_info)

    if plot_info is not None:
        draw_progress(self, plot_info)