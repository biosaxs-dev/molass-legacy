"""
Optimizer.JobStatePlot.py
Job status plot for optimization GUI.
"""
import numpy as np
import matplotlib.pyplot as plt

def draw_suptitle(self):
    from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
    from molass_legacy.KekLib.BasicUtils import ordinal_str
    from molass_legacy.Optimizer.OptimizerUtils import get_model_name, get_method_name
    job_name = "%03d" % (self.num_trials,)
    in_folder = get_in_folder()
    model_name = get_model_name(self.func_code)
    text = "Job %s State at %s local minimum on %s with model=%s method=%s" % (
        job_name, ordinal_str(self.curr_index), in_folder, model_name, get_method_name())
    if self.suptitle is None or True:
        self.suptitle = self.fig.suptitle(text, fontsize=20)
    else:
        self.suptitle.set_text(text)

def plot_job_state(self, params, plot_info=None, niter=20):
    from matplotlib.gridspec import GridSpec
    import seaborn
    seaborn.set_theme()
    from importlib import reload
    import molass_legacy.Optimizer.ProgressChart
    reload(molass_legacy.Optimizer.ProgressChart)
    from molass_legacy.Optimizer.ProgressChart import draw_progress

    self.fig = fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(33, 15, wspace=1.3, hspace=1.0)
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
    plot_objective_func(self.optimizer, params, axis_info=(self.fig, self.axes))

    if plot_info is not None:
        draw_progress(self, plot_info, niter=niter)

def plot_objective_func(optimizer, params, axis_info=None):
    from .FvScoreConverter import convert_score
    fv_ = optimizer.objective_func(params)
    sv = convert_score(fv_)

    if axis_info is None:
        fig, axes = plt.subplots(ncols=3, figsize=(18,4.5))
        ax1, ax2, ax3 = axes
        axt = ax2.twinx()
        axt.grid(False)
        axis_info = (fig, (*axes, axt))
    else:
        fig, axes = axis_info
        ax1, ax2, ax3 = axes[:3]

    ax1.set_title("UV Decomposition", fontsize=16)
    ax2.set_title("Xray Decomposition", fontsize=16)
    ax3.set_title("Objective Function Scores in SV=%.3g" % sv, fontsize=16)
    optimizer.objective_func(params, plot=True, axis_info=axis_info)