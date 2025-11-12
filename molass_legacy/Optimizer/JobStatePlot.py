"""
Optimizer.JobStatusPlot.py
Job status plot for optimization GUI.
"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_job_state(self, params=None, job_state=None):
    from importlib import reload
    import molass_legacy.Optimizer.ProgressChart
    reload(molass_legacy.Optimizer.ProgressChart)
    from molass_legacy.Optimizer.ProgressChart import draw_progress

    if params is None:
        assert job_state is not None, "Either params or state_info must be provided."
        x_array = job_state[-1]
        if len(x_array) == 0:
            return
        params = x_array[-1]

    self.fig = fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(33, 15)
    axes = []
    for j in range(3):
        j_ = j*5
        ax = fig.add_subplot(gs[0:16,j_:j_+5])
        axes.append(ax)

    axt = axes[1].twinx()
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

    axis_info = (fig, (*axes,))
    self.optimizer.objective_func(params, plot=True, axis_info=axis_info)

    if job_state is not None:
        draw_progress(self, job_state)