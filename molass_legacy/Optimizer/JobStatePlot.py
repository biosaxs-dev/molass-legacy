"""
Optimizer.JobStatePlot.py
Job status plot for optimization GUI.
"""
import numpy as np
import matplotlib.pyplot as plt


def _draw_monitor_anomaly_bands(monitor):
    """Draw anomaly bands on UV (axes[0]) and XR (axes[1]) panels if available."""
    jv = getattr(monitor, 'anomaly_jv', None)
    mask = getattr(monitor, 'anomaly_mask', None)
    if jv is None or mask is None or not np.any(mask):
        return

    color, alpha = 'red', 0.08
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    for group in np.split(idx, breaks):
        lo, hi = jv[group[0]], jv[group[-1]]
        # axes[1] = XR decomposition panel
        monitor.axes[1].axvspan(lo, hi, color=color, alpha=alpha, zorder=0)
        # axes[0] = UV decomposition panel (same frame range — approximate)
        monitor.axes[0].axvspan(lo, hi, color=color, alpha=alpha, zorder=0)

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

def plot_job_state(self, params, plot_info=None, niter=20, display_optimizer=None, best_sv=None):
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
    # Use display_optimizer (subprocess-equivalent) for the objective re-eval
    # so on-screen SV matches callback.txt SV (issue #118). Other panels
    # (peak positions, Rg history, mapping) still read parent state.
    # Issue #50: display_optimizer is None during live in-process runs to prevent
    # concurrent access to the BH optimizer. Show a placeholder instead.
    if display_optimizer is not None:
        plot_objective_func(display_optimizer,
                            params, axis_info=(self.fig, self.axes), best_sv=best_sv)
    else:
        # Live in-process run: skip objective_func re-evaluation to avoid racing
        # with _run_solve's BH sub-minimizer.  Fill panels with placeholder text.
        _panel_titles = ["UV Decomposition", "XR Decomposition", "Decomposition SV"]
        for ax, title in zip(self.axes[:3], _panel_titles):
            ax.set_title(title, fontsize=16)
            ax.text(0.5, 0.5, "(updating at trial completion)",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=11, color='gray', style='italic')
        if best_sv is not None:
            self.axes[2].set_title("best SV=%.3g" % best_sv, fontsize=16)

    # Anomaly exclusion bands — consistent with plot_compact() and plot_components()
    _draw_monitor_anomaly_bands(self)

    if plot_info is not None:
        draw_progress(self, plot_info, niter=niter)

def plot_objective_func(optimizer, params, axis_info=None, best_sv=None):
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
    # Issue #128: show best accepted SV only (current SV removed to avoid confusion).
    if best_sv is not None:
        ax3.set_title("best SV=%.3g" % best_sv, fontsize=16)
    else:
        ax3.set_title("Objective Function Scores in SV=%.3g" % sv, fontsize=16)
    optimizer.objective_func(params, plot=True, axis_info=axis_info)