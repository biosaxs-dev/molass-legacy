"""
    Optimizer.MplJobState.py

    migration of JobStateCanvas to Jupyter Notebook
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from molass_legacy._MOLASS.SerialSettings import get_setting

class MplJobState:
    def __init__(self, monitor):
        self.monitor = monitor
        self.elution_model = get_setting("elution_model")
        self.update_init_state(monitor.state_info)
        self.iconified = False
        self.logger = monitor.logger
        self.optinit_info = monitor.optinit_info
        self.composite = self.optinit_info.composite
        self.suptitle = None

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
        self.optview_ranges = None
        self.initial_view_ranges = None
