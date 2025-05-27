"""
    Optimizer.AllParameters.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from ScrolledFrame import ScrolledFrame

class AllParameters(Dialog):
    def __init__(self, parent, optimizer, x_array, curr_index, best_index, x_, xmin, xmax):
        self.optimizer = optimizer
        self.x_array = x_array
        self.curr_index = curr_index
        self.best_index = best_index
        self.x_ = x_
        self.xlim = (xmin, xmax)
        Dialog.__init__(self, parent, "Variation of All Parameters", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = ScrolledFrame(body_frame)
        cframe.pack(side=Tk.LEFT)
        cframe.add_bind_mousewheel()

        cframe_ = Tk.Frame(cframe.interior)
        cframe_.pack()
        tframe_ = Tk.Frame(cframe.interior)
        tframe_.pack(side=Tk.LEFT)

        x_array = self.x_array
        nrows = x_array.shape[1]
        fig = plt.figure(figsize=(16, nrows*1))
        self.fig = fig
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe_)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1)
        self.draw_all_parames(fig, x_array, nrows)
        self.after(300, self.set_better_geometry)

    def set_better_geometry(self):
        geometry = self.geometry()
        geometry = re.sub(r"(\d+x)(\d+)(.+)", lambda m: m.group(1) + "400" + m.group(3), geometry)
        self.geometry(geometry)

    def draw_all_parames(self, fig, x_array, nrows):
        x_ = self.x_
        xmin, xmax = self.xlim

        names = self.optimizer.params_type.get_parameter_names()

        c = self.curr_index
        j = self.best_index
        k = len(x_) - 1
        ncols = 8
        gs = GridSpec(nrows, ncols)
        axes = []
        for i in range(nrows):
            ax1 = fig.add_subplot(gs[i,0])
            ax1.set_axis_off()
            ax1.text(0.5, 0.5, names[i], ha="center", va="center")
            ax2 = fig.add_subplot(gs[i,1:])
            ax2.set_xticklabels([])
            ax2.plot(x_, x_array[:,i])
            ax2.set_xlim(xmin, xmax)
            ymin, ymax = ax2.get_ylim()
            ax2.set_ylim(ymin, ymax)
            ax2.plot(x_[[j,j]], [ymin, ymax], color="red")
            if c != j:
                ax2.plot(x_[[c,c]], [ymin, ymax], color="yellow")
            if j != k:
                ax2.plot(x_[[k,k]], [ymin, ymax], color="gray", alpha=0.3)
            axes.append((ax1, ax2))
        self.axes = np.array(axes)

        fig.tight_layout()
        fig.subplots_adjust(top=0.99, bottom=0.01, hspace=0.3, right=0.96)
        self.mpl_canvas.draw()
