"""
    TrimmingResult.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy._MOLASS.Version import get_version_string
from molass_legacy.SerialAnalyzer.DevSettings import get_dev_setting
from .TrimmingDebugUtils import trimming_result_plot_impl

class TrimmingResultDialog(Dialog):
    def __init__(self, parent, pre_recog):
        self.pre_recog = pre_recog
        self.popup_menu = None
        Dialog.__init__(self, parent, "Trimming at a glance - " + get_version_string(), visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack(padx=20, pady=10)

        fig = plt.figure(figsize=(20,9))
        gs = GridSpec(2,10)

        for i, name in enumerate(["UV", "Xray"]):
            ax = fig.add_subplot(gs[i,0])
            ax.set_axis_off()
            ax.text(0.8, 0.5, name, va="center", ha="center", fontsize=20)

        axes = []
        for i in range(2):
            axis_row = []
            for j in range(3):
                start = 1+3*j
                ax = fig.add_subplot(gs[i,start:start+3])
                axis_row.append(ax)
            axes.append(axis_row)
        axes = np.array(axes)
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        self.fig = fig
        self.axes = axes

        result_info = self.pre_recog.ar.result_info
        self.twinxes = trimming_result_plot_impl(fig, axes, *result_info)

        fig.tight_layout()
        self.mpl_canvas.draw()
        if get_dev_setting("enable_dnd_debug"):
            self.mpl_canvas.mpl_connect('button_press_event', self.on_figure_click)

    def buttonbox(self):
        frame = Tk.Frame(self)
        frame.pack(fill=Tk.X)
        tframe = Tk.Frame(frame)
        tframe.pack(side=Tk.LEFT, padx=20)
        bframe = Tk.Frame(frame)
        bframe.pack(side=Tk.RIGHT, padx=20)

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        w = Tk.Button(bframe, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=50, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def on_figure_click(self, event):
        if event.button == 1:
            return
        elif event.button == 3:
            if event.inaxes is not None:
                self.show_popup_menu(event)
                return

    def show_popup_menu(self, event):
        from molass_legacy.KekLib.TkUtils import split_geometry
        if self.popup_menu is None:
            self.popup_menu = Tk.Menu(self, tearoff=0 )
            self.popup_menu.add_command(label="Debug DnD", command=self.show_debug_dnd)
        canvas = self.mpl_canvas_widget
        cx = canvas.winfo_rootx()
        cy = canvas.winfo_rooty()
        w, h, x, y = split_geometry(canvas.winfo_geometry())
        self.popup_menu.post(cx + int(event.x), cy + h - int(event.y))

    def show_debug_dnd(self):
        from .TrimmingDebugDnd import TrimmingDebugDnd
        print("show_debug_dnd")
        dnd = TrimmingDebugDnd(self)

    def get_trimming_info(self):
        return self.pre_recog.ar.result_info[4]

    def redraw(self, alt_info):
        print("redraw: alt_info=", alt_info)

        for axes_row in self.axes:
            for ax in axes_row:
                ax.cla()
        for ax in self.twinxes:
            ax.remove()

        result_info = self.pre_recog.ar.result_info
        self.twinxes = trimming_result_plot_impl(self.fig, self.axes, *result_info, alt_info=alt_info)
        self.mpl_canvas.draw()
