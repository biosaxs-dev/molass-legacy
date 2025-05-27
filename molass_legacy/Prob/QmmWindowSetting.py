# coding: utf-8
"""
    QmmWindowSetting.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
from matplotlib.patches import Rectangle
from bisect import bisect_right
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting

class QmmWindowSetting(Dialog):
    def __init__(self, parent, outline_fig_frame=None):
        self.outline_fig_frame = outline_fig_frame
        if outline_fig_frame is None:
            self.cs = None
            self.restrict = None
        else:
            self.cs = outline_fig_frame.pre_recog.cs
            self.restrict = outline_fig_frame.xray_elution_restrict
        self.window_patch = None
        Dialog.__init__(self, parent, title="QMM Window Setting", visible=False)

    def show(self):
        # self.after(0, self._adjust_geometry)
        self._show()

    def body(self, body_frame):

        cframe = Tk.Frame(body_frame)
        cframe.pack()
        bframe = Tk.Frame(body_frame)
        bframe.pack(fill=Tk.BOTH, expand=Tk.YES)
        tframe = Tk.Frame(bframe)
        tframe.pack(side=Tk.LEFT)
        pframe = Tk.Frame(bframe)
        pframe.pack(side=Tk.RIGHT)

        fig = plt.figure(figsize=(10,6))
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        self.ax = fig.add_subplot(111)
        self.build_figure(self.ax)

        fig.tight_layout()
        self.mpl_canvas.draw()

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()
        self.build_manipulation_parts(self.ax, pframe)

    def build_figure(self, ax):
        if self.outline_fig_frame is None:
            return

        self.outline_fig_frame.draw_outline_elution(ax)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        tx = xmin*0.5 + xmax*0.5
        ty = ymin*0.2 + ymax*0.8
        ax.text(tx, ty, "Drag to set the Window Range.", alpha=0.3, fontsize=30,
                ha="center", va="center")

    def build_manipulation_parts(self, ax, pframe):
        self.y = self.outline_fig_frame.exact_elution
        length = len(self.y)
        self.x = np.arange(length)

        if self.restrict is None:
            init_min = 0
            init_max = length-1
        else:
            init_min = self.restrict.start
            init_max = self.restrict.stop-1

        self.init_min = init_min
        self.init_max = init_max

        qmm_window_slice = get_setting('qmm_window_slice')

        if qmm_window_slice is None:
            min_ = init_min
            max_ = init_max
        else:
            min_ = qmm_window_slice.start
            max_ = qmm_window_slice.stop - 1
            self.draw_window(min_, max_)

        self.from_  = Tk.IntVar()
        self.from_.set(min_)
        self.to_  = Tk.IntVar()
        self.to_.set(max_)
        self.ivars  = [ self.from_, self.to_ ]

        k = -1
        for i, t in enumerate(["from ", "to "]):
            if i == 1:
                k += 1
                space = Tk.Label(pframe, width=1)
                space.grid(row=0, column=k)

            k += 1
            label = Tk.Label(pframe, text=t)
            label.grid(row=0, column=k)

            k += 1
            entry   = Tk.Spinbox( pframe, textvariable=self.ivars[i],
                        from_=init_min, to=init_max, increment=1, 
                        justify=Tk.CENTER, width=6 )
            entry.grid(row=0, column=k)

        self.spinbox_stop_trace = False
        for k, dvar in enumerate(self.ivars):
            dvar.trace('w', lambda *args, k_=k:self.spinbox_tracer(k_))

        self.span = SpanSelector(ax, self.onselect, 'horizontal', useblit=True,
                            props=dict(alpha=0.5))

        clear_button = Tk.Button(pframe, text="Clear", command=self.clear_window)
        clear_button.grid(row=0, column=5, padx=10)

    def draw_window(self, xmin, xmax):
        ax = self.ax
        ymin, ymax = ax.get_ylim()
        if self.window_patch is None:
            rect = Rectangle(
                    (xmin, ymin),   # (x,y)
                    xmax - xmin,    # width
                    ymax - ymin,    # height
                    facecolor   = 'cyan',
                    alpha       = 0.2,
                )
            self.window_patch = ax.add_patch(rect)
        else:
            rect = self.window_patch
            rect.set_xy((xmin, ymin))
            rect.set_width(xmax - xmin)
        self.mpl_canvas.draw()

    def clear_window(self):
        if self.window_patch is None:
            return

        self.window_patch.remove()
        self.window_patch = None
        self.spinbox_stop_trace = True
        self.from_.set(0)
        self.to_.set(len(self.x)-1)
        self.update()
        self.spinbox_stop_trace = True
        self.mpl_canvas.draw()

    def onselect(self, xmin, xmax):
        if xmax - xmin < 1e-6:
            return

        # print('onselect', (xmin, xmax))
        xmin = max(self.init_min, xmin)
        xmax = min(self.init_max, xmax)
        self.spinbox_stop_trace = True
        try:
            max_i = len(self.x)-1
            start = min(max_i, bisect_right(self.x, xmin))
            j = min(max_i, bisect_right(self.x, xmax))
            stop = j+1
            self.from_.set(self.x[start])
            self.to_.set(self.x[j])
            self.update()
            self.draw_window(xmin, xmax)
        except:
            if True:
                from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
                etb = ExceptionTracebacker()
                print(etb.last_lines())
        self.spinbox_stop_trace = False

    def spinbox_tracer(self, k):
        if self.spinbox_stop_trace:
            return

        values = []
        try:
            for ivar in self.ivars:
                values.append(ivar.get())
        except:
            return

        if values[0] > values[1]:
            print('values=', values)
            if k == 0:
                v = values[1]
            else:
                v = values[0]
            self.spinbox_stop_trace = True
            self.ivars[k].set(v)
            values[k] = v
            self.update()
            self.spinbox_stop_trace = False

        start = values[0]
        stop = values[1]
        self.draw_window(start, stop)

    def apply(self):
        if self.window_patch is None:
            qmm_window_slice = None
            set_setting('qmm_window_slice_uv', None)
        else:
            start = self.from_.get()
            stop = self.to_.get() + 1
            qmm_window_slice = slice(start, stop)
            self.set_uv_window(start, stop)
        set_setting('qmm_window_slice', qmm_window_slice)

    def set_uv_window(self, start, stop):
        if self.cs is None:
            qmm_window_slice_uv = None
        else:
            cs = self.cs
            i_list = []
            for j in [start, stop]:
                i_list.append(int(round(cs.slope*j + cs.intercept)))
            qmm_window_slice_uv = slice(*i_list)
        set_setting('qmm_window_slice_uv', qmm_window_slice_uv)
