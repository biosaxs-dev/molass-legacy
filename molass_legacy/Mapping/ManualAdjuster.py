"""
    ManualAdjuster.py

    Copyright (c) 2019-2023, SAXS Team, KEK-PF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
try:
    import molass_legacy.KekLib.CustomMessageBox as MessageBox
except:
    import OurMessageBox as MessageBox
from molass_legacy.Elution.CurveUtils import simple_plot
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting

def single_float(f):
    return float('%g' % f)

class ManualAdjuster(Dialog):
    def __init__(self, parent, sd, mapper):
        self.a_curve = sd.absorbance.a_curve
        self.x_curve = sd.xray_curve
        A, B = mapper.map_params
        self.init_A = A
        self.init_B = B

        time_shift = get_setting('manual_time_shift')
        time_scale = get_setting('manual_time_scale')
        if time_shift is not None:
            A = 1/time_scale
            B = -time_shift/time_scale

        self.A = A
        self.B = B
        self.applied = False

        Dialog.__init__(self, parent, "Manual Syncronizer", visible=False)

    def body(self, body_frame):
        tk_set_icon_portable(self)

        iframe = Tk.Frame(body_frame)
        iframe.pack(padx=20, pady=10)

        fig_frame = Tk.Frame(iframe)
        fig_frame.pack()
        lower_frame = Tk.Frame(iframe)
        lower_frame.pack(fill=Tk.X)

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))
        self.mpl_canvas = FigureCanvasTkAgg(fig, fig_frame)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.fig = fig
        fig.suptitle("Manual Adjustment for " + get_setting('in_folder'), fontsize=20)
        self.axes = axes

        self.draw_figures()

        tool_frame = Tk.Frame(lower_frame)
        tool_frame.pack(side=Tk.LEFT)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tool_frame)
        self.toolbar.update()

        manip_frame = Tk.Frame(lower_frame)
        manip_frame.pack(side=Tk.RIGHT, padx=100)

        self.build_manip_panel(manip_frame)

    def draw_figures(self):
        for ax in self.axes:
            ax.cla()

        a_curve = self.a_curve
        x_curve = self.x_curve 

        ax1, ax2, ax3 = self.axes
        ax1.set_title("UV data", fontsize=16)
        ax2.set_title("Xray data", fontsize=16)
        ax3.set_title("UV/Xray overlay (Xray intensity scale)", fontsize=16)

        simple_plot(ax1, a_curve, color='blue')
        simple_plot(ax2, x_curve, color='orange')
        simple_plot(ax3, x_curve, color='orange', legend=False)

        x = x_curve.x
        j = self.A*x + self.B
        scale = x_curve.max_y/a_curve.max_y
        y = a_curve.spline(j)*scale
        ax3.plot(x, y, color='blue', label='mapped data')

        ax1.set_xlim(ax1.get_xlim())
        ax1.set_ylim(ax1.get_ylim())
        ax1.plot( ax1.get_xlim(), [0, 0], ':', color='red' )

        xlim_list = []
        ylim_list = []
        for ax in [ax2, ax3]:
            xmin, xmax = ax.get_xlim()
            xlim_list.append((xmin, xmax))
            ymin, ymax = ax.get_ylim()
            ylim_list.append((ymin, ymax))

        xlim_array = np.array(xlim_list)
        ylim_array = np.array(ylim_list)

        xmin = np.min(xlim_array[:,0])
        xmax = np.max(xlim_array[:,1])
        ymin = np.min(ylim_array[:,0])
        ymax = np.min(ylim_array[:,1])

        for ax in [ax2, ax3]:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.plot( [xmin, xmax], [0, 0], ':', color='red' )

        ax3.legend()

        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.85)
        self.mpl_canvas.draw()

    def build_manip_panel(self, manip_frame):
        shift_label = Tk.Label(manip_frame, text="time shift")
        shift_label.grid(row=0, column=0)

        max_shift = len(self.x_curve.x)*2
        self.time_shift = Tk.DoubleVar()
        self.time_shift.set(single_float(-self.B/self.A))
        shift_spinbox = Tk.Spinbox(manip_frame, textvariable=self.time_shift,
                                            from_=-max_shift, to=max_shift, increment=0.1,
                                            justify=Tk.CENTER, width=8)
        shift_spinbox.grid(row=0, column=1)

        space = Tk.Label(manip_frame, width=4)
        space.grid(row=0, column=2)

        scale_label = Tk.Label(manip_frame, text="time scale")
        scale_label.grid(row=0, column=3)

        min_scale = 0.1
        max_scale = 2
        self.time_scale = Tk.DoubleVar()
        self.time_scale.set(single_float(1/self.A))
        scale_spinbox = Tk.Spinbox(manip_frame, textvariable=self.time_scale,
                                            from_=min_scale, to=max_scale, increment=0.001,
                                            justify=Tk.CENTER, width=8)
        scale_spinbox.grid(row=0, column=4)

        self.time_shift.trace('w', self.time_params_tracer)
        self.time_scale.trace('w', self.time_params_tracer)

        space = Tk.Label(manip_frame, width=4)
        space.grid(row=0, column=5)

        reset_btn = Tk.Button(manip_frame, text="Reset", command=self.reset)
        reset_btn.grid(row=0, column=6)

    def show(self):
        self._show()

    def time_params_tracer(self, *args):
        try:
            time_shift = self.time_shift.get()
            time_scale = self.time_scale.get()
            self.A = 1/time_scale               # need to cover zero divide
            self.B = -time_shift/time_scale
        except:
            return

        self.draw_figures()

    def reset_to_the_init_params(self):
        A, B = self.init_A, self.init_B

        A_ = single_float(1/A)      # so as to look like single precision
        B_ = single_float(-B/A)     # so as to look like single precision

        print('B_=', B_, 'A_=', A_)
        self.time_shift.set(B_)
        self.time_scale.set(A_)

    def reset(self):
        self.reset_to_the_init_params()
        self.draw_figures()

    def apply(self):
        self.applied = True

        time_shift = self.time_shift.get()
        time_scale = self.time_scale.get()

        set_setting('manual_time_shift', time_shift)
        set_setting('manual_time_scale', time_scale)
