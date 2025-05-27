# coding: utf-8
"""
    DelayAdjuster.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from lmfit import Parameters, minimize
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk, is_empty_val
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable, BlinkingFrame
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
from molass_legacy.KekLib.TkUtils import is_low_resolution
from molass_legacy._MOLASS.SerialSettings import get_setting, get_xray_picking
from DelayOptimizer import DelayOptimizer
from molass_legacy._MOLASS.SerialSettings import set_setting
from SvdDenoise import get_denoised_data

MODEL_ID_LIST = ["One State", "Two-State Unfolding", "Three-State Unfolding"]

class DelayAdjuster(Dialog):
    def __init__(self, parent, xdata, in_folder):
        self.logger = logging.getLogger(__name__)
        self.busy = False
        self.applied = False
        self.xdata = xdata
        self.mapper = None
        self.in_folder = in_folder
        set_setting('conc_factor', 5)   # TODO: change setting according to data date
        Dialog.__init__(self, parent, "Delay Volume Factor Adjuster", visible=False)

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()
        bframe = Tk.Frame(body_frame)
        bframe.pack(fill=Tk.X, expand=1)
        self.build_canvas(cframe)
        self.draw_init()

        pframe = Tk.Frame(bframe)
        pframe.pack(side=Tk.RIGHT)
        self.build_params_frame(pframe)

        tbframe = Tk.Frame(bframe)
        tbframe.pack(side=Tk.LEFT, fill=Tk.X, expand=1)

        self.toolbar = NavigationToolbar(self.mpl_canvas, tbframe)
        self.toolbar.update()

    def get_canvas_width( self ):
        return int( self.mpl_canvas_widget.cget('width'))

    def build_canvas(self, cframe):
        figsize = (17,6) if is_low_resolution() else (23,8)
        fig = plt.figure(figsize=figsize)
        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        ax1 = fig.add_subplot(131, projection='3d') # it seems this must be done after creation of self.mpl_canvas
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        axes = [ax1, ax2, ax3]
        self.fig = fig
        self.axes = axes

        fig.tight_layout()
        fig.subplots_adjust(top=0.88, wspace=0.15, left=0.03, right=0.97)

    def build_params_frame(self, frame):
        width = int(self.mpl_canvas_widget.cget('width'))//3

        space = Tk.Frame(frame, width=width//3)
        space.pack(side=Tk.RIGHT)
        pframe = Tk.Frame(frame)
        pframe.pack(side=Tk.RIGHT)
        space = Tk.Frame(frame, width=width//3)
        space.pack(side=Tk.RIGHT)

        label = Tk.Label(pframe, text="DVF: ")
        label.grid(row=0, column=0)
        label = Tk.Label(pframe, text="%.3g" % self.sm_info.factor)
        label.grid(row=0, column=1)
        space = Tk.Label(pframe, text="   ")
        space.grid(row=0, column=2)
        label = Tk.Label(pframe, text="Conc Scale: ")
        label.grid(row=0, column=3)
        label = Tk.Label(pframe, text="%.3g" % self.sm_info.vscale)
        label.grid(row=0, column=4)

    def show(self):
        self._show(wait=False)
        self.waited_widget = Tk.Frame(self)
        self.wait_window(self.waited_widget)

    def ok(self):
        self.apply()
        self.close()

    def exit(self):
        self.close()

    def close(self):
        self.withdraw()
        self.update_idletasks()
        self.grab_release()
        self.waited_widget.destroy()

    def apply(self):
        self.applied = True

    def draw_init(self):
        xdata = self.xdata

        optimizer = DelayOptimizer(self.in_folder, xdata)
        self.slice_ = slice_ = optimizer.get_slice()
        smt, smx, smy = optimizer.get_initial_data()

        ax1, ax2, ax3 = self.axes
        self.fig.suptitle("Delay Volume Factor Adjustment for " + self.in_folder, fontsize=20)
        xdata.plot(ax1)
        init_sm_info = optimizer.sm_info
        self.draw_elution(ax2, xdata, slice_, smt, smx, smy, sm_info=init_sm_info)

        smt, smx, smy, sm_info = optimizer.get_optimized_data()
        self.draw_elution(ax3, xdata, slice_, smt, smx, smy, sm_info=sm_info, mapped=True)

        y_channel_p = optimizer.get_y_channel_point()
        for ax in [ax2, ax3]:
            ax.plot(*y_channel_p, 'o', color='yellow')

        self.mpl_canvas.draw()
        self.sm_info = sm_info

    def draw_elution(self, ax, xdata, slice_, smt, smx, smy, sm_info=None, mapped=False):
        adjust_text = " (optimized mapping)" if mapped else " (input data)"
        pick_pos = get_xray_picking()
        ax.set_title("Elution at Q=%.2g" % pick_pos + adjust_text, fontsize=16)
        if mapped:
            start = sm_info.start
            stop = len(xdata.e_y)
            e_x = np.arange(start, stop)
            e_y = xdata.e_y[start:stop]
        else:
            e_y = xdata.e_y
            e_x = np.arange(len(e_y))

        ax.plot(e_x, e_y, color='orange')

        if mapped:
            axt = ax
            smy *= sm_info.vscale
            sm_slice = slice(sm_info.start, sm_info.stop)
            x_ = smx[sm_slice]
            y_ = smy[sm_slice]
        else:
            axt = ax.twinx()
            axt.grid(False)
            x_ = smx
            y_ = smy

        axt.plot(x_, y_, color='blue', alpha=0.3)

        if sm_info is not None:
            for k in [sm_info.yc_peak, sm_info.lin_start, sm_info.lin_end]:
                axt.plot(smx[k], smy[k], 'o', color='red')

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)

        f, t = [slice_.start, slice_.stop]
        for k, p in enumerate([f, t]):
            ax.plot([p, p], [ymin, ymax], ':', color='gray')

        rect = mpl_patches.Rectangle(
                (f, ymin),      # (x,y)
                t - f,          # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )
        ax.add_patch(rect)
