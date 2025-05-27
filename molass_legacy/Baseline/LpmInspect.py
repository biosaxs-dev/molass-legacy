# coding: utf-8
"""
    Baseline.LpmInspect.py

    Copyright (c) 2020-2023, SAXS Team, KEK-PF
"""
import copy
import logging
import numpy as np
from bisect import bisect_right
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.DENSS.DenssUtils import fit_data
import molass_legacy.KekLib.DebugPlot as dplt
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from MatrixData import simple_plot_3d
from .Baseline import compute_baseline, better_integrative_curve, END_WIDTH
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from DataUtils import cut_upper_folders
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
import tkinter.ttk as ttk
from molass_legacy.KekLib.TkSupplements import BlinkingFrame
import molass_legacy.KekLib.CustomMessageBox as MessageBox
from .BaselineGuinier import BaselineGuinier, get_peaktop_slice

TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 14
USE_OPTINAL_BUTTONS = False
DEBUG = False

class EcurveProxyCds:
    def __init__(self, ecurve, j_slice):
        self.ecurve = ecurve
        self.j_start = j_slice.start
        if self.j_start is None:
            self.j_start = 0
        self.y = ecurve.y[j_slice]

    def get_major_peak_info(self):
        ret_info = []
        for rec in self.ecurve.get_major_peak_info():
            ret_info.append([j - self.j_start for j in rec])
        return ret_info

def demo(root, in_folder, md, integral=False, for_all_q=True, peakno=None, smooth_line=True, show_preview=False):
    print(in_folder)

    if show_preview:
        from molass_legacy.Tools.ThreeDimViewer import ThreeDimViewer
        dialog = ThreeDimViewer(root, md)
        dialog.show()

    fig = dplt.figure(figsize=(21, 11))
    demo_impl(fig, in_folder, md, integral=integral, for_all_q=for_all_q, peakno=peakno, smooth_line=smooth_line, show_lpm_fixed=True)
    dplt.show()

def demo_impl(fig, in_folder, md, j0=0, integral=False, for_all_q=True, peakno=None,
                smooth_line=False, show_scd=False,
                show_lpm_fixed=False, suppress_titles=False, end_slices=None, logger=None):

    xr = md.xr

    if False:
        xr.set_elution_curve()

    i_slice = slice(0, xr.i_slice.stop)
    j_slice = xr.j_slice
    i_index = xr.e_index - i_slice.start

    whole_data = xr.data.copy()
    data = whole_data[i_slice,j_slice]
    error = xr.error.data[i_slice,j_slice]
    q = xr.vector[i_slice]

    if peakno is None:
        peakno = xr.e_curve.primary_peak_no

    if show_scd:
        from Conc.ConcDepend import ConcDepend
        ecurve_for_scd = EcurveProxyCds(xr.e_curve, j_slice)
        cd = ConcDepend(q, data, error, ecurve_for_scd)
        scd_list = cd.compute_judge_info()
        scd0 = scd_list[peakno][1]

    # z = xr.e_curve.y[j_slice]
    z = data[i_index,:]

    if False:
        x = xr.e_curve.x
        y = xr.e_curve.y
        j0 = j_slice.start
        x_ = j0 + np.arange(len(z))

        dplt.push()
        fig, ax = dplt.subplots()
        ax.plot(x, y)
        ax.plot(x_, z)
        fig.tight_layout()
        dplt.show()
        dplt.pop()

    gs = GridSpec(3,3)

    fig.suptitle("Baseline Correction Inspection for " + cut_upper_folders(in_folder), fontsize=24)
    start = 0 if j_slice.start is None else j_slice.start

    topx = xr.e_curve.peak_info[peakno][1] - start
    print('topx=', topx, start+topx)

    q1 = q[i_index]
    size = data.shape[1]
    x = np.ones(size)*q1
    y = j0 + np.arange(size)

    diff_axes_info = []
    diff_ymin = None
    diff_ymax = None
    def update_diff_ylim(ax):
        ymin, ymax = ax.get_ylim()
        nonlocal diff_ymin, diff_ymax
        if diff_ymin is None or ymin < diff_ymin:
            diff_ymin = ymin
        if diff_ymax is None or ymax > diff_ymax:
            diff_ymax = ymax
    ylim_list = []
    axes_list = []

    if end_slices is None:
        end_slices = xr.e_curve.get_end_slices()

    if DEBUG:
        print("----------------- (2) end_slices=", end_slices)

    peaktop_slice = get_peaktop_slice(topx)
    bg = BaselineGuinier(data, error, q, i_index, peaktop_slice, end_slices=end_slices)
    curve0, curve1, curve2 = bg.get_scattering_curves()
    sg0, sg1, sg2 = bg.get_guinier_results()
    guinier_length0, guinier_length01, guinier_length2 = bg.get_guinier_lengths()

    yp = curve0[:,1]
    ype = curve0[:,2]
    yp_lpm1 = curve1[:,1]
    yp_lpm2 = curve2[:,1]

    try:
        if sg2.basic_quality is None or sg2.basic_quality == 0:
            gn_max = sg1.guinier_stop
        elif sg1.basic_quality is None or sg1.basic_quality == 0:
            gn_max = sg2.guinier_stop
        else:
            gn_max = max(sg1.guinier_stop, sg2.guinier_stop)
        gi = bisect_right(q, min(0.1, q[gn_max]*1.2))
    except:
        # log_exception(logger, "SimpleGuinier error: ")
        gi = bisect_right(q, 0.05)
        logger.warning("gi is set from bisect_right(q, 0.05) due to SimpleGuinier failure")
        if False:
            gslice = slice(0, gi)
            q2 = q[gslice]**2
            dplt.push()
            fig, (ax1, ax2) = dplt.subplots(ncols=2, figsize=(14, 7))
            ax1.plot(y, z)
            ax1.plot(y[topx], z[topx], 'o', color='red')
            for i, gy in enumerate([yp_lpm1, yp_lpm2]):
                ax2.plot(q2, np.log(gy[gslice]), label=str(i))
            ax2.legend()
            fig.tight_layout()
            dplt.pop()
            dplt.show()
            exit()

    gslice = slice(0, gi)
    q2 = q[gslice]**2

    def draw_scd_text(ax, scd):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        tx = xmin*0.95 + xmax*0.05
        ty = np.power(10, np.log10(ymin)*0.9 + np.log10(ymax)*0.1)
        inc = scd - scd0
        sign  = '-' if inc < 0 else '+'
        ax.text(tx, ty, "SCD=%.2f %s %.2f" % (scd0, sign, abs(inc)), va="center", fontsize=30, alpha=0.5)

    scd_pairs = []
    ret_axes = []
    for i, xray_baseline_type in enumerate(["No Correctin", "Linear Baseline", "Integral Baseline"]):
        gs_start = i
        gs_stop = (i+1)
        ax1 = fig.add_subplot(gs[gs_start:gs_stop,0])
        ax2 = fig.add_subplot(gs[gs_start:gs_stop,1])
        ax3 = fig.add_subplot(gs[gs_start:gs_stop,2])
        # axes_list.append((ax2, ax3))
        ret_axes.append((ax1, ax2, ax3))

        for ax in [ax2]:
            ax.axes.get_xaxis().set_ticks([])

        if i == 0:
            ax1.set_title("Elution Curve", fontsize=TITLE_FONTSIZE)
            ax2.set_title("Scattering Curve (Log-Log Plot)", fontsize=TITLE_FONTSIZE)
            ax3.set_title("Scattering Curve (Guinier Plot)", fontsize=TITLE_FONTSIZE)

        ax1.plot(y, z, color='orange')

        integral = i == 2

        if i == 0:
            b = np.zeros(len(z))
        elif i == 1:
            b = compute_baseline(z, integral=False, end_slices=end_slices)
        else:
            b, convex = better_integrative_curve(z, end_slices=end_slices)

        ax1.plot(y, b, ':', color='black', linewidth=3)
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax)

        f, t = y[[peaktop_slice.start, peaktop_slice.stop]]
        p = Rectangle(
                (f, ymin),      # (x,y)
                t - f,          # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )
        ax1.add_patch(p)

        ax2.set_yscale("log")
        ax2.set_xscale("log")

        alpha = 0.3 if smooth_line else 1
        ax2.plot(q, yp, label='No Correction', color='blue', alpha=alpha)
        ax3.plot(q2, np.log(yp[gslice]), label='No Correction', color='blue')

        if i == 0:
            ax2.legend(loc="lower left", fontsize=LEGEND_FONTSIZE)
            sg_ = sg0

        elif i == 1:
            ax2.plot(q, yp_lpm1, label='LPM Corrected', color='red', alpha=alpha)
            ylim_list.append(ax2.get_ylim())

            if smooth_line:
                labels = ["No Correction", "Varying %"]
                for k, yp_ in enumerate([yp,  yp_lpm1]):
                    qc, ac, ec, dmax = fit_data(q, yp_, ype)
                    ax2.plot(qc, ac, color='C%d' % k, linewidth=3, label="DENSS fit-data (%s)" % labels[k])

            ax3.plot(q2, np.log(yp_lpm1[gslice]), label='LPM Corrected', color='red')
            sg_ = sg1

        else:
            for slice_ in end_slices:
                ax1.plot(y[slice_], z[slice_], 'o', color='yellow', markersize=3)

            ax2.set_yscale("log")
            ax2.set_xscale("log")

            colors = ["blue", "red"]

            ax2.plot(q, yp_lpm2, label="Integral Corrected", color=colors[1], alpha=alpha)

            if smooth_line:
                for k, yp_ in enumerate([yp, yp_lpm2]):
                    qc, ac, ec, dmax = fit_data(q, yp_, ype)
                    ax2.plot(qc, ac, color=colors[k], linewidth=3, label="DENSS fit-data (%s)" % labels[k])

            ax3.plot(q2, np.log(yp_lpm2[gslice]), label="Integral Corrected", color='red')
            sg_ = sg2

        ymin, ymax = ax3.get_ylim()
        ax3.set_ylim(ymin, ymax)

        try:
            f = q[sg_.guinier_start]**2
            t = q[sg_.guinier_stop]**2
            p = Rectangle(
                    (f, ymin),      # (x,y)
                    t - f,          # width
                    ymax - ymin,    # height
                    facecolor   = 'green',
                    alpha       = 0.2,
                    label='Guinier Region',
                )
            ax3.add_patch(p)
        except:
            pass

        ax2.legend(loc="lower left", fontsize=LEGEND_FONTSIZE)
        ax3.legend(fontsize=LEGEND_FONTSIZE)

        if show_scd:
            cd = ConcDepend(q, lpm2.data, error, ecurve_for_scd)
            scd_list = cd.compute_judge_info()
            scd2 = scd_list[peakno][1]
            scd_pairs.append((scd1, scd2))

        def set_legend_ylim(ax):
            # ax.legend(fontsize=LEGEND_FONTSIZE)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymax-4.5, ymax)

        set_legend_ylim(ax2)
        ylim_list.append(ax2.get_ylim())

    for axes in diff_axes_info:
        for ax in axes[1:]:
            ax.set_ylim(diff_ymin, diff_ymax)

    ymin, ymax = np.average(np.array(ylim_list), axis=0)
    for axes, pair in zip(axes_list, scd_pairs):
        for ax, scd in zip(axes, pair):
            ax.set_ylim(ymin, ymax)
            draw_scd_text(ax, scd)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    if False:

        for ax2, ax2d in diff_axes_info:
            ay0 = ax2.axes.get_position().y0
            for ax in [ax2d]:
                bbox = ax.axes.get_position()
                bbox.y1 = bbox.y0 + (ay0 - bbox.y0)*0.8
                ax.axes.set_position(bbox)

    return ret_axes

class FromToSpinBoxPair(Tk.Frame):
    def __init__(self, parent, text=None, range_=slice(0, 3), bound=(0, 10)):
        Tk.Frame.__init__(self, parent)
        if text is not None:
            label = Tk.Label(self, text=text)
            label.pack(side=Tk.LEFT)

        self.f = Tk.IntVar()
        self.f.set(range_.start)
        self.t = Tk.IntVar()
        self.t.set(range_.stop)
        self.spinboxes = []
        for k, var in enumerate([self.f, self.t]):
            f_, t_ = bound
            sbox  = Tk.Spinbox(self, textvariable=var,
                              from_=f_, to=t_, increment=1,
                              justify=Tk.CENTER, width=6)
            sbox.pack(side=Tk.LEFT)
            self.spinboxes.append(sbox)
    def get_vars(self):
        return [self.f, self.t]

    def config(self, state):
        for w in self.spinboxes:
            w.config(state=state)

class LpmInspector(Dialog):
    def __init__(self, parent, md, suppress_titles=False):
        parent.config(cursor='wait')
        parent.update()
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.md = md
        self.j0 = md.xr.j_slice.start
        end_slices = get_setting("manual_end_slices")
        if end_slices is None:
             end_slices = md. xr.e_curve.get_end_slices()

        if DEBUG:
            print("---------------- (0): data.shape=", md.xr.data.shape, "end_slices=", end_slices)
        self.init_end_slices = end_slices
        self.primary_top_x = md.xr.e_curve.max_x - self.j0
        self.suppress_titles = suppress_titles
        self.ax = None
        self.tracing = True
        self.xray_baseline_type_auto = 0
        self.in_folder = get_setting('in_folder')
        Dialog.__init__(self, parent, "Baseline Inspector", visible=False)
        parent.config(cursor='')

    def cancel(self):
        # overiding cancel to cleanup self.fig
        # because the call to the destructor __del__ seems to be delayed.
        # note also that this is not only with "Cancel"
        # but also called in the normal exit with "OK".
        plt.close(self.fig)
        # print("LpmInspector: close fig")
        Dialog.cancel(self)

    def show(self):
        self._show()

    def body(self, frame):
        cframe = Tk.Frame(frame)
        cframe.pack()
        self.c1frame = c1frame = Tk.Frame(cframe)
        c1frame.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)
        c2frame = Tk.Frame(cframe)
        c2frame.pack(side=Tk.LEFT)
        bframe = Tk.Frame(frame)
        bframe.pack(fill=Tk.X)
        adjust_frame = Tk.Frame(bframe)
        adjust_frame.pack(side=Tk.LEFT)
        tframe = Tk.Frame(bframe)
        tframe.pack(side=Tk.LEFT, padx=30)

        self.buid_baseline_type_buttons(c1frame)

        self.fig = plt.figure(figsize=(18, 11))
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, c2frame)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()
        self.build_control_widgets(adjust_frame)
        self.redraw()

    def buid_baseline_type_buttons(self, c1frame):
        self.xray_baseline_type = Tk.IntVar()
        self.xray_baseline_type.set(get_setting("xray_baseline_type"))

        for i, w in enumerate([1, 2, 2, 2]):
            c1frame.grid_rowconfigure(i, weight=w)
        space = Tk.Frame(c1frame)
        space.grid(row=0, column=0)
        rb_frame0 = Tk.Frame(c1frame)
        rb_frame0.grid(row=1, column=0)
        rb_frame1 = Tk.Frame(c1frame)
        rb_frame1.grid(row=2, column=0)
        rb_frame2 = Tk.Frame(c1frame)
        rb_frame2.grid(row=3, column=0)

        b0 = ttk.Radiobutton(rb_frame0, variable=self.xray_baseline_type, value=0)
        b0.pack()
        t0 = Tk.Label(rb_frame0, text="None", font=(None, TITLE_FONTSIZE))
        t0.pack(padx=5)
        b1 = ttk.Radiobutton(rb_frame1, variable=self.xray_baseline_type, value=1)
        b1.pack()
        t1 = Tk.Label(rb_frame1, text="Linear", font=(None, TITLE_FONTSIZE))
        t1.pack(padx=5)
        b2 = ttk.Radiobutton(rb_frame2, variable=self.xray_baseline_type, value=2)
        b2.pack()
        t2 = Tk.Label(rb_frame2, text="Integral", font=(None, TITLE_FONTSIZE))
        t2.pack(padx=5)

    def build_control_widgets(self, pframe):
        to_trace_vars = []

        self.baseline_manually = Tk.IntVar()
        self.baseline_manually.set(get_setting("baseline_manually"))
        self.baseline_manually_cb = Tk.Checkbutton(pframe, variable=self.baseline_manually, 
                                                    text="Manual Spec.")
        self.baseline_manually_cb.pack(side=Tk.LEFT, padx=20)

        label = Tk.Label(pframe, text="Peak No: ")
        label.pack(side=Tk.LEFT)
        ecurve = self.md.xr.e_curve
        self.peakno = Tk.IntVar()
        self.peakno.set(ecurve.primary_peak_no + 1)

        max_peakno = len(ecurve.peak_info)
        self.spinbox = Tk.Spinbox(pframe, textvariable=self.peakno,
                                  from_=1, to=max_peakno, increment=1,
                                  justify=Tk.CENTER, width=6)
        self.spinbox.pack(side=Tk.LEFT, padx=10)
        to_trace_vars.append(self.peakno)

        if USE_OPTINAL_BUTTONS:
            self.smooth_lines = Tk.IntVar()
            self.smooth_lines.set(0)
            toggle_cb = Tk.Checkbutton(pframe, text="show smooth lines", variable=self.smooth_lines)
            toggle_cb.pack(side=Tk.LEFT, padx=20, pady=5)
            to_trace_vars.append(self.smooth_lines)

            self.show_scd = Tk.IntVar()
            self.show_scd.set(0)
            show_scd_cb = Tk.Checkbutton(pframe, text="show SCD", variable=self.show_scd)
            show_scd_cb.pack(side=Tk.LEFT, pady=5)
            to_trace_vars.append(self.show_scd)

        tx = int(round(self.primary_top_x))
        xmax = self.md.xr.data.shape[1]

        end_ranges = self.get_init_end_ranges(xmax)
        self.sbp_list = []
        sbp = FromToSpinBoxPair(pframe, text="Integral Left Base", range_=end_ranges[0], bound=(0, tx))
        sbp.pack(side=Tk.LEFT, padx=10)
        to_trace_vars += sbp.get_vars()
        self.sbp_list.append(sbp)
        sbp = FromToSpinBoxPair(pframe, text="Integral Right Base", range_=end_ranges[1], bound=(tx, xmax))
        sbp.pack(side=Tk.LEFT, padx=10)
        to_trace_vars += sbp.get_vars()
        self.sbp_list.append(sbp)

        self.redraw_btn_blink = BlinkingFrame(pframe)
        self.redraw_btn_blink.pack(side=Tk.LEFT, padx=10, pady=5)
        self.redraw_btn = Tk.Button(self.redraw_btn_blink, text="Redraw", width=10, command=self.redraw)
        self.redraw_btn.pack()
        self.redraw_btn_blink.objects = [self.redraw_btn]

        for v in to_trace_vars:
            v.trace("w", self.blink_start)

        self.baseline_manually_tracer(xmax, init_call=True)
        self.baseline_manually.trace("w", lambda *args: self.baseline_manually_tracer(xmax))
        self.xray_baseline_type.trace("w", self.xray_baseline_type_tracer)

    def xray_baseline_type_tracer(self, *args):
        if not self.tracing:
            return

        manually = self.baseline_manually.get()
        if not manually:
            yn = MessageBox.askyesno("Manual Specification Implied",
                'Changing baseline type implies "Manual Spec."\n'
                "where the automatic baseline type control will be disabled.\n"
                "Do you really mean it?",
                parent=self.c1frame)

            if yn:
                self.baseline_manually.set(1)
            else:
                self.tracing = False
                self.xray_baseline_type.set(self.xray_baseline_type_auto)
                self.update()
                self.tracing = True

    def get_init_end_ranges(self, xmax):
        end_ranges = []
        if DEBUG:
            print("j0=", self.j0, "init_end_slices=", self.init_end_slices)
        for k, slice_ in enumerate(self.init_end_slices):
            offset = self.j0 if k == 0 else 0
            # offset is a temporary fix for ...
            start = slice_.start
            if start < 0:
                start += xmax
            stop = slice_.stop
            if stop is None:
                stop = xmax
            end_ranges.append(slice(offset+start, offset+stop))
        if DEBUG:
            print("end_ranges=", end_ranges)
        return end_ranges

    def baseline_manually_tracer(self, xmax, init_call=False):
        manually = self.baseline_manually.get()
        state = Tk.NORMAL if manually else Tk.DISABLED
        for sbp in self.sbp_list:
            sbp.config(state=state)
        self.update_selector()
        if not manually and not init_call:
            end_ranges = self.get_init_end_ranges(xmax)
            for sbp, slice_ in zip(self.sbp_list, end_ranges):
                sbp.f.set(self.j0 + slice_.start)
                sbp.t.set(self.j0 + slice_.stop)

    def blink_start(self, *args):
        self.redraw_btn_blink.start()

    def redraw(self):
        self.redraw_btn_blink.stop()
        self.config(cursor='wait')
        self.update()
        self.fig.clf()
        peakno = self.peakno.get() - 1
        kwargs = {}
        if USE_OPTINAL_BUTTONS:
            smooth_line=self.smooth_lines.get()
            show_scd = self.show_scd.get()
            kwargs = {'smooth_line':smooth_line, 'show_scd':show_scd}
        end_slices = self.get_end_slices()
        if DEBUG:
            print("----------------- (1) (must be restricted)  end_slices=", end_slices)
        ret_axes = demo_impl(self.fig, self.in_folder, self.md, j0=self.j0, peakno=peakno, end_slices=end_slices,
                                logger=self.logger, **kwargs)
        self.ax = ret_axes[2][0]
        self.update_selector()
        self.mpl_canvas.draw()
        self.config(cursor='')

    def get_end_slices(self):
        end_slices = []
        for sbp in self.sbp_list:
            # note that - self.j0 means the restricted indeces
            vals = [v.get() - self.j0 for v in sbp.get_vars()]
            end_slices.append(slice(vals[0], vals[1]))
        return end_slices

    def update_selector(self):
        if self.baseline_manually.get() and self.ax is not None:
            self.span = SpanSelector(self.ax, self.onselect, 'horizontal', useblit=True,
                        props=dict(alpha=0.5))
        else:
            self.span = None

    def onselect(self, xmin, xmax):
        i = 0 if xmin < self.primary_top_x else 1
        sbp = self.sbp_list[i]
        sbp.f.set(int(round(xmin)))
        sbp.t.set(int(round(xmax)))

    def apply(self):
        print('apply')

        manually = self.baseline_manually.get()
        set_setting('baseline_manually', manually)
        if manually:
            xray_baseline_type = self.xray_baseline_type.get()
            end_slices = self.get_end_slices()
        else:
            xray_baseline_type = None
            end_slices = None
        set_setting('xray_baseline_type', xray_baseline_type)
        set_setting('manual_end_slices', end_slices)

    def save_the_figure(self, folder=None, file=None):
        import os
        from molass_legacy._MOLASS.SerialSettings import get_setting
        if folder is None:
            folder = get_setting('analysis_folder')
        if file is None:
            file = get_setting('analysis_name')
        path = os.path.join(folder, file)
        self.fig.savefig(path)
