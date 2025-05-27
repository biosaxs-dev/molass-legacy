# coding: utf-8
"""
    Baseline.LpmInspect_for_paper.py

    Copyright (c) 2020-2021, SAXS Team, KEK-PF
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.DENSS.DenssUtils import fit_data
import molass_legacy.KekLib.DebugPlot as dplt
from molass_legacy._MOLASS.SerialSettings import get_setting
from MatrixData import simple_plot_3d
from .Baseline import compute_baseline, better_integrative_curve, END_WIDTH
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from LPM import LPM_3d
from DataUtils import cut_upper_folders
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import BlinkingFrame

TITLE_FONTSIZE = 16
LEGEND_FONTSIZE = 11

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

def demo(root, in_folder, md, integral=False, for_all_q=True, peakno=None, whole_angle=True, smooth_line=True, show_preview=False):
    print(in_folder)

    if show_preview:
        from ThreeDimViewer import ThreeDimViewer
        dialog = ThreeDimViewer(root, md)
        dialog.show()

    fig = dplt.figure(figsize=(21, 11))
    demo_impl(fig, in_folder, md, integral=integral, for_all_q=for_all_q, peakno=peakno, whole_angle=whole_angle, smooth_line=smooth_line, show_lpm_fixed=True)
    dplt.show()

def demo_impl(fig, in_folder, md, integral=False, for_all_q=True, peakno=None, whole_angle=True,
                smooth_line=False, show_scd=False,
                show_lpm_fixed=False, suppress_titles=False):
    print('demo_impl: smooth_line=', smooth_line)

    xr = md.xr
    xr.set_elution_curve()
    if whole_angle:
        i_slice = slice(2, None)
    else:
        i_slice = xr.i_slice
    j_slice = xr.j_slice
    i_index = xr.e_index - i_slice.start

    whole_data = xr.data
    data = whole_data[i_slice,j_slice]
    error = xr.error.data[i_slice,j_slice]
    q = xr.vector[i_slice]
    if peakno is None:
        peakno = xr.e_curve.primary_peak_no

    if show_scd:
        from Conc.ConcDepend import ConcDepend
        ecurve_for_cds = EcurveProxyCds(xr.e_curve, j_slice)
        cd = ConcDepend(q, data, error, ecurve_for_cds)
        cds_list = cd.compute_judge_info()
        cds0 = cds_list[peakno][1]

    gs = GridSpec(8,2)

    fig.suptitle("Baseline Correction Inspection for " + cut_upper_folders(in_folder), fontsize=28)

    # z = xr.e_curve.y[j_slice]
    z = data[i_index,:]
    start = 0 if j_slice.start is None else j_slice.start

    topx = xr.e_curve.peak_info[peakno][1] - start
    print('topx=', topx, start+topx)

    q1 = q[i_index]
    size = data.shape[1]
    x = np.ones(size)*q1
    y = np.arange(size)

    hw = 5
    yp = np.average(data[:,topx-hw:topx+hw], axis=1)
    ype = np.average(error[:,topx-hw:topx+hw], axis=1)

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

    end_slices = xr.e_curve.get_end_slices()

    def draw_scd_text(ax, cds):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        tx = xmin*0.95 + xmax*0.05
        ty = np.power(10, np.log10(ymin)*0.9 + np.log10(ymax)*0.1)
        inc = cds - cds0
        sign  = '-' if inc < 0 else '+'
        ax.text(tx, ty, "CDS=%.2f %s %.2f" % (cds0, sign, abs(inc)), va="center", fontsize=30, alpha=0.5)

    cds_pairs = []
    for i, title in enumerate(["Linear Baseline", "Integral Baseline"]):
        gs_start = i*4
        gs_stop = (i+1)*4
        ax1 = fig.add_subplot(gs[gs_start:gs_stop,0])
        ax2 = fig.add_subplot(gs[gs_start:gs_stop,1])
        # ax2d = fig.add_subplot(gs[gs_stop-1:gs_stop,1])
        # ax3 = fig.add_subplot(gs[gs_start:gs_stop-1,2])
        # ax3d = fig.add_subplot(gs[gs_stop-1:gs_stop,2])
        # axes_list.append((ax2, ax3))

        for ax in [ax2]:
            ax.axes.get_xaxis().set_ticks([])

        # diff_axes_info.append((ax2, ax2d))

        ax1.set_title(title, fontsize=TITLE_FONTSIZE)
        ax1.plot(z, color='orange')

        integral = i == 1
        if i == 0:
            b = compute_baseline(z, integral=False, end_slices=end_slices)
        else:
            b, convex = better_integrative_curve(z, end_slices=end_slices)

        ax1.plot(b, ':', color='black', linewidth=3)
        # ax1.plot(topx, z[topx], 'o', color='blue')
        x = np.arange(len(z))
        # tslice = slice(topx-hw, topx+hw)
        # ax1.plot(x[tslice], z[tslice], color='blue')

        if i==1:
            for slice_ in end_slices:
                ax1.plot(y[slice_], z[slice_], 'o', color='yellow', markersize=3)

        if show_lpm_fixed:
            lpm0 = LPM_3d(data, integral=integral, for_all_q=True, e_index=i_index)
            yp_lpm0 = np.average(lpm0.data[:,topx-hw:topx+hw], axis=1)

        if i == 0:
            lpm1 = LPM_3d(data, ecurve_y=z, integral=integral, for_all_q=True, e_index=i_index)
            yp_lpm1 = np.average(lpm1.data[:,topx-hw:topx+hw], axis=1)

            ax2.set_yscale("log")
            ax2.set_xscale("log")

            title = "" if suppress_titles else "Correction Effects to the Scattering Curve (for all Q)"
            ax2.set_title(title, fontsize=TITLE_FONTSIZE)

            alpha = 0.3 if smooth_line else 1
            ax2.plot(q, yp, label='Average(top-%d:top+%d), No Correction' % (hw, hw),color='blue', alpha=alpha)
            if show_lpm_fixed:
                ax2.plot(q, yp_lpm0, label='Average(top-%d:top+%d), LPM (Fixed %%)' % (hw, hw), color='blue', alpha=alpha)
            ax2.plot(q, yp_lpm1, label='Average(top-%d:top+%d), LPM (Varying %%)' % (hw, hw), color='red', alpha=alpha)
            ylim_list.append(ax2.get_ylim())

            diff_y = (yp - yp_lpm1)/ype
            # ax2d.plot(q, diff_y, color='red')
            # update_diff_ylim(ax2d)

            if smooth_line:
                labels = ["No Correction", "Fixed %", "Varying %"]
                yp_fixed = yp_lpm0 if show_lpm_fixed else None
                for k, yp_ in enumerate([yp, yp_fixed, yp_lpm1]):
                    if k == 1 and not show_lpm_fixed:
                        continue
                    qc, ac, ec, dmax = fit_data(q, yp_, ype)
                    ax2.plot(qc, ac, color='C%d' % k, linewidth=3, label="DENSS fit-data (%s)" % labels[k])

            if show_scd:
                cd = ConcDepend(q, lpm1.data, error, ecurve_for_cds)
                cds_list = cd.compute_judge_info()
                cds1 = cds_list[peakno][1]

            def set_legend_ylim(ax):
                # ax.legend(fontsize=LEGEND_FONTSIZE)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymax-4.5, ymax)

            set_legend_ylim(ax2)
        else:
            # ax2.axes.get_yaxis().set_ticks([])
            # ax2d.axes.get_xaxis().set_ticks([])
            # ax2d.axes.get_yaxis().set_ticks([])
            pass

        if i == 1:
            lpm2 = LPM_3d(data, ecurve_y=z, integral=integral, for_all_q=False, e_index=i_index, end_slices=end_slices)
            yp_lpm2 = np.average(lpm2.data[:,topx-hw:topx+hw], axis=1)

            ax2.set_yscale("log")
            ax2.set_xscale("log")

            title = "" if suppress_titles else "Correction Effects to the Scattering Curve (diff expanded)"
            ax2.set_title(title, fontsize=TITLE_FONTSIZE)

            labels = ["No Correction", "Diff Corrected"]
            colors = ["blue", "red"]

            ax2.plot(q, yp, label='Average(top-%d:top+%d), %s' % (hw, hw, labels[0]), color=colors[0], alpha=alpha)
            ax2.plot(q, yp_lpm2, label='Average(top-%d:top+%d), %s' % (hw, hw, labels[1]), color=colors[1], alpha=alpha)

            if smooth_line:
                for k, yp_ in enumerate([yp, yp_lpm2]):
                    qc, ac, ec, dmax = fit_data(q, yp_, ype)
                    ax2.plot(qc, ac, color=colors[k], linewidth=3, label="DENSS fit-data (%s)" % labels[k])

            if show_scd:
                cd = ConcDepend(q, lpm2.data, error, ecurve_for_cds)
                cds_list = cd.compute_judge_info()
                cds2 = cds_list[peakno][1]
                cds_pairs.append((cds1, cds2))

            set_legend_ylim(ax2)
            ylim_list.append(ax2.get_ylim())

            diff_y = (yp - yp_lpm2)/ype
            # ax2d.plot(q, diff_y, color='red')
            # update_diff_ylim(ax2d)

        else:
            pass
            # ax3.axes.get_yaxis().set_ticks([])
            # ax3d.axes.get_xaxis().set_ticks([])
            # ax3d.axes.get_yaxis().set_ticks([])

    for axes in diff_axes_info:
        for ax in axes[1:]:
            ax.set_ylim(diff_ymin, diff_ymax)

    ymin, ymax = np.average(np.array(ylim_list), axis=0)
    for axes, pair in zip(axes_list, cds_pairs):
        for ax, cds in zip(axes, pair):
            ax.set_ylim(ymin, ymax)
            draw_scd_text(ax, cds)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    if False:

        for ax2, ax2d in diff_axes_info:
            ay0 = ax2.axes.get_position().y0
            for ax in [ax2d]:
                bbox = ax.axes.get_position()
                bbox.y1 = bbox.y0 + (ay0 - bbox.y0)*0.8
                ax.axes.set_position(bbox)

class LpmInspector(Dialog):
    def __init__(self, parent, md, suppress_titles=False):
        parent.config(cursor='wait')
        parent.update()
        self.parent = parent
        self.md = md
        self.suppress_titles = suppress_titles
        self.in_folder = get_setting('in_folder')
        Dialog.__init__(self, parent, "Baseline Inspector", visible=False)

    def show(self):
        self._show()

    def body(self, frame):
        cframe = Tk.Frame(frame)
        cframe.pack()
        bframe = Tk.Frame(frame)
        bframe.pack(fill=Tk.X)
        tframe = Tk.Frame(bframe)
        tframe.pack(side=Tk.LEFT)
        pframe = Tk.Frame(bframe)
        pframe.pack(side=Tk.RIGHT)

        self.fig = plt.figure(figsize=(14, 11))
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()
        self.build_control_widgets(pframe)
        demo_impl(self.fig, self.in_folder, self.md,
                    smooth_line=self.smooth_lines.get(), show_lpm_fixed=False,
                    suppress_titles=self.suppress_titles)
        self.mpl_canvas.draw()

        self.parent.config(cursor='')

    def build_control_widgets(self, pframe):
        label = Tk.Label(pframe, text="Peak No: ")
        label.pack(side=Tk.LEFT)
        ecurve = self.md.xr.e_curve
        self.peakno = Tk.IntVar()
        self.peakno.set(ecurve.primary_peak_no + 1)
        max_peakno = len(ecurve.peak_info)
        self.spinbox = Tk.Spinbox(pframe, textvariable=self.peakno,
                                  from_=1, to=max_peakno, increment=1,
                                  justify=Tk.CENTER, width=6)
        self.spinbox.pack(side=Tk.LEFT)

        self.smooth_lines = Tk.IntVar()
        self.smooth_lines.set(0)
        toggle_cb = Tk.Checkbutton(pframe, text="show smooth lines", variable=self.smooth_lines)
        toggle_cb.pack(side=Tk.LEFT, padx=20, pady=5)

        self.show_scd = Tk.IntVar()
        self.show_scd.set(0)
        show_cds_cb = Tk.Checkbutton(pframe, text="show SCD", variable=self.show_scd)
        show_cds_cb.pack(side=Tk.LEFT, pady=5)

        self.redraw_btn_blink = BlinkingFrame(pframe)
        self.redraw_btn_blink.pack(side=Tk.LEFT, padx=20, pady=5)
        self.redraw_btn = Tk.Button(self.redraw_btn_blink, text="Redraw", width=10, command=self.redraw)
        self.redraw_btn.pack()
        self.redraw_btn_blink.objects = [self.redraw_btn]

        for v in [self.peakno, self.smooth_lines, self.show_scd]:
            v.trace("w", self.blink_start)

    def blink_start(self, *args):
        self.redraw_btn_blink.start()

    def redraw(self):
        self.redraw_btn_blink.stop()
        self.config(cursor='wait')
        self.update()
        self.fig.clf()
        peakno = self.peakno.get() - 1
        smooth_line=self.smooth_lines.get()
        show_scd = self.show_scd.get()
        demo_impl(self.fig, self.in_folder, self.md, peakno=peakno, smooth_line=smooth_line, show_scd=show_scd)
        self.mpl_canvas.draw()
        self.config(cursor='')
        