# coding: utf-8
"""
    ConjugateGradientDemo.py

    Copyright (c) 2019-2020, SAXS Team, KEK-PF
"""
from bisect import bisect_right
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpl_patches
from matplotlib.widgets import Button

import logging
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from DataUtils import cut_upper_folders
from DecompUtils import CorrectedBaseline, make_range_info_impl, decompose_elution_better, get_peak2elem_dict
from molass_legacy.Models.ElutionCurveModels import EMGA
from molass_legacy.ElutionDecomposer import ElutionDecomposer
from ExtrapolationSolver import ExtrapolationSolver
from molass_legacy.PeaksetSelector import PeakSetSelector
from molass_legacy.DataStructure.AnalysisRangeInfo import convert_to_paired_ranges
from molass_legacy.KekLib.OurMatplotlib import get_color
import molass_legacy.KekLib.DebugPlot as dplt
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from ConjugateGradientThreeD import GdTreedPlot

ENABLE_GD_TRACE = True
TITLE_FONTSIZE  = 16
LINEWIDTH       = 3
DEMO_MIN_ANGLE  = 0.03
DEMO_ANGLE_SIZE = 40        # points

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class ConjugateGradientDemo(Dialog):
    def __init__(self, parent, in_folder, sd, mapper):
        self.logger = logging.getLogger(__name__)
        self.in_folder = cut_upper_folders(in_folder)
        self.sd = sd
        self.mapper = mapper
        self.popup_menu = None
        self.make_demo_range_info()
        self.use_elution_models_save = get_setting('use_elution_models')
        set_setting('use_elution_models', 1)

        Dialog.__init__( self, parent, "Gradient Descent Demo", visible=False )

    def __del__(self):
        # print('__del__: restore use_elution_models')
        set_setting('use_elution_models', self.use_elution_models_save)

    def make_demo_range_info(self):
        self.corbase_info = CorrectedBaseline(self.sd, self.mapper)
        self.decomp_ret = ret = decompose_elution_better(self.corbase_info, self.mapper, EMGA(),
                                                    logger=self.logger)

        self.x = ret.x
        self.y = ret.y
        self.uv_y = ret.uv_y
        self.opt_recs    = ret.opt_recs
        self.opt_recs_uv = ret.opt_recs_uv
        self.num_eltns = len(self.opt_recs_uv)

        ret_info = ret.get_range_edit_info(logger=self.logger)
        self.range_info = make_range_info_impl(ret.opt_recs_uv, ret_info)

        ret = convert_to_paired_ranges(self.range_info)
        self.cnv_ranges = ret[0]
        selector = PeakSetSelector(self.cnv_ranges, self.mapper.x_curve)
        self.peakset_info, elem_no = selector.select_demo_ranges_for_gd(self.uv_y, self.opt_recs_uv)
        opt_rec_uv = self.opt_recs_uv[elem_no]
        self.evaluator = opt_rec_uv[1]
        self.elm_color = get_color(len(self.opt_recs_uv) - 1)
        self.eval_range = self.evaluator.get_range_params(self.x)

        self.q = self.sd.intensity_array[0,:,0]
        self.qf = bisect_right(self.q, DEMO_MIN_ANGLE)
        self.qsize = DEMO_ANGLE_SIZE
        self.qt = self.qf + self.qsize

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()

        self.fig = fig = plt.figure(figsize=(21, 7))

        gs = GridSpec( 2, 3 )

        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.mpl_canvas.draw()

        self.ax01 = self.fig.add_subplot(gs[0, 0])
        self.ax02 = self.fig.add_subplot(gs[1, 0])
        self.ax1 = self.fig.add_subplot(gs[:, 1], projection='3d')
        self.ax2 = self.fig.add_subplot(gs[:, 2], projection='3d')
        fig.suptitle("Gradient Descent Demo using " + self.in_folder, fontsize=20)

        self.fig.subplots_adjust( top=0.9, bottom=0.08, left=0.05, right=0.97, wspace=0.1, hspace=0.3 )
        self.draw_elution()
        self.solve_extrapolation()
        self.draw_scattering()
        self.draw_data_3d()
        self.draw_gd_3d()
        pos = plt.axes([0.93, 0.04, 0.04, 0.04])
        self.animate_btn = Button(pos, 'Animate')
        self.animate_btn.on_clicked(self.show_animation)

    def buttonbox( self ):
        box = Tk.Frame(self)
        box.pack()
        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.bind("<Return>", self.ok)

    def draw_elution(self):
        ax = self.ax01
        ax.set_title("Focused range in the UV-elution curve", fontsize=TITLE_FONTSIZE)

        x = self.x
        y = self.uv_y
        ax.plot(x, y, color='blue')
        ax.plot(x, self.evaluator(x), ':', color=self.elm_color, linewidth=LINEWIDTH)

        ymin, ymax = ax.get_ylim()
        f, _, t = self.eval_range
        p = mpl_patches.Rectangle(
                (f, ymin),  # (x,y)
                t - f,   # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )
        ax.add_patch( p )

    def solve_extrapolation(self):
        from PreviewData import PreviewData, PreviewOptions
        self.x_curve = self.mapper.x_curve
        self.data = self.sd.intensity_array[:,:,1].T
        self.error = self.sd.intensity_array[:,:,2].T
        conc_factor = 5

        pdata = PreviewData(sd=self.sd, mapper=self.mapper, paired_ranges=self.cnv_ranges)
        popts = PreviewOptions(use_elution_models=False)

        self.solver = ExtrapolationSolver(pdata, popts)
        start, _, stop = self.eval_range

        # TODO: "de-couple" cnv_ranges, peakset_info from the solver

        demo_info = np.array([self.qf, self.qt]) if ENABLE_GD_TRACE else None
        self.solver_result = self.solver.extrapolate_wiser( start, stop, self.peakset_info,
                                demo_info=demo_info,
                                )

    def draw_scattering(self):
        A, B, Z, E, _, C = self.solver_result

        if False:
            f, _, t = self.eval_range
            fig = dplt.figure()
            axd = fig.gca()
            axd.set_title("solver_result C[0,:]")
            j = np.arange(f, t)
            axd.plot(j, C[0,:])
            fig.tight_layout()
            dplt.show()

        ax = self.ax02
        ax.set_title("Focused range in the Xray-scattering curve", fontsize=TITLE_FONTSIZE)

        _, top_x, _ = self.eval_range
        y = self.data[:,top_x]
        ax.plot(self.q, y)
        self.compute_p2m_scale(A, y)
        ax.plot(self.q, A*self.p2m_scale, color='orange')

        qf_ = self.q[self.qf]
        qt_ = self.q[self.qt]
        self.p1 = zf_ = A[self.qf]*self.p2m_scale
        self.p2 = zt_ = A[self.qt]*self.p2m_scale

        qf_ = self.q[self.qf]
        qt_ = self.q[self.qt]
        self.ax02.plot([qf_, qf_], [0, zf_], color='red', alpha=0.3, linewidth=5)
        self.ax02.plot([qt_, qt_], [0, zt_], color='green', alpha=0.3, linewidth=5)

        ymin, ymax = ax.get_ylim()
        p = mpl_patches.Rectangle(
                (qf_, ymin),  # (x,y)
                qt_ - qf_,   # width
                ymax - ymin,    # height
                facecolor   = 'orange',
                alpha       = 0.2,
            )
        ax.add_patch( p )

    def compute_p2m_scale(self, A, y):
        k = bisect_right(self.q, 0.05)
        slice_ = slice(k-2, k+3)
        self.p2m_scale = np.average(y[slice_]) / np.average(A[slice_])
        print('p2m_scale=', self.p2m_scale)

    def draw_data_3d(self):
        A = self.solver_result[0]

        f, _, t = self.eval_range
        esize = t - f
        y = np.arange(f, t)

        elution_z = self.evaluator(y)
        self.elution_z = elution_z = elution_z/np.max(elution_z) * self.p2m_scale

        color = get_color(0)
        qf = self.qf
        qt = self.qt
        qsize = self.qsize

        qt1 = qt + 1
        q_ = self.q[qf:qt1]
        xx, yy = np.meshgrid(q_, y)

        ax = self.ax1
        ax.set_title("Differences between the data and the factorized surface", fontsize=TITLE_FONTSIZE)

        _, top_x, _ = self.eval_range
        zf_ = self.p1
        zt_ = self.p2
        qf_ = self.q[qf]
        qt_ = self.q[qt]

        ax.plot([qf_, qf_], [top_x, top_x], [0, zf_], color='red', alpha=0.3, linewidth=5)
        ax.plot([qf_], [top_x], [zf_], 'o', color='red', markersize=10)

        zz_ = []
        for i in range(qf, qt1):
            q_ = self.q[i]
            x = np.ones(esize) * q_
            z = self.data[i, f:t]
            ax.plot(x, y, z, 'o', color=color, markersize=1)
            z_ = elution_z * A[i]
            zz_.append( z_ )
            if i in [qf, qt]:
                self.draw_diff_lines(ax, q_, y, z, z_)

            if False:
                print('A[%d]=' % i, A[i])
                fig = dplt.figure()
                axd = fig.gca()
                axd.set_title( 'A[%d]=%.3g' % ( i, A[i] ) )
                axd.plot(y, z)
                axd.plot(y, z_)
                fig.tight_layout()
                dplt.show()

        zz = np.array(zz_).T
        ax.plot_surface(xx, yy, zz, color='orange', alpha=0.3)

        ax.plot([qt_, qt_], [top_x, top_x], [0, zt_], color='green', alpha=0.3,linewidth=5)
        ax.plot([qt_], [top_x], [zt_], 'o', color='green', markersize=10)

    def draw_diff_lines(self, ax, q_, y, z, z_):
        for k, j in enumerate(y):
            ax.plot([q_, q_], [j, j], [z[k], z_[k]], color='pink', linewidth=3)

    def draw_gd_3d(self):
        self.gd3d = GdTreedPlot(self, ENABLE_GD_TRACE)
        self.gd3d.draw(self.ax2, TITLE_FONTSIZE)

    def show_animation(self, event):
        print('show_animation')
        from ConjugateGradientThreeD import GdAnimationDialog
        dialog = GdAnimationDialog(self, self.gd3d)
        dialog.show()
