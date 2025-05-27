"""
    Optimizer.GlobalInspector.py

    Copyright (c) 2020,2023, SAXS Team, KEK-PF
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from molass_legacy._MOLASS.SerialSettings import get_setting
from DataUtils import get_in_folder
from .OptimalElution import compute_optimal_elution
from MatrixData import simple_plot_3d
from RawData import get_row_index_impl
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar

BUG_FIX_WITH_REDRAW = False
TITLE_FONTSIZE = 16
MATRIX_SYM_FONTSIZE = 200

Q_LIST = [0.02, 0.05, 0.1]
MODELS = ["EMG", "EGH"]

FIG_SIZE = (21,12)
AXIS_NUM_H_DIVS = 5

class GlobalInspector:
    def __init__(self, D, E, ecurve, legacy_info=None, logger=None):
        self.logger = logging.getLogger(__name__) if logger is None else logger
        results = []
        norm_ratios = []
        for model_type in [0, 1]:
            M, rank, C_init, optC, recs = compute_optimal_elution(D, E, ecurve, model_type, legacy_info=legacy_info, logger=self.logger)
            P_init = M @ np.linalg.pinv(C_init)
            norm0 = np.linalg.norm(P_init@C_init - M)

            P = M @ np.linalg.pinv(optC)
            norm1 = np.linalg.norm(P@optC - M)
            norm_ratios.append(norm1/norm0)
            results.append((M, rank, optC, P, norm0, norm1, recs))
        self.results = np.array(results)
        self.norm_ratios = norm_ratios
        self.ecurve = ecurve

    def plot_results(self, fig, D, qvector, same_scale_residuals=True):
        in_folder = get_setting('in_folder')

        ecurve = self.ecurve
        results = self.results
        norm_ratios = self.norm_ratios

        folder_name = get_in_folder(in_folder=in_folder)
        fig.suptitle("Global Fitting Inspection for " + folder_name + "; Norm Reduction Ratios: EMG(%.3g), EGH(%.3g)" % tuple(norm_ratios), fontsize=20)
        nrows, ncols = 3, 3
        gs = GridSpec(nrows*AXIS_NUM_H_DIVS, ncols)
        axes = []
        axes_d = []
        for i in range(nrows):
            row = []
            row_d = []
            i_start = i*AXIS_NUM_H_DIVS
            for j in range(ncols):
                if i+j == 0:
                    projection = '3d'
                    i_stop = i_start + AXIS_NUM_H_DIVS
                else:
                    projection = None
                    i_stop = i_start + (AXIS_NUM_H_DIVS - 1)
                ax = fig.add_subplot(gs[i_start:i_stop,j], projection=projection)
                # ax.get_xaxis().set_visible(False)
                row.append(ax)
                if projection is None:
                    ax_d = fig.add_subplot(gs[i_stop,j])
                    ax_d.get_xaxis().set_visible(False)
                else:
                    ax_d = None
                row_d.append(ax_d)

            axes.append(row)
            axes_d.append(row_d)

        axes = np.array(axes)
        axes_d = np.array(axes_d)
        ax00 = axes[0,0]
        ax01 = axes[0,1]
        ax02 = axes[0,2]

        sub_fontsize = 16
        std_Q = Q_LIST[0]

        x = ecurve.x
        y = ecurve.y

        ax00.set_title("3D View", y=1.07, fontsize=sub_fontsize)
        simple_plot_3d(ax00, D, x=qvector)

        residual_ylims = []
        residual_indeces = []
        for j, (name, ax, recs) in enumerate(zip(MODELS, axes[0,1:], results[:,-1])):
            ax.set_title("%s Initial Decomposition at %.2g" % (name, std_Q), y=0.98, fontsize=sub_fontsize)
            ax.plot(x, y, color='orange', alpha=0.5, linewidth=3, label='data')
            ty = np.zeros(len(y))
            for k, rec in enumerate(recs):
                f = rec.evaluator
                y_ = f(x)
                ax.plot(x, y_, ':', linewidth=3, label='element %d' % (k+1))
                ty += y_
            ax.plot(x, ty, ':', color='red', linewidth=3, label='model total')
            ax.legend()
            index_d = (0,j+1)
            ax_d = axes_d[index_d]
            ax_d.plot(x, ty-y, color='gray', label='residual')
            ax_d.legend(bbox_to_anchor=(1, 0.9), loc='lower right')
            residual_indeces.append(index_d)
            residual_ylims.append(ax_d.get_ylim())

        average_width = get_setting('num_points_intensity')
        hw = average_width//2

        for i, (M, rank, optC, P, norm0, norm1, recs) in enumerate(results):
            name = MODELS[i]
            for j, qv in enumerate(Q_LIST):
                ax = axes[i+1,j]
                ax.set_title("%s Result Decomposition at %.2g" % (name, qv), y=0.98, fontsize=sub_fontsize)
                k = get_row_index_impl(qvector, qv)
                ey = np.average(D[k-hw:k+hw+1,:], axis=0)
                ax.plot(x, ey, color='orange', alpha=0.5, linewidth=3, label='data')

                ty = np.zeros(len(y))
                if True:
                    for m, c in enumerate(optC):
                        scale = P[k,m]
                        y_ = scale*c
                        ax.plot(x, y_, ':', linewidth=3, label='element %d' % (m+1))
                        ty += y_
                else:
                    for m, rec in enumerate(recs):
                        scale = P[k,m]
                        f = rec.evaluator
                        y_ = scale*f(x)
                        ax.plot(x, y_, ':', linewidth=3, label='element %d' % (m+1))
                        ty += y_
                ax.plot(x, ty, ':', color='red', linewidth=3, label='model total')
                ax.legend()
                index_d = (i+1,j)
                ax_d = axes_d[index_d]
                ax_d.plot(x, ty-ey, color='gray', label='residual')
                ax_d.legend(bbox_to_anchor=(1, 0.9), loc='lower right')
                residual_indeces.append(index_d)
                residual_ylims.append(ax_d.get_ylim())

        if same_scale_residuals:
            ylims = np.array(residual_ylims)
            ymin = np.min(ylims[:,0])
            ymax = np.max(ylims[:,1])
            w = -0.1
            ymin_ = ymin*(1-w) + ymax*w
            w = 1.1
            ymax_ = ymin*(1-w) + ymax*w
            for index in residual_indeces:
                axes_d[index].set_ylim(ymin_, ymax_)

        fig.subplots_adjust(top=0.93, bottom=0.02, left=0.05, right=0.97, hspace=1.1)

class GlobalInspectorDialog(Dialog):
    def __init__(self, parent, dialog, sd):
        D, E, qvector, ecurve = sd.get_xr_data_separate_ly()
        legacy_info = [sd, dialog.mapper, dialog.corbase_info]
        self.inspector = GlobalInspector(D, E, ecurve, legacy_info)
        self.D = D
        self.qvector = qvector

        Dialog.__init__(self, parent, "Global Inspector", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        bframe = Tk.Frame(body_frame)
        bframe.pack(fill=Tk.X)
        tframe = Tk.Frame(bframe)
        tframe.pack(side=Tk.LEFT)
        btn_frame = Tk.Frame(bframe)
        btn_frame.pack(side=Tk.RIGHT, padx=40)

        self.fig = fig = plt.figure(figsize=FIG_SIZE)
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        self.inspector.plot_results(fig, self.D, self.qvector)
        self.mpl_canvas.draw()

        self.add_buttons(btn_frame)

    def buttonbox(self):
        pass

    def add_buttons(self, frame):
        self.same_scale = Tk.IntVar()
        self.same_scale.set(1)
        w = Tk.Checkbutton(frame, text="Same scale for residuals", variable=self.same_scale)
        w.pack(side=Tk.LEFT, padx=10)
        w = Tk.Button(frame, text="Close", command=self.cancel)
        w.pack(side=Tk.LEFT, padx=10)
        self.same_scale.trace("w", self.same_scale_tracer)

    def same_scale_tracer(self, *args):
        same_scale_residuals = self.same_scale.get() == 1
        fig = self.fig
        fig.clf()
        self.inspector.plot_results(fig, self.D, self.qvector, same_scale_residuals)
        self.mpl_canvas.draw()
