"""
    CdInspection.py

    Copyright (c) 2020-2023, SAXS Team, KEK-PF
"""
import logging
from bisect import bisect_right
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkUtils import split_geometry
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from OurPlotUtils import draw_as_image
from DataUtils import cut_upper_folders
from SvdDenoise import get_denoised_data
from molass_legacy._MOLASS.SerialSettings import get_setting, get_xray_picking
from molass_legacy.KekLib.SciPyCookbook import smooth
from .ConcDepend import compute_min_norm_scaled, compute_min_norm_bq_s_aq

@ticker.FuncFormatter
def major_formatter(x, pos):
    apparant = False if pos is None else (pos - 1)%2 == 0
    return "%.0f" % x if apparant else ""

class CdInspectionDailog(Dialog):
    def __init__( self, parent, dialog, M, E, C, q, eslice, from_ax=None, xray_scale=None, plotter_info=None):
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.dialog = dialog
        self.M = M
        self.E = E
        self.svd = np.linalg.svd(M)
        self.C = C
        self.q = q
        self.xray_picking = get_xray_picking()
        self.q_index = bisect_right(q, self.xray_picking)
        self.q_index2 = bisect_right(q, self.xray_picking+0.01)
        self.q_limit = bisect_right(q, 0.2)

        if False:
            import molass_legacy.KekLib.DebugPlot as plt
            from MatrixData import simple_plot_3d
            print('M.shape=', M.shape, 'xray_scale=', xray_scale, 'q_limit=', self.q_limit, 'sigmas=', self.svd[1][0:2])
            c = C[0,:]
            c = c/np.max(c)
            plt.push()
            fig = plt.figure(figsize=(14,7))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122)
            simple_plot_3d(ax1, M)
            ax2.plot(c)
            fig.tight_layout()
            plt.show()
            plt.pop()

        self.eslice = eslice    # used just in title and logging
        self.plotter_info = plotter_info
        self.from_ax = from_ax
        self.xray_scale = xray_scale
        self.showing_sn = True
        self.popup_menu = None
        Dialog.__init__( self, parent, "Conc. Dependency Inspection", visible=False )

    def cancel(self):
        # overiding cancel to cleanup self.fig
        # because the call to the destructor __del__ seems to be delayed
        plt.close(self.fig)
        # print("CdInspectionDailog: closed fig")
        Dialog.cancel(self)

    def show(self):
        self._show()

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        self.fig = fig = plt.figure(figsize=(23,10))
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        in_folder = cut_upper_folders(get_setting('in_folder'))
        fig.suptitle("Conc. Dependency Inspection on Elution Range %d-%d in %s" % (self.eslice.start, self.eslice.stop, in_folder), fontsize=28)

        self.create_axes(fig)
        self.compute_scores()
        self.draw_elution(self.ax0)
        self.draw_singular_values(self.ax1)

        self.draw_curves()

        fig.subplots_adjust(top=0.9, bottom=0.07, wspace=0.3, hspace=0.4, left=0.05, right=0.97)

        self.mpl_canvas.draw()

    def draw_curves(self):
        axes = self.axes

        score1, sy2 = self.draw_diff_curve(axes[2,0], self.A1, self.A2, "$A_1 - scale*A_2$", self.scale)
        self.draw_cd1_curve(axes[0,0], self.A1, "$A_1$", sy2=sy2, curve_label2="$scale*A_2$")
        self.draw_cd2_curve(axes[1,0], self.A2, "$A_2$", yb=self.B2)

        score2, sy2 = self.draw_diff_curve(axes[2,1], self.AE1, self.AE2, "$A_1/E_1 - scale*A_2/E_2$", self.scale2)
        self.draw_cd1_curve(axes[0,1], self.AE1, "$A_1/E_1$", sy2=sy2, curve_label2="$scale*A_2/E_2$")
        self.draw_cd2_curve(axes[1,1], self.AE2, "$A_2/E_2$", yb=self.B2)

        score3, sy2, sy2_no_bnd, score3_smoothed = self.draw_diff_curve(axes[2,2], self.A2, self.B2, "$B_2 - scale*A_2$", self.scale, aq_scale=False, with_smoothed=True)
        self.draw_cd1_curve(axes[0,2], self.A2, "$A_2$", log_scale=False, curve_label2="$scale*B_2$")
        self.draw_cd2_curve(axes[1,2], self.B2, "$B_2$", log_scale=False, sy2=sy2, sy2_no_bnd=sy2_no_bnd)

        sigmas = self.svd[1]
        self.logger.info("cd_scores=[%.3g, %.3g, %.3g, %.3g], sigmas=[%.3g, %.3g] for elution range(%d, %d)", score1, score2, score3, score3_smoothed, sigmas[0], sigmas[1], self.eslice.start, self.eslice.stop)

    def buttonbox( self ):
        bottom_frame = Tk.Frame(self)
        bottom_frame.pack(fill=Tk.BOTH, expand=1)

        width = int(self.mpl_canvas_widget.cget('width'))
        padx = width*0.05

        tframe = Tk.Frame(bottom_frame)
        tframe.pack(side=Tk.LEFT, padx=padx)
        self.toolbar = NavigationToolbar( self.mpl_canvas, tframe )
        self.toolbar.update()

        space = Tk.Frame(bottom_frame, width=width*0.25)
        space.pack(side=Tk.RIGHT)

        box = Tk.Frame(bottom_frame)
        box.pack(side=Tk.RIGHT)

        w = Tk.Button(box, text="Close", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def create_axes(self, fig):
        gs = GridSpec(3,4)
        self.ax0 = fig.add_subplot(gs[0,0])
        self.ax1 = fig.add_subplot(gs[1:3,0])

        axes = []
        for i in range(3):
            ax_row = []
            for j in range(1,4):
                ax_row.append(fig.add_subplot(gs[i,j]))
            axes.append(ax_row)

        self.axes = axes = np.array(axes)

    def compute_scores(self):
        M1 = get_denoised_data(self.M, rank=1, svd=self.svd)
        C1 = self.C[0,:]
        C1 = C1/np.max(C1)
        C1inv = np.linalg.pinv(C1[np.newaxis,:])
        P1 = M1 @ C1inv
        E1 = np.sqrt((self.E**2) @ (C1inv**2))

        self.A1 = P1[:,0]
        self.AE1 = P1[:,0]/E1[:,0]

        M2 = get_denoised_data(self.M, rank=2, svd=self.svd)
        C2 = np.array([C1, C1**2])

        C2inv = np.linalg.pinv(C2)
        P2 = M2 @ C2inv
        E2 = np.sqrt((self.E**2) @ (C2inv**2))

        self.A2 = P2[:,0]
        self.AE2 = P2[:,0]/E2[:,0]
        self.B2 = P2[:,1]

        i_slice = slice(self.q_index, self.q_index2)
        if self.xray_scale is None:
            s1 = np.average(self.A1[i_slice])
            s2 = np.average(self.A2[i_slice])
            self.scale = np.sqrt(s1*s2)
        else:
            self.scale = self.xray_scale
        s1 = np.average(self.AE1[i_slice])
        s2 = np.average(self.AE2[i_slice])
        self.scale2 = np.sqrt(s1*s2)

    def draw_elution(self, ax):
        from_ax = self.from_ax
        if from_ax is None:
            plotter, i, pno = self.plotter_info
            self.logger.info("draw_elution: i=%d, pno=%d", i, pno)
            if i == 1:
                plotter.draw_fig2(ax, title=False, peak_no=pno)
            else:
                plotter.draw_mapped(ax, title=False, peak_no=pno)
        else:
            ax.set_axis_off()
            draw_as_image(ax, self.dialog.fig, from_ax, exp_elements=True)

    def draw_singular_values(self, ax):
        ax.set_title("Top Five Singular Values", fontsize=16)
        ax.set_xlabel("Value No", fontsize=12)
        ax.set_ylabel("Scale", fontsize=12)
        ax.xaxis.set_major_formatter(major_formatter)

        sigmas = self.svd[1]
        n = min(5, len(sigmas))
        """
            for cases where len(sigmas) < 5 as in 20190315_2
        """
        ax.plot(np.arange(n), sigmas[0:n], ':', marker='o')

    def draw_cd1_curve(self, ax, y, curve_label, log_scale=True, sy2=None, curve_label2=None):
        if log_scale:
            ax.set_yscale("log")
        ax.set_title("%s Curve" % curve_label, fontsize=16)

        ax.plot(self.q, y, color='C1', label=curve_label)
        if sy2 is not None:
            color = 'C2' if log_scale else 'pink'
            ax.plot(self.q, sy2, color=color, label=curve_label2, alpha=0.5)

        ax.legend()

    def draw_cd2_curve(self, ax, y, curve_label, yb=None, log_scale=True, sy2=None, sy2_no_bnd=None):
        if log_scale:
            ax.set_yscale("log")
        ax.set_title("%s Curve" % curve_label, fontsize=16)

        color = 'C2' if log_scale else 'pink'
        ax.plot(self.q, y, color=color, label=curve_label)

        if yb is not None:
            axt = ax.twinx()
            axt.grid(False)
            axt.plot(self.q, yb, color='pink', label='$B_2$')
            axt.legend(bbox_to_anchor=(1, 0.82), loc='upper right')

            # delete ticks which appear due to twinx()
            ax.tick_params(which='both', left=False)

        if sy2 is not None:
            ax.plot(self.q, sy2, color='C1', label='$scale*A_2$')

        if sy2_no_bnd is not None:
            ax.plot(self.q, sy2_no_bnd, color='yellow', label='$scale*A_2$ (no bound)')

        ax.legend()

    def draw_diff_curve(self, ax, y1, y2, curve_label, scale, aq_scale=True, with_smoothed=True):
        if aq_scale:
            sy2 = compute_min_norm_scaled(y1, y2)
            y = y1 - sy2
        else:
            sigmas = self.svd[1]
            init_scale = sigmas[1]/sigmas[0]
            sy2, sy2_no_bnd = compute_min_norm_bq_s_aq(y2, y1, init_scale, self.logger, ret_no_bnd=True)
            y = y2 - sy2

        ax.set_title("Scaled Difference Curve", fontsize=16)

        aslice = slice(0,self.q_limit)
        y_ = y[aslice]
        cd_score = np.sqrt(np.average(y_**2))*100/scale

        if not aq_scale and sy2_no_bnd is not None:
            y = y2 - sy2_no_bnd
            ax.plot(self.q, y, color='yellow', label=curve_label + ' (no bound)')
            y_ = y[aslice]
            cd_score_no_bnd = np.sqrt(np.average(y_**2))*100/scale
        else:
            cd_score_no_bnd = None

        if aq_scale:
            ax.plot(self.q, y, color='gray', label=curve_label)
            self.draw_score_text(ax, y[-1], cd_score, cd_score_no_bnd)
        else:
            if with_smoothed:
                b = smooth(y2)
                a = smooth(y1)
                sa, min_scale = compute_min_norm_bq_s_aq(b, a, init_scale, ret_scale=True)
                y_ = b - sa
                cd_score_smoothed = np.sqrt(np.average(y_[aslice]**2))*100/scale
                ax.plot(self.q, y_, color='blue', label='$smooth\ B_2 - scale* smooth\ A_2$')

                ax.plot(self.q, y, color='cyan', alpha=0.5, label=curve_label)
                self.draw_score_text(ax, y[-1], cd_score_smoothed, cd_score, min_scale=min_scale)
            else:
                ax.plot(self.q, y, color=color, label=curve_label)
                self.draw_score_text(ax, y[-1], cd_score, None)

        ax.legend()

        if aq_scale:
            return cd_score, sy2
        else:
            if with_smoothed:
                return cd_score, sy2, sy2_no_bnd, cd_score_smoothed
            else:
                return cd_score, sy2, sy2_no_bnd

    def draw_score_text(self, ax, yp, score, cd_score_no_bnd, min_scale=None):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        tx = xmin*0.4 + xmax*0.6
        yratio = (yp - ymin)/(ymax - ymin)
        if yratio > 0.5:
            ty = ymin*0.4 + yp*0.6
        else:
            ty = yp*0.6 + ymax*0.4
        ty_ = ty if cd_score_no_bnd is None else ty + (ymax - ymin)*0.11
        ax.text(tx, ty_, "SCD=%.3g" % score, ha='center', va="center", alpha=0.5, fontsize=40, color='blue')
        if cd_score_no_bnd is not None:
            ty_ = ty - (ymax - ymin)*0.11
            ax.text(tx, ty_, "SCD=%.3g" % cd_score_no_bnd, ha='center', va="center", alpha=0.5, fontsize=40, color='cyan')
        if min_scale is not None:
            ty_ = ty - (ymax - ymin)*0.34
            ax.text(tx, ty_, "scale=%.3g" % min_scale, ha='center', va="center", alpha=0.5, fontsize=40, color='gray')


    def save_the_figure(self, folder, pno, ad=None):
        import os
        from molass_legacy._MOLASS.SerialSettings import get_setting
        analysis_name = get_setting('analysis_name')
        ad_ = '' if ad is None else '-%d' % ad
        filename = analysis_name.replace( 'analysis', 'figure' ) + '-%d' % pno + ad_
        path = os.path.join( folder, filename )
        self.fig.savefig( path )
