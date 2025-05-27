# coding: utf-8
"""
    QmmDialog.py

    Copyright (c) 2020-2024, SAXS Team, KEK-PF
"""
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon
import matplotlib.ticker as ticker
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar, get_color
import molass_legacy.KekLib.DebugPlot as dplt
from OurPlotUtils import draw_as_image
from DataUtils import get_in_folder
from .GroupingMatrix import GroupingMatrix, num_to_char
from molass_legacy.AutorgKekAdapter import AutorgKekAdapter
from molass_legacy._MOLASS.SerialSettings import get_setting
from .QmmDenssMenu import QmmDenssMenu

DISTINCTION_Q = 0.15
DISTINCTION_WL = 220
APPLY_DENSS_FIT = True
if APPLY_DENSS_FIT:
    from molass_legacy.DENSS.DenssUtils import fit_data

class QmmDialog(Dialog):
    def __init__(self, parent, controller, **kwargs):
        self.applied = False
        self.datasets = controller.datasets
        self.qmm = controller.qmm
        self.model_name = self.qmm.get_model_name()
        self.y_list = controller.y_list
        self.data_list = controller.data_list
        cd = get_setting('conc_dependence')
        self.consider_ipe = cd == 2
        Dialog.__init__(self, parent, title="%s Quad Mixture Model" % self.model_name, visible=False)

    def show(self):
        self._show()

    def body(self, bframe):
        self.base_frames = []
        for k, dtype in enumerate(["UV", "Xray"]):
            k2 = k*2
            k_slice = slice(k2, k2+2)
            frame = QmmFrame(bframe, self, self.datasets, dtype, self.qmm,
                                self.y_list[k_slice])
            frame.pack()
            self.base_frames.append(frame)

        self.current_frame = 1
        self.base_frames[1 - self.current_frame].pack_forget()

    def get_current_frame(self):
        return self.base_frames[self.current_frame]

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack(fill=Tk.X, expand=1, padx=10)

        for j in range(3):
            box.columnconfigure(j, weight=1)

        w = Tk.Button(box, text="◀ Main", width=10, command=self.cancel)
        w.grid(row=0, column=0, sticky=Tk.W, padx=10, pady=5)

        sw_text = self.get_switch_text()
        w = Tk.Button(box, text=sw_text, command=self.toggle)
        w.grid(row=0, column=1)
        self.switch_button = w

        w = Tk.Button(box, text="▶ Serial Analysis", command=self.ok, state=Tk.DISABLED)
        w.grid(row=0, column=2, sticky=Tk.E, padx=10, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def get_switch_text(self):
        dtype = 'UV' if self.current_frame == 1 else 'Xray'
        return "◇ Switch to " + dtype

    def toggle(self):
        i = self.current_frame
        j = 1 - i
        self.base_frames[i].pack_forget()
        self.base_frames[j].pack()
        self.current_frame = j
        sw_text = self.get_switch_text()
        self.switch_button.config(text=sw_text)

    def apply(self):
        self.applied = True

    def get_rg_array(self):
        return self.base_frames[1].get_rg_array()

class QmmFrame(Tk.Frame):
    def __init__(self, parent, dialog, datasets, dtype, qmm, y_list):
        Tk.Frame.__init__(self, parent)
        self.dialog = dialog
        self.model_name = dialog.model_name
        self.datasets = datasets
        self.S_init = None
        self.dtype = dtype
        self.rg_set_list = None
        self.i_pos_values = datasets.get_i_pos_values(dtype)
        self.i_pos_format = '@WL=%.3g' if dtype == 'UV' else '@Q=%.2g'
        if dtype=='UV':
            self.center_title = "Component Absorbance Curves"
            self.no_base = 0
            self.dataset = self.datasets.pair[0]
            self.pos_format = '@WL=%.3g'
            self.pos_list = [DISTINCTION_WL]
            self.consider_ipe = False
        else:
            self.logger = logging.getLogger(__name__)
            self.center_title = "Component Scattering Curves"
            self.no_base = 2
            self.dataset = self.datasets.pair[1]
            self.pos_format = '@Q=%.3g'
            self.pos_list = [DISTINCTION_Q]
            self.consider_ipe = dialog.consider_ipe

        self.j0 = self.dataset.j0
        main_frame = Tk.Frame(self)
        main_frame.pack()
        bottom_frame = Tk.Frame(self)
        bottom_frame.pack(fill=Tk.X, expand=1)
        left_frame = Tk.Frame(main_frame)
        left_frame.pack(side=Tk.LEFT)
        right_frame = Tk.Frame(main_frame)
        right_frame.pack(side=Tk.LEFT)
        toolbar_frame = Tk.Frame(bottom_frame)
        toolbar_frame.pack(side=Tk.LEFT)
        button_frame = Tk.Frame(bottom_frame)
        button_frame.pack(side=Tk.RIGHT)

        self.build_canvases(left_frame, dtype, qmm, y_list, toolbar_frame)
        self.build_panels(right_frame, button_frame)
        self.update()
        self.copy_to_panel_fig(self.fig, self.axes[0,:])
        self.popup_menu = None

    def on_mpl_button_press(self, event):
        if event.xdata is None:
            return

        if event.button == 3:
            from molass_legacy.KekLib.TkUtils import split_geometry
            self.create_popup_menu()
            w, h, x, y = split_geometry(self.dialog.geometry())
            self.popup_menu.post(x + event.x + 20, y + h - event.y - 50)
            return

    def create_popup_menu(self):
        if self.popup_menu is None:
            self.popup_menu = Tk.Menu(self, tearoff=0)
            self.popup_menu.add_command(label='Show 3D Figure', command=self.show_3d_figure)
            self.popup_menu.add_command(label='Vaious Denoise Figures', command=self.show_various_denoise_figures)

    def grouped(self, S):
        if self.S_init is None or S is None:
            return False

        sdiff = np.sum(np.abs(S - self.S_init))
        return sdiff > 0

    def build_canvases(self, frame, dtype, qmm, y_list, toolbar_frame):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
        self.fig = fig
        self.axes = axes
        if self.dtype == 'Xray':
            self.twin_axes = [ax.twinx() for ax in axes[:,2]]

        self.mpl_canvas = FigureCanvasTkAgg(fig, frame)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        if self.dtype == 'Xray':
            self.mpl_canvas.mpl_connect('button_press_event', self.on_mpl_button_press)

        fig.suptitle("Decomposition of %s Data from %s using %s-QMM" % (dtype,  get_in_folder(), self.model_name), fontsize=30)

        xy_list, C_list, gy_list, mu_list, sigma_list = self.get_draw_lists(qmm, y_list)
        self.num_components = qmm.K
        self.mu_list = mu_list
        self.sigma_list = sigma_list
        self.draw_all(xy_list, C_list, gy_list)
        self.xy_list = xy_list
        self.C_list = C_list
        self.gy_list = gy_list

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        self.modify_major_xtick_positions(shrink_axis=True)

        self.mpl_canvas.draw()
        self.toolbar = NavigationToolbar(self.mpl_canvas, toolbar_frame)
        self.toolbar.update()

    def get_draw_lists(self, qmm, y_list):
        xy_list = []
        C_list = []
        gy_list = []
        mu_list = []
        sigma_list = []
        for n, y in enumerate(y_list):
            mm = qmm.get_sub(self.no_base+n)
            x = np.arange(len(y))
            C, gy = mm.get_anim_C(x, y, -1, total=True)
            xy_list.append((x,y))
            C_list.append(C)
            gy_list.append(gy)
            mu_list.append(mm.get_peak_mean_x())
            sigma_list.append(mm.sigma)
        return xy_list, C_list, gy_list, mu_list, sigma_list

    def draw_all(self, xy_list, C_list, gy_list, k_rows_list=None, effective_sizes=None):
        axes = self.axes
        self.draw_elution(axes[:,0], xy_list, C_list, gy_list, effective_sizes)
        ret_list = self.draw_featuring_curves(axes[:,1], C_list, effective_sizes)
        point_set_list = self.draw_distinction(axes[:,2], xy_list, ret_list, k_rows_list)
        if k_rows_list is None:
            self.point_set_list = np.array(point_set_list)

    def draw_elution(self, axes, xy_list, C_list, gy_list, effective_sizes=None):
        for n, (ax, xy, C, gy) in enumerate(zip(axes, xy_list, C_list, gy_list)):
            ax.cla()
            if n == 0:
                ax.set_title("Component Elution Curves", fontsize=16)
            x, y = xy
            x_ = x + self.j0
            ax.plot(x_, y, label='input')
            ax.plot(x_, gy, label='total')
            size = None if effective_sizes is None else effective_sizes[n]
            self.plot_components(ax, x_, C[:size])
            self.draw_i_pos_text(n, ax)
            ax.legend()

    def plot_components(self, ax, x, cgys):
        ret_lines = []
        for k, gy in enumerate(cgys):
            line, = ax.plot(x, gy, ':', label='c-%s' % num_to_char(k))
            ret_lines.append(line)
        return ret_lines

    def draw_i_pos_text(self, n, ax, fontsize=30, xpos=0.2):
        text = self.i_pos_format % self.i_pos_values[n]
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        tx = xmin*(1-xpos) + xmax * xpos
        ty = ymin*0.1 + ymax * 0.9
        ax.text(tx, ty, text, ha='center', va='center', fontsize=fontsize, alpha=0.1)

    def draw_featuring_curves(self, axes, C_list, effective_sizes=None):
        self.data_list = []     # for DENSS
        self.result_list = []
        ret_list = []
        for n, (ax, C) in enumerate(zip(axes, C_list)):
            ax.cla()
            if n == 0:
                ax.set_title(self.center_title, fontsize=16)
            size = None if effective_sizes is None else effective_sizes[n]
            _, ylist, e11n = self.datasets.draw_exprapolated(self.no_base+n, ax, C, size, ipe=self.consider_ipe)
            self.result_list.append(e11n)
            ret_list.append(ylist)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            for p in [self.i_pos_values[n], self.pos_list[0]]:
                ax.plot([p, p], [ymin, ymax], ':', color='gray')
            self.draw_i_pos_text(n, ax, xpos=0.5)
        return ret_list

    def draw_distinction(self, axes, xy_list, ret_list, k_rows_list=None):
        point_set_list = []
        rg_set_list = []

        for n, (ax, xy, ylist) in enumerate(zip(axes, xy_list, ret_list)):
            x = self.mu_list[n] + self.j0

            e11n = self.result_list[n]
            ax.cla()

            if self.dtype == 'Xray':
                axt = self.twin_axes[n]
                axt.cla()
                axt.grid(False)
            if n == 0:
                ax.set_title("Components Distinction", fontsize=16)

            if k_rows_list:
                # print('self.point_set_list[%d]=' % n, self.point_set_list[n])
                for k, y_ in self.point_set_list[n]:
                    ax.plot(x[int(k)], y_, 'o', color='gray', alpha=0.3)

            if k_rows_list:
                k_rows = k_rows_list[n]
            else:
                k_rows = None

            assert len(self.pos_list) == 1

            pos_text = self.pos_format % self.pos_list[0]

            for p in self.pos_list:
                i = self.dataset.get_row_index(p)
                # picked_line = np.average(ylist[:,i-2:i+3], axis=1)    # this won't necessarily impove the line's stability
                picked_line = ylist[:,i]
                if k_rows is None:
                    y_for_line = picked_line
                else:
                    init_y_list = self.point_set_list[n,:,1]
                    y_for_line = self.expand_line(init_y_list, k_rows, picked_line)
                ax.plot(x, y_for_line, ':', label='Intensities' + pos_text)
                point_set = []
                for k, y_ in enumerate(picked_line):
                    if k_rows is None:
                        ax.plot(x[k], y_, 'o', color='C%d' % (k+2))
                        point_set.append((k, y_))
                    else:
                        k_ = k_rows[k]
                        y_ = np.ones(len(k_))*y_
                        ax.plot(x[k_], y_, marker='o', linewidth=3, color='C%d' % (k+2))

                if k_rows is None:
                    point_set_list.append(point_set)

            if self.dtype == 'Xray':
                rg_array = self.compute_rg_list(e11n)
                if k_rows is None:
                    rg_line = rg_array
                    rg_set_list.append(rg_array)
                else:
                    init_rg_array = self.rg_set_list[n]
                    rg_line = self.expand_line(init_rg_array, k_rows, rg_array)
                rg_line_y = rg_line[:,0]
                try:
                    axt.errorbar(x, rg_line_y, rg_line[:,1], fmt='o:', markersize=5, capsize=3, elinewidth=3, mfc='black', label="Rg's")
                except:
                    from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
                    etb = ExceptionTracebacker()
                    self.logger.warning("resorting to plain plot because errorbar plot failed due to %s", etb.last_lines())
                    axt.plot(x, rg_line_y, ':', marker='^', label="Rg's")
                quality = rg_line[:,2]
                for quality_limit in [0.5, 0.3]:
                    is_major = np.logical_and(np.isfinite(rg_line_y), quality > quality_limit)
                    major_rgs = rg_line_y[is_major]
                    major_average = np.average(major_rgs)
                    if len(major_rgs) > 2 and np.isfinite(major_average):
                        break
                if not np.isnan(major_average):
                    if len(major_rgs) > 1:
                        major_std = np.std(major_rgs)
                        print('major_rgs=', major_rgs, 'major_std=', major_std)
                        dy = max(10, major_std*2)
                    else:
                        dy = 10
                    axt.set_ylim(major_average-dy, major_average+dy)

                if False:
                    self.add_rg_values_from_v1_result(axt)

                # axt.legend(loc='center right')
                axt.legend(bbox_to_anchor=(1, 0.92), loc='upper right')

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            tx = xmin*0.8 + xmax * 0.2
            ty = ymin*0.1 + ymax * 0.9
            ax.text(tx, ty, pos_text, ha='center', va='center', fontsize=30, alpha=0.1)

            ax0 = self.axes[n,0]
            ax.set_xlim(ax0.get_xlim())
            ymin0, ymax0 = ax0.get_ylim()

            def transform_y(y):
                return (y - ymin0)*(ymax - ymin)/(ymax0 - ymin0) + ymin

            xp, yp = xy
            xp_ = xp + self.j0
            yp_ = transform_y(yp)
            zp_ = transform_y(0)
            poly_points = [(xp_[0], zp_)] + list(zip(xp_, yp_)) + [(xp_[-1], zp_)]
            poly = Polygon( poly_points, alpha=0.1, color='gray')
            ax.add_patch(poly)

            ax.legend()
            if self.rg_set_list is None:
                self.rg_set_list = rg_set_list

        return point_set_list

    def modify_major_xtick_positions(self, shrink_axis=False):
        # should be called after subplots_adjust
        for n, ax in enumerate(self.axes[:,2]):
            if shrink_axis:
                box = ax.get_position()
                dh = box.height * 0.05
                ax.set_position([box.x0, box.y0+dh, box.width, box.height-dh])

            ax0 = self.axes[n,0]
            ax.xaxis.set_major_locator(ticker.FixedLocator(ax0.get_xticks()))
            for k, label in enumerate(ax.xaxis.get_majorticklabels()):
                label.set_y(-0.05)
            ax.grid(False, which='major')

            def format_func(value, tick_number):
                """
                Customizing Ticks
                https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html
                """
                return num_to_char(tick_number)

            x = self.mu_list[n] + self.j0
            ax.xaxis.set_minor_locator(ticker.FixedLocator(x))
            ax.xaxis.set_minor_formatter(ticker.FuncFormatter(format_func))
            ax.grid(True, which='minor')

    def compute_rg_list(self, e11n):
        denss_fitted_rg = get_setting('denss_fitted_rg')
        rg_list = []
        q = self.dataset.vector
        P = e11n.P
        E = e11n.E
        num_curves = P.shape[1]//2 if self.consider_ipe else P.shape[1]
        for k in range(num_curves):
            y_ = P[:,k]
            e_ = E[:,k]

            if APPLY_DENSS_FIT and denss_fitted_rg:
                q_, y_, e_, _ = fit_data(q, y_, e_)
            else:
                q_ = q

            A_data = np.vstack( [q_, y_, e_] ).T
            autorg_kek = AutorgKekAdapter( A_data )
            result = autorg_kek.run()
            if result.Rg is None:
                rg_list.append((np.nan, np.nan, np.nan))
            else:
                rg_list.append((result.Rg, result.Rg_stdev, result.Quality))

        self.rg_array = np.array(rg_list)
        return self.rg_array

    def get_rg_array(self):
        return self.rg_array

    def expand_line(self, init_y_list, k_rows, picked_line):
        ret_y_list = init_y_list.copy()
        for k_row, y in zip(k_rows, picked_line):
            ret_y_list[k_row] = y
        return ret_y_list

    def build_panels(self, frame, button_frame):
        fig_frame = Tk.Frame(frame)
        fig_frame.pack(padx=10)
        space = Tk.Frame(frame, height=5)
        space.pack()
        matrix_frame = Tk.Frame(frame)
        matrix_frame.pack()

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3, 7.2))
        self.fig2 = fig
        self.axes2 = axes
        self.mpl_canvas2 = FigureCanvasTkAgg(fig, fig_frame)
        self.mpl_canvas2_widget = self.mpl_canvas2.get_tk_widget()
        self.mpl_canvas2_widget.pack(fill=Tk.BOTH, expand=1)

        label = Tk.Label(matrix_frame, text="Reduction Matrix", font=Tk.font.Font(size=20))
        label.pack()
        k = self.num_components
        self.gm = GroupingMatrix(matrix_frame, nrows=k, ncols=k, read_only=False)
        self.gm.pack()
        self.S_init = self.gm.get_matrix().copy()

        reset_button = Tk.Button(button_frame, text="Reset", width=10, command=self.reset)
        reset_button.pack(side=Tk.LEFT, padx=5, pady=5)
        compute_button = Tk.Button(button_frame, text="Compute", width=10, command=self.compute)
        compute_button.pack(side=Tk.LEFT, padx=5, pady=5)
        save_button = Tk.Button(button_frame, text="Save", width=10, command=self.save)
        save_button.pack(side=Tk.LEFT, padx=5, pady=5)

        if self.dtype == 'Xray':
            self.denss_menu = QmmDenssMenu(button_frame, self, state=Tk.DISABLED)
            self.denss_menu.pack(side=Tk.LEFT, padx=5)

    def copy_to_panel_fig(self, from_fig, from_axes):
        for to_ax, from_ax in zip(self.axes2, from_axes):
            to_ax.set_axis_off()
            to_title = from_ax.get_title().replace('Component', 'Individual')
            to_ax.set_title(to_title, y=0.98)
            draw_as_image(to_ax, from_fig, from_ax)
        self.fig2.subplots_adjust(top=0.95, bottom=0, left=0, right=1, wspace=0)
        self.mpl_canvas2.draw()

    def reset(self):
        self.gm.reset()
        self.compute()

    def compute(self):
        S = self.gm.get_matrix()
        assert S.shape[0] == S.shape[1]

        grouped_C_list, k_rows_list, effective_sizes = self.get_grouped_C_list(S)
        self.draw_all(self.xy_list, grouped_C_list, self.gy_list, k_rows_list, effective_sizes)
        self.modify_major_xtick_positions()
        self.mpl_canvas.draw()

        if self.dtype == 'Xray':
            self.denss_menu.config(state=Tk.NORMAL if self.grouped(S) else Tk.DISABLED)

    def get_grouped_C_list(self, S):
        new_C_list = []
        effective_sizes = []
        k_rows_list = []
        for C in self.C_list:
            C_rows = []
            k_rows = []
            for i in range(S.shape[0]):
                g_list = []
                k_vect = []
                for j, selected in enumerate(S[i,:]):
                    if selected:
                        g_list.append(C[j,:])
                        k_vect.append(j)
                if len(g_list) > 0:
                    c = np.sum(g_list, axis=0)
                    C_rows.append(c)
                    k_rows.append(np.array(k_vect))
            k_rows_list.append(k_rows)
            effective_sizes.append(len(C_rows))
            for j in range(S.shape[1]):
                if np.sum(S[:,j]) == 0:
                    # output-suppressed C_row,
                    # which will be exluded using effective_sizes
                    C_rows.append(C[j,:])
            new_C_list.append(np.array(C_rows))

        ret_k_rows_list = k_rows_list if self.grouped(S) else None
        return new_C_list, ret_k_rows_list, effective_sizes

    def save(self):
        from .QmmResultSaver import QmmResultSaverDialog
        dialog = QmmResultSaverDialog(self.dialog, self.result_list[0])
        dialog.show()

    def save_the_figure(self, folder, analysis_name):
        filename = analysis_name.replace( 'analysis', 'figure' )
        path = os.path.join( folder, filename )
        self.fig.savefig( path )
        return path

    def show_3d_figure(self):
        from .Qmm3dFigure import Qmm3dFigure
        fig = Qmm3dFigure(self.dialog, self.dataset, self)
        fig.show()

    def show_various_denoise_figures(self):
        from .DenoiseFigures import DenoiseFigures
        fig = DenoiseFigures(self.dialog, self.dataset, self)
        fig.show()

    def add_rg_values_from_v1_result(self, ax):
        from DataUtils import get_pytools_folder
        from molass_legacy.Reports.ReportUtils import just_load_guinier_result
        from molass_legacy.KekLib.NumpyUtils import np_loadtxt
        report_folder = get_pytools_folder() + '/reports/20180602'
        # report_folder = get_pytools_folder() + '/reports/sample_data'
        guinier_result_csv = report_folder + '/.guinier_result/--serial_result.csv'
        array = np.array(just_load_guinier_result(guinier_result_csv))
        self.logger.info('array.shape=%s', str(array.shape))
        xr = self.dataset
        j0 = xr.j0
        # j0= 604 xr.data.shape= (1074, 101) array.shape= (311, 33)
        print('j0=', j0, 'xr.data.shape=', xr.data.shape, 'array.shape=', array.shape)
        size = xr.data.shape[1]
        x = j0 + np.arange(size)
        y = np.array([float(rg) for rg in array[:,11]])
        ax.plot(x, y[j0:j0+size], color='green', label="Rg's (V1 Guinier)")

        extrapolated_csv = report_folder + '/extrapolated.csv'
        xy, _ = np_loadtxt(extrapolated_csv)
        ax.plot(xy[:,0], xy[:,1], color='orange', label="Rg's (V1 Extrapolated)")
