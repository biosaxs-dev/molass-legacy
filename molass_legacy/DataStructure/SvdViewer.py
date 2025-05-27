# coding: utf-8
"""
    SvdViewer.py

    Copyright (c) 2018-2022, SAXS Team, KEK-PF
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkUtils import is_low_resolution
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy._MOLASS.SerialSettings import get_setting
from ScatteringBaseCorrector import ScatteringBaseCorrector
from ScatteringViewUtils import compute_baselines, draw_3d_scattering
from molass_legacy.UV.AbsorbanceViewUtils import draw_3d

DEBUG = False

TITLE_FONTSIZE = 16
LABLE_FONTSIZE = 16
TICKLABEL_SIZE = 7
MAX_NUM_SVALUES = 10

ADD_TOOLBAR = False
if ADD_TOOLBAR:
    from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar


class SvdViewer(Dialog):
    def __init__(self, parent, title, sd, mapper):
        self.parent = parent
        self.title_ = title
        self.sd = sd
        self.mapper = mapper

    def show(self):
        self.parent.config(cursor='wait')
        Dialog.__init__(self, self.parent, self.title_)

    def body(self, body_frame):  # overrides parent class method
        tk_set_icon_portable(self)

        self.vframe = vframe = Tk.Frame(body_frame)
        vframe.pack()

        absorbance = self.sd.absorbance
        data = absorbance.get_corrected_data()
        ecurve = self.mapper.a_curve.y
        curves = [data[:, int(info[1] + 0.5)] for info in self.mapper.a_curve.peak_info]
        self.uv_frame = SvdFrame(vframe, self.sd.lvector, data, [ecurve], curves, "UV Absorbance",
                                 absorbance=absorbance, toggle_cb=self.show_the_other, toggle_text="Show Xray",
                                 dialog=self)
        self.uv_frame.grid(row=0, column=0)
        self.uv_frame.grid_forget()

        data = self.sd.intensity_array[:, :, 1].T
        ecurve = self.mapper.x_curve.y
        curves = [data[:, int(info[1] + 0.5)] for info in self.mapper.x_curve.peak_info]
        self.xray_frame = SvdFrame(vframe, self.sd.qvector, data, [ecurve], curves, "Xray Scattering",
                                   serial_data=self.sd, mapper=self.mapper,
                                   toggle_cb=self.show_the_other, toggle_text="Show UV",
                                   dialog=self)
        self.xray_frame.grid(row=0, column=0)
        self.xray_frame.grid_forget()

        self.bframe = bframe = Tk.Frame(body_frame)
        bframe.pack()

        self.current_data = 1
        # self.toggle_btn = Tk.Button( bframe, text="", command=self.show_the_other )
        # self.toggle_btn.grid( row=0, column=0 )
        self.show_the_other()
        self.parent.config(cursor='')

    def show_the_other(self):
        self.current_data = 1 - self.current_data
        if self.current_data == 0:
            self.uv_frame.grid_forget()
            self.xray_frame.grid(row=0, column=0)
            self.xray_frame.draw()
            # self.toggle_btn.config(text="Show UV")
        else:
            self.xray_frame.grid_forget()
            self.uv_frame.grid(row=0, column=0)
            self.uv_frame.draw()
            # self.toggle_btn.config(text="Show Xray")


class SvdFrame(Tk.Frame):
    def __init__(self, parent, vector, array, ecurves, ucurves, data_name,
                 serial_data=None, mapper=None, absorbance=None,
                 toggle_cb=None, toggle_text=None, dialog=None):
        self.parent = parent
        self.vector = vector
        self.array = array
        self.ecurves = ecurves
        self.ucurves = ucurves
        self.data_name = data_name
        self.serial_data = serial_data
        self.mapper = mapper
        self.absorbance = absorbance
        self.toggle_cb = toggle_cb
        self.toggle_text = toggle_text
        self.dialog = dialog

        U, s, VT = np.linalg.svd(array)
        self.U = U
        self.s = s
        self.V = VT.T
        Tk.Frame.__init__(self, self.parent)
        self.build_body()

    def build_body(self):  # overrides parent class method
        tk_set_icon_portable(self)

        body_frame = Tk.Frame(self)
        body_frame.pack()

        cframe = Tk.Frame(body_frame)
        cframe.pack()
        ctlframe = Tk.Frame(body_frame)
        ctlframe.pack(fill=Tk.X)

        if is_low_resolution():
            figsize = (17, 7)
        else:
            figsize = (21, 10)

        self.fig = fig = plt.figure(figsize=figsize)

        in_folder = get_setting('in_folder')
        fig.suptitle(self.data_name + ' from ' + in_folder, fontsize=TITLE_FONTSIZE)

        gs = gridspec.GridSpec(2, 7)
        ax0 = self.fig.add_subplot(gs[:, 0:3], projection='3d')
        ax1 = self.fig.add_subplot(gs[0, 3:5])
        ax2 = self.fig.add_subplot(gs[1, 3:5])
        ax3 = self.fig.add_subplot(gs[:, 5:7])

        self.axes = [ax0, ax1, ax2, ax3]
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas.draw()
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        if ADD_TOOLBAR:
            self.toolbar = NavigationToolbar(self.mpl_canvas, cframe)
            self.toolbar.update()

        ctlframe.columnconfigure(0, weight=1)
        ctlframe.columnconfigure(1, weight=1)

        toggle_frame = Tk.Frame(ctlframe)
        toggle_frame.grid(row=0, column=0)

        toggble_btn = Tk.Button(toggle_frame, text=self.toggle_text, command=self.toggle_cb)
        toggble_btn.pack()

        control_panel = Tk.Frame(ctlframe)
        control_panel.grid(row=0, column=1, sticky=Tk.E, padx=40)

        spinbox_frame = Tk.Frame(control_panel)
        spinbox_frame.grid(row=0, column=0, padx=10)

        spinbox_label = Tk.Label(spinbox_frame, text="Number of components to show: ")
        spinbox_label.grid(row=0, column=0)

        self.num_components = Tk.IntVar()
        self.num_components.set(2)

        self.spinbox = Tk.Spinbox(spinbox_frame, textvariable=self.num_components,
                                  from_=1, to=MAX_NUM_SVALUES, increment=1,
                                  justify=Tk.CENTER, width=6)
        self.spinbox.grid(row=0, column=1)

        self.uv_sign = 1

        self.reverse_button = Tk.Button(control_panel, text='Reverse', command=self.reverse)
        self.reverse_button.grid(row=0, column=1, padx=10)

        space = Tk.Frame(control_panel, width=400)
        space.grid(row=0, column=2)

        self.save_button = Tk.Button(control_panel, text='Results Saver', command=self.show_save_dialog)
        self.save_button.grid(row=0, column=3, padx=10)

        self.draw3d()
        self.fig.subplots_adjust(top=0.9, bottom=0.06, left=0.02, right=0.98, wspace=0.32, hspace=0.2)

        self.num_components.trace('w', self.num_components_tracer)

    def num_components_tracer(self, *args):
        self.draw()

    def reverse(self):
        if self.reverse_button.config('relief')[-1] == 'sunken':
            self.reverse_button.config(relief="raised")
            self.uv_sign = 1
        else:
            self.reverse_button.config(relief="sunken")
            self.uv_sign = -1

        self.draw()

    def draw3d(self):
        ax0 = self.axes[0]
        title = self.data_name + ' in 3D'

        if self.absorbance is None:
            serial_data = self.serial_data
            assert serial_data is not None
            assert self.mapper is not None

            data = serial_data.intensity_array
            qvector = serial_data.qvector
            slice_ = serial_data.xray_slice
            index = (slice_.start + slice_.stop) // 2
            xray_curve_y = serial_data.xray_curve.y
            opt_params = self.mapper.opt_params

            data_copy = copy.deepcopy(data)
            corrector = ScatteringBaseCorrector(
                serial_data.jvector,
                serial_data.qvector,
                data_copy,
                curve=serial_data.xray_curve,
                affine_info=self.mapper.get_affine_info(),
                inty_curve_y=xray_curve_y,
                baseline_opt=opt_params['xray_baseline_opt'],
                baseline_type=opt_params['xray_baseline_type'],
                need_adjustment=opt_params['xray_baseline_adjust'] == 1,
                parent=self)

            vsa_base_list, all_base_list = compute_baselines(qvector, index, corrector)

            draw_3d_scattering(ax0, data, qvector, index, xray_curve_y,
                               title, vsa_base_list, all_base_list
                               )
        else:
            absorbance = self.absorbance
            # TODO: consider unifying these params with those of AbsorbanceViewer
            wvlen_lower = 245
            wvlen_upper = 450
            i_slice = slice(0, len(absorbance.wl_vector))
            draw_3d(ax0, absorbance, wvlen_lower, wvlen_upper, i_slice,
                    title=title, title_fontsize=TITLE_FONTSIZE, low_percentile=False)

    def draw(self):
        num_components = self.num_components.get()
        self.draw_U(num_components)
        self.draw_V(num_components)
        self.draw_s(num_components)

        for ax in self.axes:
            if ax is None:
                continue
            ax.tick_params(labelsize=LABLE_FONTSIZE)

        self.mpl_canvas.draw()

    def draw_U(self, num_components):
        ax1 = self.axes[1]
        ax1.cla()
        ax1.set_title('Major U Spectra', fontsize=TITLE_FONTSIZE)
        # ax1.tick_params(axis='y', labelsize=TICKLABEL_SIZE)         # does not work

        for i in range(num_components):
            ax1.plot(self.vector, self.U[:, i] * self.uv_sign, label='U%d' % i, linewidth=5)

        ax1.legend(fontsize=LABLE_FONTSIZE)

    def draw_V(self, num_components):
        ax2 = self.axes[2]
        ax2.cla()
        ax2.set_title('Major V Spectra', fontsize=TITLE_FONTSIZE)
        # ax2.tick_params(axis='y', labelsize=TICKLABEL_SIZE)         # does not work

        for i in range(num_components):
            ax2.plot(self.V[:, i] * self.uv_sign, 'o', label='V%d' % i)

        ax2.legend(fontsize=LABLE_FONTSIZE)

    def draw_s(self, num_components):
        ax = self.axes[3]
        ax.cla()
        ax.set_title('Major Singular Values', fontsize=TITLE_FONTSIZE)

        num_see_values = self.num_components.get()
        num_show_values = max(6, min(MAX_NUM_SVALUES, num_see_values + 2))

        ax.plot(self.s[0:num_show_values], ':', color='gray')

        for i in range(num_show_values):
            ax.plot(i, self.s[i], 'o', markersize=12)

    def show_save_dialog(self):
        from SvdDataSaver import SvdDataSaverDialog
        dialog = SvdDataSaverDialog(self.dialog, self.data_name,
                        svd_results=[self.vector, self.U, self.s, self.V],
                        num_components=self.num_components.get(),
                        location='lower right')
        dialog.show()
