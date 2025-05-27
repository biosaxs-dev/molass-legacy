"""
    SecTools.CorMap.CormapMakerDialog.py

    Copyright (c) 2021-2023, SAXS Team, KEK-PF
"""
import os
import logging
from bisect import bisect_right
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import SpanSelector
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
from SerialDataUtils import load_intensity_files
from SaferSpinbox import SaferSpinbox
from molass_legacy.KekLib.TkSupplements import BlinkingFrame
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from molass_legacy.Elution.CurveUtils import simple_plot
from DataUtils import get_in_folder
from molass_legacy.KekLib.TkUtils import split_geometry
from .CormapVersion import get_version_string
from .AngularUnit import angstrom_qv

ALLOW_SPAN_CHANGE = False
DATA_VIESW_NAMES = ["2D View", "3D View"]
CORREL_TYPE_NAMES = ["numpy corrcoef", "ATSAS DATCMP"]
TRANSPOSE_NAMES = ["Frames", "Angular"]

class CormapMakerDialog(Dialog):
    def __init__(self, parent, demo=False, elution=False, atsas=None):
        self.logger = logging.getLogger(__name__)
        self.demo = demo
        self.elution = elution
        if atsas is None:
            from Env.EnvInfo import get_global_env_info
            env_info = get_global_env_info()
            atsas = env_info.atsas_is_available
        self.threed = False
        self.atsas = atsas
        self.rect = None
        self.colorbar = None
        self.cbar_ax = None
        self.cormap_drawn = False
        self.showing_transposed = True
        self.showing_datcmp = False
        self.datafiles = None
        self.popup_menu1 = None
        self.popup_menu3 = None
        self.datcmp_data = None
        self.datcmp_data_input = None
        Dialog.__init__(self, parent, get_version_string(), visible=False)

    def body(self, body_frame):
        frame = Tk.Frame(body_frame)
        frame.pack(padx=20, pady=10)

        input_frame = Tk.Frame(frame)
        input_frame.pack(pady=20)

        label = Tk.Label(input_frame, text="Input Folder: ")
        label.pack(side=Tk.LEFT)

        self.in_folder = Tk.StringVar()
        self.fe = FolderEntry(input_frame, textvariable=self.in_folder, width=80,
                                on_entry_cb=self.on_in_folder_entry)
        self.fe.pack(side=Tk.LEFT)

        cframe = Tk.Frame(frame)
        cframe.pack(padx=20)

        self.fig = fig = plt.figure(figsize=(18,6))
        self.gs = gs = GridSpec(2,12)
        ax1 = fig.add_subplot(gs[:,0:4])
        ax2 = fig.add_subplot(gs[0,4:8])
        ax3 = fig.add_subplot(gs[1,4:8])
        ax4 = fig.add_subplot(gs[:,8:12])
        self.axes = [ax1, ax2, ax3, ax4]    # this must be a list to be able to repalce ax1
        self.reset_cbar_ax()

        self.draw_titles()

        # self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.85, left=0.05, right=0.94, bottom=0.1, wspace=1.8)

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        tframe = Tk.Frame(frame)
        tframe.pack(padx=0, fill=Tk.X)      # padx=0: suppress changes from toolbar cursor display
        tframe_left = Tk.Frame(tframe)
        tframe_left.pack(side=Tk.LEFT, fill=Tk.X)
        tframe_right = Tk.Frame(tframe)
        tframe_right.pack(side=Tk.RIGHT)

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe_left)
        self.toolbar.update()

        self.build_control_bar(tframe_right)

        self.mpl_canvas.mpl_connect('button_press_event', self.on_figure_click)

    def reset_cbar_ax(self):
        if self.cbar_ax is not None:
            self.cbar_ax.remove()
        self.cbar_ax = self.fig.add_axes([0.89, 0.0, 0.06, 1.0])
        self.cbar_ax.set_axis_off()

    def build_control_bar(self, frame):
        span_state = Tk.NORMAL if ALLOW_SPAN_CHANGE else Tk.DISABLED
        self.span_selector = Tk.BooleanVar()
        w = Tk.Checkbutton(frame, text="Span Selector", variable=self.span_selector, state=span_state)
        w.pack(side=Tk.LEFT, padx=10)
        self.span_selector_cb = w
        self.span_selector.trace("w", self.span_selector_tracer)

        self.whole_data_view = Tk.StringVar()
        w = ttk.Combobox(frame, textvariable=self.whole_data_view, values=DATA_VIESW_NAMES, width=10)
        w.pack(side=Tk.LEFT, padx=10)
        w.current(0)
        self.whole_data_view_cbox = w
        self.whole_data_view.trace("w", self.whole_data_view_tracer)

        select_frame = Tk.Frame(frame)
        select_frame.pack(side=Tk.LEFT)

        self.build_select_frame(select_frame, span_state)

        space = Tk.Frame(frame, width=10)
        space.pack(side=Tk.LEFT)

        self.correl_type = Tk.StringVar()
        state = Tk.NORMAL if self.demo or self.atsas else Tk.DISABLED
        w = ttk.Combobox(frame, textvariable=self.correl_type, values=CORREL_TYPE_NAMES, width=14, state=state)
        w.pack(side=Tk.LEFT, padx=10)
        w.current(int(self.atsas))
        self.correl_type_cbox = w
        self.correl_type.trace("w", self.correl_type_tracer)

        self.transpose = Tk.StringVar()
        w = ttk.Combobox(frame, textvariable=self.transpose, values=TRANSPOSE_NAMES, width=10)
        w.pack(side=Tk.LEFT, padx=10)
        w.current(0)
        self.transpose.trace("w", self.transpose_tracer)

        button_frame = Tk.Frame(frame)
        button_frame.pack(side=Tk.LEFT)

        self.cormap_btn_blink = BlinkingFrame(button_frame)
        self.cormap_btn_blink.pack(side=Tk.LEFT, padx=10, pady=5)
        w = Tk.Button(self.cormap_btn_blink, text="Draw Cormap", width=12, command=self.draw_cormap, state=Tk.DISABLED)
        w.pack()
        self.cormap_btn_blink.objects = [w]
        self.cormap_btn = w

        w = Tk.Button(button_frame, text="Save", width=6, command=self.save, state=Tk.DISABLED)
        w.pack(side=Tk.LEFT, padx=10, pady=5)
        self.save_btn = w

    def build_select_frame(self, frame, span_state):
        self.select_vars = []
        self.sb_widgets = []
        range_name = "Elution-Range:" if self.elution else "Frame-Range"
        label = Tk.Label(frame, text=range_name)
        label.pack(side=Tk.LEFT)
        for k, sb_namel in enumerate(["start", "end"]):
            label = Tk.Label(frame, text=sb_namel)
            label.pack(side=Tk.LEFT, padx=5)

            var = Tk.IntVar()
            self.select_vars.append(var)
            sb = SaferSpinbox(frame, textvariable=var,
                        from_=0, to=200, increment=1, 
                        justify=Tk.CENTER, width=6, state=span_state)
            sb.pack(side=Tk.LEFT, padx=5)
            sb.set_tracer(self.redraw_elution_rect)
            var.trace("w", sb.tracer)
            self.sb_widgets.append(sb)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=50, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def show(self):
        if self.demo:
            self.after(500, self.perpare_temp_demo)
        self._show()

    def perpare_temp_demo(self):
        from molass_legacy._MOLASS.SerialSettings import get_setting
        in_folder = get_setting("in_folder")
        self.in_folder.set(in_folder)
        self.on_in_folder_entry()

    def on_in_folder_entry(self):
        self.update()
        in_folder = self.in_folder.get()
        try:
            data_array, datafiles = load_intensity_files(in_folder)
            self.datcmp_data_input = None
            self.datafiles = datafiles
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(None, "on_in_folder_entry: ")
            return

        in_folder = get_in_folder(in_folder.replace("\\", "/"))
        self.fig.suptitle("Correlation Analysis on %s" % in_folder, fontsize=20)
        for ax in self.axes:
            ax.cla()

        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        self.draw_titles()
        self.draw_ranges(data_array)
        self.activate_buttons()

    def activate_buttons(self):
        self.cormap_btn.config(state=Tk.NORMAL)
        if self.atsas:
            self.correl_type_cbox.config(state=Tk.NORMAL)

    def draw_titles(self):
        ax1, ax2, ax3, ax4 = self.axes
        ax2.set_title("Scattering Range", fontsize=16)
        ax4.set_title("Correlation Map", fontsize=16)

    def draw_ranges(self, data_array):
        self.qv = angstrom_qv(data_array[0,:,0])
        self.D = data_array[:,:,1].T
        self.E = data_array[:,:,2].T

        ax1, ax2, ax3, ax4 = self.axes
        self.draw_sequence_range(ax1)
        self.draw_scattering_range(ax2, ax3)
        self.update_select_vars()

        self.mpl_canvas.draw()

    def draw_sequence_range(self, ax, update_vars=True):
        title1 = "Elution Range" if self.elution else "Frame Sequence Range"
        ax.set_title(title1, fontsize=16)

        seq_name = "seqno." if self.elution else "Frame No."
        ax.set_xlabel(seq_name)
        ax.set_ylabel("Intensity")

        picking_q = 0.02
        i = bisect_right(self.qv, picking_q)
        ey = self.D[i,:]
        ex = np.arange(len(ey))
        N = 11
        sey = np.convolve(ey, np.ones(N)/N, mode='same')

        if self.elution:
            ecurve = ElutionCurve(sey)
            self.ecurve = ecurve
            simple_plot(ax, ecurve, legend=False)
            paired_ranges = ecurve.get_default_paired_ranges()
            ranges = paired_ranges[0].get_fromto_list()
            f, t = ranges[0][0], ranges[-1][1]
            info = ecurve.get_primary_peak_info()
            j_pair = (info[0], info[2])
        else:
            ax.plot(ex, sey, label="intensities at q=%g" % picking_q)
            f, t = 0, len(ey)-1
            j_pair = tuple(int(round(v)) for  v in np.linspace(0, self.D.shape[1], 4)[[1,2]])

        self.seq_slice = slice(f, t+1)
        self.j_pair = j_pair
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)

        self.rect = Rectangle(
                (f, ymin),      # (x,y)
                t - f,          # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )
        ax.add_patch(self.rect)

        self.vlines = []
        label = None
        for j in j_pair:
            line,  = ax.plot([j, j], [ymin, ymax], ':', color="yellow", label=label)
            self.vlines.append(line)
            label = "selected frames"

        ax.legend()

        if update_vars:
            jmax = len(ex) - 1
            for k, j in [(0,f), (1,t)]:
                self.select_vars[k].set(j)
                sb = self.sb_widgets[k]
                sb.config(to=jmax)

    def redraw_elution_rect(self, *args):
        if self.rect is None:
            return

        self.cormap_drawn = False
        self.cormap_btn_blink.start()
        ax = self.axes[0]
        ymin, ymax = ax.get_ylim()
        f, t = [w.get() for w in self.select_vars[0:2]]
        self.rect.set_xy((f, ymin))
        self.rect.set_width(t - f)
        self.seq_slice = slice(f, t+1)
        self.mpl_canvas.draw()

    def on_span_select(self, xmin, xmax):
        if self.rect is None:
            return

        f = max(0, int(round(xmin)))
        t = min(self.D.shape[1]-1, int(round(xmax)))
        for k, j in enumerate([f, t]):
            self.select_vars[k].set(j)

    def cormap_available(self):
        correl_type = self.correl_type.get()
        transpose = self.transpose.get()
        ret_ok = correl_type == CORREL_TYPE_NAMES[0] or transpose == TRANSPOSE_NAMES[0]
        if not ret_ok:
            self.show_not_implemented_error(correl_type, TRANSPOSE_NAMES[1])
        return ret_ok

    def show_not_implemented_error(self, correl_type, transpose):
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        MessageBox.showerror("Not Implemented Error",
            '"%s" is not available with "%s"' % (transpose, correl_type),
            parent=self)

    def correl_type_tracer(self, *args):
        if not self.cormap_available():
            return

        self.cormap_drawn = False
        self.cormap_btn_blink.start()

    def transpose_tracer(self, *args):
        if self.rect is None:
            return

        if not self.cormap_available():
            return

        self.cormap_drawn = False
        self.cormap_btn_blink.start()

    def draw_scattering_range(self, ax2, ax3):
        ax2.set_yscale("log")
        ax2.set_ylabel("Intensity")

        # ax2.axes.get_xaxis().set_ticks([])
        ax2.set_xticklabels([])
        ax3.set_yscale("log")
        ax3.set_ylabel("Stdev")
        ax3.set_xlabel(r"$q(\AA^{-1})$")

        for j in self.j_pair:
            sy = self.D[:,j]
            ax2.plot(self.qv, sy, label="curve at frame %d" % j)
            ey = self.E[:,j]
            ax3.plot(self.qv, ey, label="curve at frame %d" % j)

        ax2.legend()
        ax3.legend()

    def update_select_vars(self):
        imax = self.D.shape[1]-1
        for k, i in [(0,0), (1,imax)]:
            self.select_vars[k].set(i)
            sb = self.sb_widgets[k]
            sb.config(to=imax)

    def draw_cormap(self):
        from .CormapMaker import CormapMaker

        transpose = self.transpose.get() == TRANSPOSE_NAMES[0]
        self.showing_transposed = transpose
        correl_type = self.correl_type.get()

        if correl_type == CORREL_TYPE_NAMES[1]:
            if not transpose:
                self.show_not_implemented_error(correl_type, TRANSPOSE_NAMES[1])
                return

        self.cormap_btn_blink.stop()

        if correl_type == CORREL_TYPE_NAMES[0]:
            M = self.D[:,self.seq_slice]
            datcmp_data = None
            from_datcmp_str = ""
            self.showing_datcmp = False
        else:
            M, datcmp_data = self.get_data_from_datcmp()
            from_datcmp_str = " (DATCMP)"
            self.showing_datcmp = True
        self.datcmp_data = datcmp_data
        seqno = np.arange(self.seq_slice.start, self.seq_slice.stop)
        seqname = "Eno." if self.elution else "Frame No."
        cm = CormapMaker(M, self.qv, seqno, seqname, transpose=transpose, from_datcmp=datcmp_data is not None)
        ax = self.axes[-1]
        ax.cla()
        self.reset_cbar_ax()
        title = "Correlation Map%s" % from_datcmp_str
        ax.set_title(title, fontsize=16)
        if self.colorbar is not None:
            self.colorbar.remove()
        self.colorbar = cm.draw(ax, self.cbar_ax)
        if self.showing_transposed:
            self.draw_selected(ax)
        self.mpl_canvas.draw()
        self.cormap_drawn = True

    def draw_selected(self, ax):
        self.selected_points = []
        for pair in [self.j_pair, reversed(self.j_pair)]:
            point, = ax.plot(*pair, 'o', color="yellow")
            self.selected_points.append(point)

    def on_figure_click(self, event):
        if not self.showing_transposed:
            return

        if event.button == 1:
            self.draw_pair_position(event)
        elif event.button == 3:
            if event.inaxes == self.axes[0]:
                self.show_popup_menu1(event)
            elif event.inaxes == self.axes[3] and self.showing_datcmp:
                self.show_popup_menu3(event)
            return

    def draw_pair_position(self, event):
        ax4 = self.axes[3]
        if not self.cormap_drawn or event.inaxes != ax4:
            return

        min_i = self.seq_slice.start
        max_i = self.seq_slice.stop - 1
        def get_index(x):
            return max(min_i, min(max_i, int(round(x))))

        self.j_pair = tuple(sorted([get_index(v) for v in [event.xdata, event.ydata]]))

        ax1, ax2, ax3 = self.axes[0:3]
        if self.threed:
            for k, line in zip(self.j_pair, self.pairframe_lines):
                ydata = np.ones(len(self.qv)) * k
                line.set_xdata(self.qv)
                line.set_ydata(ydata)
                # https://stackoverflow.com/questions/46685326/how-to-set-zdata-for-a-line3d-object-in-python-matplotlib
                line.set_3d_properties(self.D[:,k])
        else:
            ymin, ymax = ax1.get_ylim()
            for k, line in zip(self.j_pair, self.vlines):
                line.set_data([k, k], [ymin, ymax])

        for ax in [ax2, ax3]:
            ax.cla()

        ax2.set_title("Scattering Range", fontsize=16)
        self.draw_scattering_range(ax2, ax3)

        for k, pair in enumerate([self.j_pair, reversed(self.j_pair)]):
            self.selected_points[k].set_data(*pair)

        self.mpl_canvas.draw()

    def show_popup_menu1(self, event):
        self.create_popup_menu1(event)
        canvas = self.mpl_canvas_widget
        cx = canvas.winfo_rootx()
        cy = canvas.winfo_rooty()
        w, h, x, y = split_geometry(canvas.winfo_geometry())
        self.popup_menu1.post(cx + int(event.x), cy + h - int(event.y))

    def create_popup_menu1(self, event):
        if self.popup_menu1 is None:
            self.popup_menu1 = Tk.Menu(self, tearoff=0 )
            if ALLOW_SPAN_CHANGE:
                self.popup_menu1.add_checkbutton(label="Span Selector", variable=self.span_selector)
            self.popup_menu1.add_command(label='Chage View (2D ⇔ 3D)', command=self.change_view_2d3d)

    def show_popup_menu3(self, event):
        self.create_popup_menu3(event)
        canvas = self.mpl_canvas_widget
        cx = canvas.winfo_rootx()
        cy = canvas.winfo_rooty()
        w, h, x, y = split_geometry(canvas.winfo_geometry())
        self.popup_menu3.post(cx + int(event.x), cy + h - int(event.y))

    def create_popup_menu3(self, event):
        if self.popup_menu3 is None:
            self.popup_menu3 = Tk.Menu(self, tearoff=0 )
            self.popup_menu3.add_command(label='Show Pairwise Map', command=self.show_pairwise_map)

    def get_data_from_datcmp(self):
        from molass_legacy.ATSAS.DatCmp import run_datcmp
        if self.datcmp_data_input is None:
            files = self.datafiles[self.seq_slice]
            self.datcmp_data_input = run_datcmp(files=files, return_dict=True)
        return self.datcmp_data_input

    def save(self):
        pass

    def whole_data_view_tracer(self, *args):
        ax1 = self.axes[0]
        if False:
            ax1.remove()    # this causes the following error
            """
            Traceback (most recent call last):
              File "C:\Program Files\Python39\lib\site-packages\matplotlib\cbook\__init__.py", line 270, in process
                func(*args, **kwargs)
              File "C:\Program Files\Python39\lib\site-packages\mpl_toolkits\mplot3d\axes3d.py", line 1223, in _button_release
                toolbar = getattr(self.figure.canvas, "toolbar")
            AttributeError: 'NoneType' object has no attribute 'canvas'
            """
        else:
            # work-around to avoid the above mentioned error, which can be causing memory leaks
            ax1.cla()
            ax1.set_axis_off()

        self.threed = self.whole_data_view.get().find("3") >= 0
        if self.threed:
            dim = "3d"
        else:
            dim = None
        ax1 = self.fig.add_subplot(self.gs[:,0:4], projection=dim)
        self.axes[0] = ax1
        self.mpl_canvas.draw()

        if self.threed:
            self.draw_data_in_3d(ax1)
            selector_state = Tk.DISABLED
        else:
            self.draw_sequence_range(ax1, update_vars=False)
            selector_state = Tk.NORMAL

        self.span_selector_cb.config(state=selector_state)
        self.mpl_canvas.draw()

    def draw_data_in_3d(self, ax):
        from MatrixData import simple_plot_3d
        ax.set_xlabel("$q(\AA^{-1})$")
        ax.set_ylabel("Frame No.")
        ax.set_zlabel("Intensity")

        ax.set_title("3D View", fontsize=16)
        simple_plot_3d(ax, self.D, x=self.qv)
        self.pairframe_lines = []
        x = self.qv
        for j in self.j_pair:
            y = np.ones(len(x)) * j
            z = self.D[:,j]
            line, = ax.plot(x, y, z, label="Frame No. %d" % j)
            self.pairframe_lines.append(line)
        ax.legend()

    def span_selector_tracer(self, *args):
        span_selector = self.span_selector.get()
        if span_selector:
            self.span = SpanSelector(self.axes[0], self.on_span_select, 'horizontal', useblit=True,
                        props=dict(alpha=0.5))
        else:
            self.span = None

    def show_pairwise_map(self):
        from .PairwiseMap import PairwiseMap
        M = self.D[:,self.seq_slice]
        dialog = PairwiseMap(self, M, self.j_pair, self.qv, self.datcmp_data)
        dialog.show()

    def change_view_2d3d(self):
        view_name = self.whole_data_view.get()
        current = 0 if view_name == DATA_VIESW_NAMES[0] else 1
        next = 1 - current
        self.whole_data_view.set(DATA_VIESW_NAMES[next])
        self.whole_data_view_cbox.current(next)
