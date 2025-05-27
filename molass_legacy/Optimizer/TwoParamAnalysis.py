"""
    Optimizer.TwoParamAnalysis.py

    Copyright (c) 2021-2022, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ToolTip
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy._MOLASS.SerialSettings import get_setting
from ttkwidgets import TickScale
from DataUtils import get_in_folder

INIT_ZOOM_ARE_RATIO = 0.5
MIN_ZOOM_ARE_RATIO = 0.1
MAX_ZOOM_ARE_RATIO = 1.0
ZOOM_AREA_RESOLUTION = 0.01

class TwoParamAnalysis(Dialog):
    def __init__(self, parent, dsets, optimizer, demo_info, curr_index, i=5, j=9):
        self.parent = parent
        self.in_folder = get_in_folder()
        self.dsets = dsets
        self.optimizer = optimizer
        self.xarray = demo_info[1]
        self.params = self.xarray[curr_index]
        (i, j), spp = self.select_default_params()
        self.i = i
        self.j = j
        self.curr_index = curr_index
        self.set_trace_xy()
        self.spp = spp
        self.temp_params = self.params.copy()
        self.bounds = optimizer.real_bounds
        self.two_param_vector_func = np.vectorize(self.two_param_func)
        self.zoom_area_ratio = None     # this will be replaced by TickScale
        Dialog.__init__(self, parent, "Two Parameter Analysis", visible=False)

    def set_trace_xy(self):
        self.trace_xy = self.xarray[0:self.curr_index+1,[self.i,self.j]].T

    def select_default_params(self):
        xr_params, xr_baseparams, rgs, mapping, uv_params, uv_baseparams, mappable_range, seccol_params = self.optimizer.split_params_simple(self.params)[0:8]

        def normalize(p):
            a = np.average(p)
            s = np.std(p)
            return (p - a)/s

        if len(xr_params.shape) == 2:
            xr_params_ = xr_params[:,0]
        else:
            xr_params_ = xr_params
        scales = normalize(xr_params_) + normalize(uv_params[:])
        k = len(scales)-2
        pp = np.argpartition(scales, k)
        # print("scales=", scales, "k=", k, "pp=", pp)
        spp = sorted(pp[k:])
        if len(xr_params.shape) == 2:
            # major two peak positions
            i, j = np.arange(1, len(xr_params.flatten()), 4)[spp]
        else:
            # major two rgs
            i, j = len(xr_params) + len(xr_baseparams) + np.arange(0, len(rgs))[spp]
        return (i, j), spp

    def two_param_func(self, x, y):
        self.temp_params[self.i] = x
        self.temp_params[self.j] = y
        return self.optimizer.objective_func(self.temp_params)

    def show(self):
        self._show()

    def cancel(self):
        plt.close(self.fig)
        Dialog.cancel(self)

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tframe = Tk.Frame(body_frame)
        tframe.pack(fill=Tk.X, padx=20)
        tframe_left = Tk.Frame(tframe)
        tframe_left.pack(side=Tk.LEFT)
        tframe_right = Tk.Frame(tframe)
        tframe_right.pack(side=Tk.RIGHT)

        self.fig = fig = plt.figure(figsize=(21, 8))
        self.fig_suptitle = fig.suptitle(self.get_suptitle_text(), fontsize=20)
        ax1 = fig.add_subplot(131, projection="3d")
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        self.axes = [ax1, ax2, ax3]
        self.draw_function()
        fig.tight_layout()
        fig.subplots_adjust(top=0.88, left=0.01, wspace=0.18, right=0.99)
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe_left)
        self.toolbar.update()
        self.build_control_frame(tframe_right)

    def build_control_frame(self, frame):
        inspect_btn = Tk.Button(frame, text="Parameter Inspection", command=self.show_params_inspect)
        inspect_btn.pack(side=Tk.LEFT, padx=20)
        change_btn = Tk.Button(frame, text="Change Parameters", command=self.show_params_selector)
        change_btn.pack(side=Tk.LEFT, padx=20)

        ratio_frame = Tk.Frame(frame)
        ratio_frame.pack(side=Tk.LEFT, padx=20)
        label = Tk.Label(ratio_frame, text="Zoom Area Ratio: ")
        label.pack(side=Tk.LEFT)
        self.zoom_area_ratio = TickScale(ratio_frame, orient='horizontal', from_=0.1, to=1.0, resolution=ZOOM_AREA_RESOLUTION)
        self.zoom_area_ratio.pack(side=Tk.LEFT)
        self.zoom_area_ratio.set(INIT_ZOOM_ARE_RATIO)
        ToolTip(ratio_frame, 'Use arrow keys (←, →) to adjust this ratio.')

        redraw_btn = Tk.Button(frame, text="Redraw", command=self.redraw)
        redraw_btn.pack(side=Tk.LEFT, padx=20)

        self.bind("<Left>", self.on_arrow_touch)
        self.bind("<Right>", self.on_arrow_touch)
        self.state_sensitive_widgets = [inspect_btn, change_btn, self.zoom_area_ratio, redraw_btn]

    def on_arrow_touch(self, event):
        if event.keysym == "Left":
            d = -ZOOM_AREA_RESOLUTION
        else:
            d = ZOOM_AREA_RESOLUTION
        new_value = max(MIN_ZOOM_ARE_RATIO, min(MAX_ZOOM_ARE_RATIO, self.zoom_area_ratio.get() + d))
        self.zoom_area_ratio.set(new_value)

    def get_zoom_area_ratio(self):
        return INIT_ZOOM_ARE_RATIO if self.zoom_area_ratio is None else self.zoom_area_ratio.get()

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="◀ Close", width=10, command=self.cancel)
        w.pack(pady=10)

        self.bind("<Escape>", self.cancel)

    def get_area(self, zoom=None):
        xmin, xmax = self.bounds[self.i]
        ymin, ymax = self.bounds[self.j]

        if zoom is None:
            xmin_, xmax_ = xmin, xmax
            ymin_, ymax_ = ymin, ymax
        else:
            cx, cy, proportion = zoom
            dx = (xmax - xmin)*proportion/2
            dy = (ymax - ymin)*proportion/2
            xmin_, xmax_ = cx - dx, cx + dx
            ymin_, ymax_ = cy - dy, cy + dy

        return xmin_, xmax_, ymin_, ymax_

    def compute_surface(self, xmin, xmax, ymin, ymax):
        x = np.linspace(xmin, xmax, 40)
        y = np.linspace(ymin, ymax, 40)
        xx, yy = np.meshgrid(x, y)
        zz = self.two_param_vector_func(xx, yy)
        return xx, yy, zz

    def get_suptitle_text(self):
        return "Two Parameter Analysis for %s with selected parameters(%d, %d)" % (self.in_folder, self.i, self.j)

    def draw_function(self):
        ax1, ax2, ax3 = self.axes
        ax1.set_title("3D View (Zoomed)", fontsize=16)
        ax2.set_title("Contour View (Whole)", fontsize=16)
        ax3.set_title("Contour View (Zoomed)", fontsize=16)

        trace_color = "green"
        trace_kwargs = {"color":trace_color, "label":"minima trace", "alpha":0.5}
        initial_kwargs = {"color":"cyan", "label":"initial position"}
        current_kwargs = {"color":"yellow", "label":"current position"}
        # cmap = "viridis"
        cmap = "copper"

        cx = self.params[self.i]
        cy = self.params[self.j]
        warea = self.get_area()
        wxx, wyy, wzz = self.compute_surface(*warea)
        ratio = self.get_zoom_area_ratio()
        sarea = self.get_area(zoom=(cx, cy, ratio))
        sxx, syy, szz = self.compute_surface(*sarea)

        xlabel = "parameter %d" % self.i
        ylabel = "parameter %d" % self.j
        ax1.plot_surface(sxx, syy, szz, cmap=cmap)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_zlabel("FV")

        tx, ty = self.trace_xy
        tz = self.two_param_vector_func(tx, ty)
        cz = self.two_param_func(cx, cy)
        ax1.plot(tx[0], ty[0], tz[0], "o", **initial_kwargs)
        ax1.plot(tx, ty, tz, **trace_kwargs)
        ax1.plot(cx, cy, cz, "o", **current_kwargs)

        levels = None
        for ax, xx, yy, zz in [(ax3, sxx, syy, szz), (ax2, wxx, wyy, wzz)]:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            cs = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap)
            levels = cs.levels
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin, xmax)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            ax.plot([cx, cx], [ymin, ymax], color="yellow")
            ax.plot([xmin, xmax], [cy, cy], color="yellow")
            ax.plot(tx[0], ty[0], "o", **initial_kwargs)
            ax.plot(tx, ty, **trace_kwargs)
            ax.plot(cx, cy, "o", **current_kwargs)

        xmin_, xmax_, ymin_, ymax_ = sarea
        x = [xmin_, xmax_, xmax_, xmin_, xmin_]
        y = [ymin_, ymin_, ymax_, ymax_, ymin_]
        ax2.plot(x, y, ":", color="red", label="zoomed area")

        for ax in [ax1, ax2, ax3]:
            ax.legend()

    def change_state(self, state):
        for w in self.state_sensitive_widgets:
            w.config(state=state)

    def redraw(self):
        self.change_state(Tk.DISABLED)
        self.update()
        self.set_trace_xy()
        self.fig_suptitle.set_text(self.get_suptitle_text())
        for ax in self.axes:
            ax.cla()

        self.draw_function()
        self.mpl_canvas.draw()
        self.change_state(Tk.NORMAL)

    def show_params_inspect(self):
        from .ParamsInspection import ParamsInspection
        dialog = ParamsInspection(self.parent, self.params, self.dsets, self.optimizer)
        dialog.show()

    def show_params_selector(self):
        from .ParamsSelector import ParamsSelector
        dialog = ParamsSelector(self, self.optimizer, np.arange(len(self.params)))
        dialog.set_selection([self.i, self.j])
        dialog.show()
        selection = dialog.get_selection()
        if selection is None:
            return
        else:
            self.i, self.j = selection
            self.redraw()
