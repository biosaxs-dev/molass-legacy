"""
    Peaks.EditCanvas.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk, ToolTip
from molass_legacy._MOLASS.SerialSettings import get_setting
import molass_legacy.KekLib.CustomMessageBox as MessageBox
import molass_legacy.KekLib.DebugPlot as dplt
from molass_legacy.Peaks.PeakParamsSet import PeakParamsSet

class EditCanvas(Dialog):
    def __init__(self, parent, sd, ecurves, original_info, working_info=None):
        self.parent = parent
        self.sd = sd
        self.ecurves = ecurves
        self.xr_peaks_orig = original_info[2]   # xr_peaks to reset to
        if working_info is None:
            working_info = original_info
        self.xr_peaks_work = working_info[2]    # xr_peaks to undo to 
        working_info[2] = self.xr_peaks_work.copy()
        self.xr_draw_info = working_info
        self.applied = False
        self.peak_params_set = None
        self.appliable = False
        Dialog.__init__(self, parent, "Edit Canvas", visible=False)

    def cancel(self):
        # overiding cancel to cleanup self.fig
        # because the call to the destructor __del__ seems to be delayed
        plt.close(self.fig)
        # print("ExtrapolSolverDialog: closed fig")
        Dialog.cancel(self)

    def show( self ):
        self._show()

    def body(self, body_frame, devel=True):
        if devel:
            from importlib import reload
            import molass_legacy.KekLib.DraggableCurves
            reload(KekLib.DraggableCurves)
        from molass_legacy.KekLib.DraggableCurves import DraggableCurves

        upper_frame = Tk.Frame(body_frame)
        upper_frame.pack(fill=Tk.X)
        lower_frame = Tk.Frame(body_frame)
        lower_frame.pack()
        self.upper_frame = upper_frame

        cframe = Tk.Frame(upper_frame)
        cframe.pack(side=Tk.LEFT)
        cframe_ = Tk.Frame(cframe)
        cframe_.pack()
        tframe_ = Tk.Frame(cframe)
        tframe_.pack(side=Tk.LEFT, fill=Tk.X)

        fig, ax = plt.subplots(figsize=(12,8))
        self.fig = fig
        self.ax = ax
        self.axt = ax.twinx()
        # https://stackoverflow.com/questions/55565393/matplotlib-picker-event-on-secondary-y-axis
        ax.set_zorder(self.axt.get_zorder() + 1)

        self.dcurves = DraggableCurves(ax, x_only=True,
                            button_release_user_callback=self.button_release_user_callback,
                            debug=True)

        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe_)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1)
        self.draw_components()

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe_)
        self.toolbar.update()

        dcurves = self.dcurves
        # self.mpl_canvas.mpl_connect('button_press_event', self.on_click)
        self.mpl_canvas.mpl_connect('button_release_event', dcurves.button_release_callback)
        self.mpl_canvas.mpl_connect('pick_event', dcurves.pick_callback)
        self.mpl_canvas.mpl_connect('motion_notify_event', dcurves.motion_notify_callback)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack(fill=Tk.X, padx=20)

        num_buttons = 5
        for j in range(num_buttons):
            box.columnconfigure(j, weight=1)

        w = Tk.Button(box, text="Cancel", width=12, command=self.user_cancel)
        col = 0
        w.grid(row=0, column=col, pady=10)
        self.cancel_btn = w
        ToolTip(w, "discard all changes here")

        col += 1
        w = Tk.Button(box, text="Reset", width=12, command=self.reset)
        w.grid(row=0, column=col, pady=10)
        self.reset_btn = w
        ToolTip(w, "reset to the initial state of Peak Editor")

        col += 1
        w = Tk.Button(box, text="Undo", width=12, command=self.undo)
        w.grid(row=0, column=col, pady=10)
        self.undo_btn = w
        ToolTip(w, "undo to the initial state here")

        col += 1
        w = Tk.Button(box, text="Adjust", width=10, command=self.adjust_heights)
        w.grid(row=0, column=col, pady=10)
        self.adjust_btn = w
        ToolTip(w, "optimize parameters while fixing peak top positions")

        col += 1
        w = Tk.Button(box, text="Ok", width=12, command=self.ok)
        w.grid(row=0, column=col, pady=10)
        self.proceed_btn = w
        ToolTip(w, "aplly the change result here")

        assert col+1 == num_buttons

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.user_cancel)
        self.bind("<Control-z>", lambda event: self.undo())

    def draw_components(self, devel=False):
        if devel:
            from importlib import reload
            import QuickAnalysis.ModeledPeaks
            reload(QuickAnalysis.ModeledPeaks)
        from molass_legacy.QuickAnalysis.ModeledPeaks import plot_curve

        ax = self.ax
        axt = self.axt
        ax.cla()
        axt.cla()
        axt.grid(False)

        ax.set_title("Xray Elution at Q=%.3g" % get_setting("intensity_picking"), fontsize=16)

        self.dcurves.clear()
        xr_x, xr_y, xr_peaks, xr_baseline = self.xr_draw_info
        self.markers = plot_curve(ax, xr_x, xr_y, xr_peaks, color='orange', baseline=xr_baseline,
                    dcurves=self.dcurves, return_markers=True)
        peaktops = []
        for marker in self.markers:
            xdata, ydata = marker.get_data()
            peaktops.append((xdata[0], ydata[0]))
        self.peaktops = np.array(peaktops)

        ax.set_xlim(*ax.get_xlim())     # fix for editting
        self.fig.tight_layout()
        self.mpl_canvas.draw()

    def button_release_user_callback(self, dcurves):
        print("button_release_user_callback")
        dx, dy = dcurves.get_last_displacement()
        x, y = dcurves.picked_artist.get_data()
        x = x - dx  # note that x -= dx would change the artist's data
        j = np.argmax(y)
        px = x[j]
        py = y[j]
        diffs = self.peaktops - np.array((px,py))
        dists = diffs[:,0]**2 + diffs[:,1]**2
        m = np.argmin(dists)
        self.markers[m].set_data(px+dx, py)

    def user_cancel(self, ask=True):
        self.cancel()

    def reset(self):
        self.appliable = False
        self.xr_draw_info[2] = self.xr_peaks_orig
        self.draw_components()

    def undo(self):
        self.xr_draw_info[2] = self.xr_peaks_work
        self.draw_components()

    def adjust_heights(self, debug=False):
        if debug:
            from importlib import reload
            import QuickAnalysis.ModeledPeaks
            reload(QuickAnalysis.ModeledPeaks)
        from molass_legacy.QuickAnalysis.ModeledPeaks import adjust_peak_heights, plot_curve

        uv_curve, xr_curve = self.ecurves
        xr_x = xr_curve.x
        xr_y = xr_curve.y
        uv_x = uv_curve.x
        uv_y = uv_curve.y

        self.update_xr_peaks()
        a, b = self.parent.get_pre_recog_mapping_params()
        new_peaks = self.get_xr_peaks()
        new_xr_peaks = adjust_peak_heights(1, 0, xr_x, xr_y, new_peaks, xr_x, xr_y, debug=debug)
        new_uv_peaks = adjust_peak_heights(a, b, xr_x, xr_y, new_xr_peaks, uv_x, uv_y, debug=debug)
        if debug:
            with dplt.Dp():
                fig, (ax1, ax2) = dplt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("adjust_heights debug")
                plot_curve(ax1, uv_x, uv_y, new_uv_peaks, color='blue')
                plot_curve(ax2, xr_x, xr_y, new_xr_peaks, color='orange')
                dplt.show()
        self.peak_params_set = PeakParamsSet(new_uv_peaks, new_xr_peaks, a, b)
        self.xr_peaks_work = new_xr_peaks
        # self.update_xr_peaks()
        self.xr_draw_info[2] = self.xr_peaks_work
        self.draw_components()
        self.appliable = True

    def get_peak_params_set(self):
        return self.peak_params_set

    def validate(self):
        if self.appliable:
            return 1
        else:
            import molass_legacy.KekLib.CustomMessageBox as MessageBox
            MessageBox.showinfo("Notification",
            'Please press "Adjust" button to confirm the edit result\n'
            'before applying.',
            parent=self,
            )
            return 0

    def apply(self, show_message=False):
        self.applied = True
        if show_message:
            MessageBox.showinfo("Remainder",
                "Remember that the results will be sorted, locally optimized, and renumbered.",
                parent=self)
        self.update_xr_peaks()

    def update_xr_peaks(self):
        displ_list = self.dcurves.get_displacements()
        print("displacements: ", displ_list)
        new_xr_peaks = []
        for k, params in enumerate(self.xr_peaks_work):
            temp_params = np.array(params)
            # note that temp_params[1] for both affine and non-affine models
            temp_params[1] += displ_list[k][0]      # move in x-coordinate only
            new_xr_peaks.append(temp_params)
        self.xr_peaks_work = np.array(sorted(new_xr_peaks, key=lambda p: p[1]))

    def get_xr_peaks(self):
        return self.xr_peaks_work
