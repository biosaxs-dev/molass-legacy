"""

    ThreeDimViewer.py

    Copyright (c) 2019-2022, SAXS Team, KEK-PF

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.TkUtils import split_geometry
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from DataUtils import get_in_folder
from MatrixData import simple_plot_3d

class ThreeDimViewer(Dialog):
    def __init__(self, parent, md):
        self.md = md
        self.popup_menu = None
        Dialog.__init__(self, parent, "3D Viewer", visible=False)

    def cancel(self):
        # overiding cancel to cleanup self.fig
        # because the call to the destructor __del__ seems to be delayed
        plt.close(self.fig)
        Dialog.cancel(self)

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tbframe = Tk.Frame(body_frame)
        tbframe.pack(fill=Tk.X, expand=1)

        md = self.md
        self.fig = fig = plt.figure(figsize=(21,7))

        self.mpl_canvas = FigureCanvasTkAgg( self.fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tbframe)
        self.toolbar.update()

        in_folder = get_in_folder()
        fig.suptitle(in_folder, fontsize=20)
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133)
        self.axes = [ax1, ax2, ax3]
        md.uv.plot(ax=ax1)
        md.xr.plot(ax=ax2)
        md.plot_pre_sync(ax3)

        """
            show guide for how to zoom
            learned at https://pythonprogramming.net/3d-graphs-matplotlib-tutorial/
        """
        axg = fig.add_axes([0,0,0.66,0.05])
        axg.set_axis_off()
        axg.text(0.5, 0.3, "Drag with the right mouse button to zoom in or out in 3d plots.", fontsize=16, alpha=0.3, ha='center')

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        fig.canvas.mpl_connect( 'button_press_event', self.on_button_press )
        self.mpl_canvas.draw()

    def show(self):
        self._show()

    def save_the_figure(self, folder=None, file=None):
        import os
        from molass_legacy._MOLASS.SerialSettings import get_setting
        if folder is None:
            folder = get_setting('analysis_folder')
        if file is None:
            file = get_setting('analysis_name')
        path = os.path.join(folder, file)
        self.fig.savefig(path)

    def on_button_press(self, event):
        if event.button == 3:
            self.on_right_button_press(event)
            return

    def on_right_button_press(self, event):
        if event.inaxes != self.axes[1]:
            return

        if self.popup_menu is None:
            self.create_popup_menu()

        rootx = self.winfo_rootx()
        rooty = self.winfo_rooty()
        w, h, x, y = split_geometry(self.mpl_canvas_widget.winfo_geometry())
        self.popup_menu.post(rootx + int(event.x), rooty + h - int(event.y))

    def create_popup_menu(self):
        self.popup_menu = Tk.Menu(self, tearoff=0)
        self.popup_menu.add_command(label='Show error data', command=self.show_error_data)

    def show_error_data(self):
        dialog = ThreeDimErrorViewer(self, self.md.xr)
        dialog.show()

class ThreeDimErrorViewer(Dialog):
    def __init__(self, parent, xr):
        self.xr = xr
        Dialog.__init__(self, parent, "3D Error View", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tbframe = Tk.Frame(body_frame)
        tbframe.pack(fill=Tk.X, expand=1)

        xr = self.xr

        self.fig = fig = plt.figure(figsize=(14,7))

        self.mpl_canvas = FigureCanvasTkAgg( self.fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tbframe)
        self.toolbar.update()

        in_folder = get_in_folder()
        fig.suptitle(in_folder, fontsize=20)
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        ax1.set_title("Xray Scattering Error (Linear)", fontsize=16, y=1.08)
        simple_plot_3d(ax1, xr.error.data, x=xr.vector)

        ax2.set_title("Xray Scattering Error (Log10)", fontsize=16, y=1.08)
        simple_plot_3d(ax2, np.log10(xr.error.data), x=xr.vector)

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

        self.mpl_canvas.draw()
