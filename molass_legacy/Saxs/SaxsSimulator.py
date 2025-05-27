# coding: utf-8
"""
    SaxsSimulator.py

    Copyright (c) 2019-2021, SAXS Team, KEK-PF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from OurToplevel import OurToplevel
from molass_legacy.KekLib.TkUtils import split_geometry
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib import get_color, MplBackGround, reset_to_default_style
from OurManim import manim_init, use_default_style, Animation, Collection, TextGroup, Parallelogram, rotation, angle
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from .EdPlotter import get_name_for_title, EdPlotter
from .SaxsDetector import SaxsDetector
from .SaxsCurve import SaxsCurve
from molass_legacy.KekLib.NumpyUtils import np_loadtxt_robust

class SaxsSimulator(OurToplevel):
    def __init__(self, parent, path=None, data=None, save_button=False, animate=False, on_close=None):
        self.parent = parent
        self.path = path
        self.save_button = save_button
        self.animate = animate
        self.data = data
        self.on_close = on_close
        self.esc = None
        self.popup_menu = None
        OurToplevel.__init__(self, parent, "SAXS Simulator")

    def close(self):
        self.destroy()
        if self.on_close is not None:
            self.on_close()

    def show(self):
        # dummy for compatibility with Dialog
        pass

    def body(self, body_frame):
        tk_set_icon_portable(self)
        self.mpl_bg = MplBackGround()
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        bframe = Tk.Frame(body_frame)
        bframe.pack(fill=Tk.BOTH, expand=1)
        tframe = Tk.Frame(bframe)
        tframe.pack(side=Tk.LEFT)
        pframe = Tk.Frame(bframe)
        pframe.pack(side=Tk.RIGHT)

        self.fig  = plt.figure(figsize=(21,7))
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas.mpl_connect('button_press_event', self.on_mpl_button_press)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        self.build_control_panel(pframe)
        self.draw(self.path)

        self.add_dnd_bind()

    def add_dnd_bind(self):
        self.mpl_canvas_widget.register_drop_target("*")

        def dnd_handler(event):
            self.on_entry(event)

        self.mpl_canvas_widget.bind("<<Drop>>", dnd_handler)

    def build_control_panel(self, pframe):
        pass

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        if self.save_button:
            w = Tk.Button(box, text="Save", width=10, command=self.save)
            w.pack(side=Tk.LEFT, padx=5, pady=5)

    def ok(self):
        self.close()

    def draw(self, file=None):
        fig = self.fig
        fig.clf()

        if file is None:
            from_name = ''
        else:
            from_name = ' from ' + get_name_for_title(file)
        fig.suptitle('SAXS Simulation' + from_name, fontsize=24)
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        self.axes = [ax1, ax2, ax3]
        fig.tight_layout()
        fig.subplots_adjust(top=0.85, wspace=0.15, bottom=0.1)

        if file is None and self.data is None:
            self.draw_blank()
        else:
            q, in_y = self.get_input_curve_data(file)

            if self.data is None:
                import mrcfile
                with mrcfile.open(file) as mrc:
                    data = mrc.data
            else:
                data = self.data

            cmap = cm.plasma
            denss_results = file is not None and os.path.exists(file)
            self.esc = EdPlotter(fig, ax1, data, cmap, file, denss_results=denss_results)

            if self.animate:
                # call self.esc.make_anim with after
                # to release the D&D source window
                self.after(0, lambda: self.esc.make_anim(random=True))

            self.detector = SaxsDetector(fig, ax3, data, q, in_y)
            r_y = self.detector.curve_y
            self.curve = SaxsCurve(fig, ax2, q, in_y, r_y)
        self.mpl_canvas.draw()

    def draw_blank(self):
        fig = self.fig
        for ax in self.axes:
            ax.set_axis_off()
        ax2 = self.axes[1]
        ax2.set_xlim(0,1)
        ax2.text(0.5, 0.5, "Drag and drop\nan mrc file to view.", ha='center', fontsize=50, alpha=0.5)
        fig.tight_layout()

    def on_entry(self, event):
        files = event.data.split(' ')
        print('on_entry:', files)
        self.mpl_bg = MplBackGround()   # required when Toplevel instead of Dialog
        self.draw(files[0])
        reset_to_default_style()

    def on_mpl_button_press(self, event):
        if self.esc is None:
            if event.button == 3:
                self.create_popup_menu()
                w, h, x, y = split_geometry(self.geometry())
                self.popup_menu.post(x + event.x + 20, y + h - event.y - 50)
            return

        self.esc.on_click(event)

    def create_popup_menu(self):
        if self.popup_menu is None:
            self.popup_menu = Tk.Menu( self, tearoff=0 )
            self.popup_menu.add_command( label='Ball', command=self.show_ball )
            self.popup_menu.add_command( label='Ellipsoid', command=self.show_ellipsoid )
            self.popup_menu.add_command( label='Disc', command=self.show_disc )
            self.popup_menu.add_command( label='Rod', command=self.show_rod )
            self.popup_menu.add_command( label='Torus', command=self.show_torus )
            self.popup_menu.add_command( label='from PDB', command=self.show_pdb_dialog)

    def get_input_curve_data(self, file):
        if file is None:
            file_exists = False
        else:
            path = file.replace('.mrc', '.dat')
            if os.path.exists(path):
                array, comments = np_loadtxt_robust(path)
                q = array[:,0]
                y = array[:,1]
                file_exists = True
            else:
                file_exists = False
        if not file_exists:
            # TODO: make an appropriate q
            q = np.linspace(0, 0.5, 600)
            y = None
        return q, y

    def save(self):
        pass

    def show_ball(self):
        from molass_legacy.Saxs.SaxsSamples import BallVoxels
        ball = BallVoxels(radius=0.2)
        self.path = "a ball"
        self.data = ball.get_data(density=0.1)
        self.mpl_bg = MplBackGround()   # required when Toplevel instead of Dialog
        self.draw()
        reset_to_default_style()

    def show_ellipsoid(self):
        from molass_legacy.Saxs.SaxsSamples import EllipsoidVoxels
        ball = EllipsoidVoxels(radii=(0.24, 0.2, 0.16))
        self.path = "an ellipsoid"
        self.data = ball.get_data(density=0.1)
        self.mpl_bg = MplBackGround()   # required when Toplevel instead of Dialog
        self.draw()
        reset_to_default_style()

    def show_disc(self):
        from molass_legacy.Saxs.SaxsSamples import DiscVoxels
        disc = DiscVoxels(radius=0.3)
        self.path = "a disc"
        self.data = disc.get_data(density=0.1)
        self.mpl_bg = MplBackGround()   # required when Toplevel instead of Dialog
        self.draw()
        reset_to_default_style()

    def show_rod(self):
        from molass_legacy.Saxs.SaxsSamples import DiscVoxels
        disc = DiscVoxels(radius=0.05, height=0.5)
        self.path = "a rod"
        self.data = disc.get_data(density=0.1)
        self.mpl_bg = MplBackGround()   # required when Toplevel instead of Dialog
        self.draw()
        reset_to_default_style()

    def show_torus(self):
        from molass_legacy.Saxs.SaxsSamples import TorusVoxels
        torus = TorusVoxels(R=0.6, r=0.2)
        self.path = "a torus"
        self.data = torus.get_data(density=0.1)
        self.mpl_bg = MplBackGround()   # required when Toplevel instead of Dialog
        self.draw()
        reset_to_default_style()

    def show_pdb_dialog(self):
        from molass_legacy._MOLASS.SerialSettings import get_setting

        analysis_folder = get_setting('analysis_folder')
        if analysis_folder is None:
            import molass_legacy.KekLib.CustomMessageBox as MessageBox
            MessageBox.showinfo(
                'Retry Suggestion',
                'This operation requires an output folder.\n'
                'Please retry after specifying "Analysis Result Folder"\n'
                'in the Main Dialog.\n',
                parent=self,
                )
            return

        from Pdb.PdbFetcherDialog import PdbFetcherDialog
        dialog = PdbFetcherDialog(self)
        dialog.show()
        if dialog.applied:
            self.mpl_bg = MplBackGround()   # required when Toplevel instead of Dialog
            mrc_file = dialog.get_mrc_file_path()
            self.draw(mrc_file)
            reset_to_default_style()
