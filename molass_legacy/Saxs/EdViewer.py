"""
    EdViewer.py

    Copyright (c) 2019-2025, SAXS Team, KEK-PF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk
from molass_legacy.KekLib.OurToplevel import OurToplevel
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib import get_color, MplBackGround, reset_to_default_style
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from .EdPlotter import ed_scatter

class EdViewer(OurToplevel):
    def __init__(self, parent, path=None, data=None, save_button=False, on_close=None):
        self.mpl_bg = MplBackGround()
        self.parent = parent
        self.path = path
        self.save_button = save_button
        self.data = data
        self.on_close = on_close
        OurToplevel.__init__(self, parent, "Electron Density Viewer")

    def close(self):
        self.destroy()
        if self.on_close is not None:
            self.on_close()

    def show(self):
        # dummy for compatibility with Dialog
        pass

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()
        bframe = Tk.Frame(body_frame)
        bframe.pack(fill=Tk.BOTH, expand=1)
        tframe = Tk.Frame(bframe)
        tframe.pack(side=Tk.LEFT)
        pframe = Tk.Frame(bframe)
        pframe.pack(side=Tk.RIGHT)

        self.fig  = plt.figure(figsize=(16,7))
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
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

    def draw(self, file):
        fig = self.fig
        fig.clf()
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, aspect='equal')
        self.axes = [ax1, ax2]

        if file is None:
            self.draw_blank()
        else:
            if self.data is None:
                import mrcfile
                with mrcfile.open(file) as mrc:
                    data = mrc.data
            else:
                data = self.data
            self.esc = ed_scatter(fig, self.axes, data, file)
            # call self.esc.make_anim with after
            # to release the D&D source window
            # self.after(0, self.esc.make_anim)

        self.mpl_canvas.draw()

    def draw_blank(self):
        fig = self.fig
        for ax in self.axes:
            ax.set_axis_off()
        ax1, ax2 = self.axes
        ax2.set_xlim(0,1)
        ax2.text(0.5, 0.5, "Drag and drop\nan mrc file to view.", ha='center', fontsize=50, alpha=0.5)
        fig.tight_layout()

    def on_entry(self, event):
        files = event.data.split(' ')
        print('on_entry:', files)

        self.mpl_bg = MplBackGround()
        self.draw(files[0])
        reset_to_default_style()

    def save(self):
        pass
