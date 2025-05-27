# coding: utf-8
"""
    VoxelFFT.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon, Circle
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib import get_color
from OurManim import manim_init, use_default_style, Animation, Collection, TextGroup, Parallelogram, rotation, angle
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from .VoxelUtils import draw_voxel

class VoxelFFT(Dialog):
    def __init__(self, parent=None):
        if parent is None:
            from SerialTestUtils import get_tk_root
            parent = get_tk_root()
        self.parent = parent
        # manim_init()
        Dialog.__init__(self, parent, "SVD Tutorial", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        self.fig = fig = plt.figure(figsize=(16,8))
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, cframe)
        self.toolbar.update()
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        self.axes = ax1, ax2
        self.draw()
        self.fig.tight_layout()
        self.mpl_canvas.draw()

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Save", width=10, command=self.save)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

    def draw(self):
        ax, _ = self.axes
        # tensor = np.random.rand(3, 3, 3)
        tensor = np.ones((3, 3, 3))
        print(tensor)
        draw_voxel(ax, tensor)

        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 20)
        ax.set_zlim(-10, 10)

    def save(self):
        pass
