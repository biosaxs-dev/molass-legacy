# coding: utf-8
"""
    InspectionInAngles.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkUtils import split_geometry
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable

class InspectionInAnglesDialog(Dialog):
    def __init__(self, parent, q_vector, e_index, drawn_info):
        self.q_vector = q_vector
        self.e_index = e_index
        self.drawn_info = drawn_info
        Dialog.__init__(self, parent, "Inspection in Various Angles",
            # location="lower right"
            visible=False, auto_geometry=False, geometry_cb=self.geometry_cb)

    def show(self):
        self._show()

    def geometry_cb(self):
        self.update()
        parent = self.parent
        rootx = parent.winfo_rootx()
        rooty = parent.winfo_rooty()
        W, H, X, Y = split_geometry(parent.geometry())
        w, h, x, y = split_geometry(self.geometry())
        print([rootx, rooty])
        print([W, H, X, Y], [w, h, x, y])
        # overwrite w, h because they cannot yet be updated correctly
        w, h = 800, 800
        offsetx = W - w - 100
        offsety = H - h - 100
        self.geometry("+%d+%d" % (rootx+offsetx, rooty+offsety))

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()
        bframe = Tk.Frame(body_frame)
        bframe.pack(fill=Tk.X, expand=1)
        tbframe = Tk.Frame(bframe)
        tbframe.pack(side=Tk.LEFT, fill=Tk.X, expand=1)
        pframe = Tk.Frame(bframe)
        pframe.pack(side=Tk.RIGHT, padx=50)
        self.builf_control_panel(pframe)

        figsize = (10, 8)
        fig = plt.figure(figsize=figsize)
        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tbframe)
        self.toolbar.update()
        self.ax = fig.gca()
        self.draw_decomposition()

    def draw_decomposition(self, e_index=None):
        ax = self.ax
        ax.cla()
        title, data, e_index_init, start, detail = self.drawn_info
        if e_index is None:
            e_index = e_index_init
        ax.set_title(title, fontsize=20)
        self.parent.solver.draw_components_with_bands(ax, data, e_index, start=start, detail=detail)
        self.mpl_canvas.draw()

    def builf_control_panel(self, pframe):
        space = Tk.Frame(pframe, width=50)
        space.pack(side=Tk.LEFT)

        label = Tk.Label(pframe, text="Q: ")
        label.pack(side=Tk.LEFT)
        self.q_value = Tk.StringVar()
        self.q_value.set('%.3g' % self.q_vector[self.e_index])
        label = Tk.Label(pframe, textvariable=self.q_value, width=6, justify=Tk.CENTER)
        label.pack(side=Tk.LEFT)

        label = Tk.Label(pframe, text="Angular index: ")
        label.pack(side=Tk.LEFT)
        self.angular_index = Tk.IntVar()
        self.angular_index.set(self.e_index)
        spinbox = Tk.Spinbox(pframe, textvariable=self.angular_index,
                                from_=0, to=len(self.q_vector)-1, increment=10,
                                justify=Tk.CENTER, width=6 )
        spinbox.pack(side=Tk.LEFT)
        self.angular_index.trace('w', self.angular_index_tracer)

    def angular_index_tracer(self, *args):
        index = self.angular_index.get()
        try:
            self.draw_decomposition(index)
            self.q_value.set('%.3g' % self.q_vector[index])
        except:
            pass
