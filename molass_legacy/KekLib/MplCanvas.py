# coding: utf-8
"""

    MplCanvas.py

    to avoid
    invalid command name "1849982244872filter_destroy"
    from DerbugPlot

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF

"""
import tkinter as Tk
import matplotlib
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg

class MplCanvas(Tk.Frame):
    def __init__(self, parent, *args):
        self.parent = parent
        self.mplt_ge_2_2 = matplotlib.__version__ >= '2.2'
        Tk.Frame.__init__(self, parent, *args)

    def attach_figure(self, fig):
        print( fig.get_size_inches() )
        self.mpl_canvas = FigureCanvasTkAgg( fig, self )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

    def detach_figure(self):
        pass

    def destroy(self):
        self.detach_figure()
        super(Tk.Frame,self).destroy()

    def show(self):
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        print( "MplCanvas: reqsize", w, h )

        if self.mplt_ge_2_2:
            self.mpl_canvas.draw()
        else:
            self.mpl_canvas.show()

        # TODO: resize appropriately
        # https://ja.osdn.net/projects/pylaf/scm/hg/pylaf/blobs/tip/src/pylafiii/mplext.py
        print(self.mpl_canvas_widget.winfo_geometry())
