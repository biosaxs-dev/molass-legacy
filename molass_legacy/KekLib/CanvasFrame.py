# coding: utf-8
"""

    ファイル名：   CanvasFrame.py

    処理内容：

        デバッグ用の plot 

    Copyright (c) 2017-2021, Masatsuyo Takahashi, KEK-PF

"""
import matplotlib.pyplot                as plt
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter             import Tk
import matplotlib

class CanvasFrame(Tk.Frame):
    def __init__( self, parent, figsize=None, func=None ):
        Tk.Frame.__init__( self, parent )

        self.mplt_ge_2_2 = matplotlib.__version__ >= '2.2'
        cframe = Tk.Frame( self )
        cframe.pack()

        self.fig = fig = plt.figure( figsize=figsize )
        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )

        if func is not None:
            func( fig )
        self.show()
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

    def close_fig(self):
        plt.close(self.fig)

    def draw( self, func ):
        func( self.fig )
        self.show()

    def show( self ):
        if self.mplt_ge_2_2:
            self.mpl_canvas.draw()
        else:
            self.mpl_canvas.show()