# coding: utf-8
"""

    ファイル名：   MplPlotCanvas.py

    処理内容：

        デバッグ用の plot 

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF

"""
import numpy                            as np
from mpl_toolkits.mplot3d               import Axes3D
import matplotlib.pyplot                as plt
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter                         import Tk, Dialog

class MplPlotCanvas( Tk.Toplevel ):
    def __init__( self, title, draw_func, parent=None ):

        Tk.Toplevel.__init__( self )

        if parent is None:
            parent = Tk.Toplevel()
            parent.withdraw()
            self.created_parent = True
        else:
            self.created_parent = False

        parent.update()
        self.parent     = parent
        self.draw_func  = draw_func

        frame = Tk.Frame( self )
        frame.pack()

        self.body( frame )

    def quit( self ):
        if self.created_parent:
            self.parent.destroy()

        self.destroy()

    def body( self, body_frame ):   # overrides parent class method

        fig = plt.figure()

        self.mpl_canvas = FigureCanvasTkAgg( fig, body_frame )
        self.mpl_canvas.draw()
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        # self.mpl_canvas.mpl_connect( 'button_press_event', self.button_press )
        # self.mpl_canvas.mpl_connect( 'key_press_event', self.key_press )

        self.draw_func( fig )

        # self.protocol( "WM_DELETE_WINDOW", self.quit )

    def button_press( self, *argv ):
        print( 'button_press' )
        self.mpl_canvas.draw()

    def key_press( self, *argv ):
        print( 'key_press' )
        self.mpl_canvas.draw()
