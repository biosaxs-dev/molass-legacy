# coding: utf-8
"""

    ファイル名：   MappingZoomCanvas.py

    処理内容：

        濃度対応図の表示

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF

"""
import matplotlib.pyplot                as plt
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter                         import Tk, Dialog
from molass_legacy.KekLib.TkUtils                            import split_geometry, is_low_resolution
from molass_legacy.KekLib.OurMatplotlib                      import NavigationToolbar

DEBUG = False

class MappingZoomCanvas( Tk.Toplevel ):
    def __init__( self, parent, draw_closure, annotation_closure=None, quit_cb=None ):
        self.parent         = parent
        self.draw_closure   = draw_closure
        self.annotation_closure = annotation_closure
        self.quit_cb        = quit_cb
        Tk.Toplevel.__init__( self, parent )
        self.create_canvas()
        self.update()

        W, H, X, Y = split_geometry( self.parent.geometry() )
        self.geometry("+%d+%d" % ( X + 50, Y + 50 ))

        self.protocol( "WM_DELETE_WINDOW", self.quit )

    def quit( self ):
        if self.quit_cb is not None:
            self.quit_cb()
        self.destroy()

    def create_canvas( self ):
        self.cframe = cframe = Tk.Frame( self )
        cframe.pack()

        figsize = ( 10, 8 ) if is_low_resolution() else ( 12, 10 )
        self.fig = fig = plt.figure( figsize=figsize )

        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

        self.ax = fig.add_subplot(111)
        self.draw()
        self.fig.tight_layout()

        self.toolbar = NavigationToolbar( self.mpl_canvas, cframe )
        self.toolbar.update()

        close_button = Tk.Button( self, text="Close", command=self.quit )
        close_button.pack( pady=5 )

    def draw( self, draw_closure=None ):
        if draw_closure is not None:
            self.draw_closure = draw_closure
        self.draw_closure( self.ax )
        if self.annotation_closure is not None:
            self.annotation_closure( self.ax, zoomed=True )
        self.mpl_canvas.draw()

    def canvas_focus_set( self ):
        self.mpl_canvas_widget.focus_set()
