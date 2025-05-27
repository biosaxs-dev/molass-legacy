# coding: utf-8
"""
    AffineDemo.py

    Copyright (c) 2018,2024, Masatsuyo Takahashi, KEK-PF
"""

import numpy                as np
import matplotlib.pyplot    as plt
from matplotlib.patches     import Polygon
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
from molass_legacy.KekLib.TkUtils                import is_low_resolution
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
from molass_legacy.KekLib.BasicUtils             import get_caller_module
from molass_legacy.KekLib.Affine                 import Affine


def demo_plot( fig, src_points, tgt_points, with_proof=False ):
    affine  = Affine( src_points, tgt_points )

    src_polygon = Polygon( src_points, alpha=0.2 )
    tgt_polygon = Polygon( tgt_points, alpha=0.2 )

    ax1 = fig.add_subplot( 121 )
    ax2 = fig.add_subplot( 122 )

    ax1.set_title( 'source shapes (on the standard mapping plane)' )
    ax2.set_title( 'affine-transformed shapes (on another Q-plane )' )

    for ax in [ ax1, ax2 ]:
        ax.set_xlim( 0, 1 )
        ax.set_ylim( 0, 1 )

    ax1.add_patch(src_polygon)

    x   = np.arange( 0.1, 0.95, 0.1 )
    src_line    = x * 0.1 + 0.05
    ax1.plot( x, src_line, color='red' )

    ax2.add_patch(tgt_polygon)

    x_, y_ = affine.transform( x, src_line )
    ax2.plot( x_, y_, color='red' )

    for point in src_points:
        x, y = point
        ax1.plot( x, y, 'o', markersize=10 )

    for point in tgt_points:
        x, y = point
        ax2.plot( x, y, 'o', markersize=10 )

    if with_proof:
        if True:
            src_x = [ p[0] for p in src_points ]
            src_y = [ p[1] for p in src_points ]
            tgt_x, tgt_y = affine.transform( src_x, src_y )
            ax2.plot( tgt_x, tgt_y, 'o', color='black', markersize=10 )
        else:
            tgt_p = affine.transform_list(src_points)
            for point in tgt_p:
                x, y = point
                ax2.plot( x, y, 'o', color='black', markersize=10 )

    fig.tight_layout()

class AffineDemoDialog( Dialog ):
    def __init__( self, title, parent=None ):
        self.grab = 'local'     # used in grab_set
        self.parent     = parent
        self.title_     = title
        self.applied    = None
        self.caller_module = get_caller_module( level=2 )

    def show( self ):
        self.parent.config( cursor='wait' )
        self.parent.update()

        Dialog.__init__( self, self.parent, self.title_ )
        # TODO: self.resizable(width=False, height=False)

    def body( self, body_frame ):   # overrides parent class method
        tk_set_icon_portable( self, module=self.caller_module )

        cframe = Tk.Frame( body_frame )
        cframe.pack()

        figsize = ( 16, 8 ) if is_low_resolution() else ( 20, 10 )
        fig = plt.figure( figsize=figsize )

        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas.draw()
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

        # it seems that draw_func should be called after the creation of mpl_canvas
        # in order to enable 3d-rotation of the figure
        self.draw( fig )
        self.parent.config( cursor='' )

        self.protocol( "WM_DELETE_WINDOW", self.ok )

    def buttonbox( self, frame=None ):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)

    def draw( self, fig ):
        src_points  = [ ( 0.1, 0.1 ), ( 0.9, 0.1 ), ( 0.5, 0.9 ) ]
        tgt_points  = [ ( 0.1, 0.1 ), ( 0.9, 0.3 ), ( 0.3, 0.7 ) ]
        demo_plot( fig, src_points, tgt_points )
