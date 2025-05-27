# coding: utf-8
"""
    AbsorbanceViewer.py

    Copyright (c) 2018-2019, SAXS Team, KEK-PF
"""
import os
import numpy                as np
from bisect                 import bisect_right
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
from mpl_toolkits.mplot3d   import Axes3D
from matplotlib.widgets     import Button
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
from molass_legacy.KekLib.TkUtils                import is_low_resolution
from molass_legacy.KekLib.BasicUtils             import get_caller_module
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib          import NavigationToolbar
from molass_legacy._MOLASS.SerialSettings         import get_setting
from molass_legacy.KekLib.TkUtils                import split_geometry
from molass_legacy.KekLib.TkCustomWidgets        import FileEntry
from .AbsorbanceViewUtils import draw_3d
import matplotlib
USE_CANVAS_3D = True
if USE_CANVAS_3D:
    from Canvas3D import Canvas3D

DEBUG = False

class AbsorbanceViewer( Dialog ):
    def __init__( self, absorbance, helper_info=None ):
        self.grab = 'local'     # used in grab_set
        self.absorbance     = absorbance
        self.helper_info    = helper_info
        self.wl_vector      = absorbance.wl_vector
        self.wvlen_lower    = 245
        self.wvlen_upper    = 450
        if True:
            f = 0
            t = len(absorbance.wl_vector)
        else:
            f = bisect_right( absorbance.wl_vector, self.wvlen_lower )
            t = bisect_right( absorbance.wl_vector, self.wvlen_upper )
        self.i_slice        = slice( f, t )
        self.applied    = None
        self.caller_module = get_caller_module( level=2 )
        self.mplt_ge_2_2 = matplotlib.__version__ >= '2.2'

    def show( self, parent, button_labels=[ "OK", "Cancel" ], toolbar=False ):
        assert( len(button_labels) == 2 )
        self.button_labels  = button_labels
        self.toolbar    = toolbar

        Dialog.__init__( self, parent, "AbsorbanceViewer" )
        # TODO: self.resizable(width=False, height=False)

    def body( self, body_frame ):   # overrides parent class method
        tk_set_icon_portable( self, module=self.caller_module )

        cframe = Tk.Frame( body_frame )
        cframe.pack()

        figsize = ( 9, 8 ) if is_low_resolution() else ( 11, 10 )
        self.fig = fig = plt.figure( figsize=figsize )

        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

        # it seems that add_subplot should be called after the creation of mpl_canvas
        # in order to enable 3d-rotation of the figure
        if USE_CANVAS_3D:
            c3d = Canvas3D(fig, 23, 5 )
            ax1 = c3d.get_axis()
            lpm_demo_ax = plt.axes([0.82, 0.26, 0.12, 0.04])
            self.lpm_demo_btn = Button(lpm_demo_ax, 'LPM area', hovercolor='0.975')
            self.lpm_demo_btn.on_clicked(self.toggle_lpm_area)
            self.show_lpm_area = False
            baseplane_ax = plt.axes([0.82, 0.22, 0.12, 0.04])
            self.baseplane_btn = Button(baseplane_ax, 'base plane', hovercolor='0.975')
            self.baseplane_btn.on_clicked(self.toggle_baseplane)
            self.show_baseplane = False
        else:
            ax1 = fig.add_subplot( 111, projection='3d' )
            # ax2 = fig.add_subplot( 122 )
        self.axes   = [ax1, None]

        if self.mplt_ge_2_2:
            self.mpl_canvas.draw()
        else:
            self.mpl_canvas.show()

        self.draw()
        self.parent.config( cursor='' )

        if self.toolbar:
            self.toolbar = NavigationToolbar( self.mpl_canvas, cframe )
            self.toolbar.update()

        self.protocol( "WM_DELETE_WINDOW", self.ok )

    def buttonbox(self):
        '''add standard button box.

        override if you do not want the standard buttons
        '''

        box = Tk.Frame(self)
        self.button_frame = box

        w = Tk.Button(box, text=self.button_labels[0], width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.ok_button = w
        w = Tk.Button(box, text="Save", width=10, command=self.save_dialog)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.save_button = w
        w = Tk.Button(box, text=self.button_labels[1], width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.cancel_button = w

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def ok( self, *args ):
        self.applied  = True
        self.destroy()

    def cancel( self, *args ):
        self.applied  = False
        self.destroy()

    def draw( self ):
        ax1 = self.axes[0]

        draw_3d( ax1, self.absorbance, self.wvlen_lower, self.wvlen_upper, self.i_slice, low_percentile=self.show_lpm_area )

        if USE_CANVAS_3D:
            if self.show_baseplane:
                self.draw_baseplane(ax1)
        else:
            self.fig.tight_layout()

        self.mpl_canvas.draw()

    def toggle_lpm_area( self, event ):
        # print('toggle_lpm_area')
        self.show_lpm_area ^= True
        ax1 = self.axes[0]
        ax1.cla()
        self.draw()

    def toggle_baseplane( self, event ):
        # print('toggle_baseplane')
        self.show_baseplane ^= True
        ax1 = self.axes[0]
        ax1.cla()
        self.draw()

    def draw_baseplane( self, ax ):
        A, B, C = self.absorbance.get_baseplane_params()
        # print( 'draw_baseplane(%d): A=%.4g, B=%.4g, C=%.4g' % ( id(self.absorbance), A, B, C ) )
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        xx, yy = np.meshgrid( np.linspace(xmin, xmax, 10), np.linspace(ymin, ymax, 10) )
        zz = xx * A + yy * B + C
        ax.plot_surface(xx, yy, zz, color='red', alpha=0.1 )

    def save_dialog( self ):
        from .AbsorbanceSaveDialog import AbsorbanceSaveDialog
        dialog = AbsorbanceSaveDialog( self )
        dialog.show()
