# coding: utf-8
"""
    AbsorbanceViewerPub.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
import numpy                as np
from bisect                 import bisect_right
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
from mpl_toolkits.mplot3d   import Axes3D
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
from molass_legacy.KekLib.TkUtils                import is_low_resolution
from molass_legacy.KekLib.BasicUtils             import get_caller_module
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib          import NavigationToolbar
from molass_legacy._MOLASS.SerialSettings         import get_setting

DEBUG = False

class AbsorbanceViewerPub( Dialog ):
    def __init__( self, absorbance ):
        self.grab = 'local'     # used in grab_set
        self.absorbance     = absorbance
        self.wl_vector      = absorbance.wl_vector
        self.wvlen_lower    = 270
        self.wvlen_upper    = 350
        f = bisect_right( absorbance.wl_vector, self.wvlen_lower )
        t = bisect_right( absorbance.wl_vector, self.wvlen_upper )
        self.i_slice        = slice( f, t )
        self.applied    = None
        self.caller_module = get_caller_module( level=2 )

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
        self.mpl_canvas.draw()
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

        # it seems that draw_func should be called after the creation of mpl_canvas
        # in order to enable 3d-rotation of the figure
        ax1 = fig.add_subplot( 111, projection='3d' )
        # ax2 = fig.add_subplot( 122 )

        self.axes   = [ax1, None]

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

        w = Tk.Button(box, text=self.button_labels[0], width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.ok_button = w
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

        self.draw_3d( ax1 )
        # self.draw_3d_detail( ax1 )
        self.draw_annotion( ax1 )

        self.fig.tight_layout()

    def draw_3d( self, ax, zlim=None ):
        lower   = self.wvlen_lower
        upper   = self.wvlen_upper

        ax.set_title( "" )
        ax.set_xlabel( '\nwave length (nm)' )
        ax.set_ylabel( '\nsequential №' )
        ax.set_zlabel( '\nabsorbance' )
        ax.set_xlim( self.wvlen_lower - 10 , self.wvlen_upper + 10 )

        if zlim is None:
            # zlim = ( -0.05, 0.2 )
            # ax.set_zlim( zlim )
            pass
        else:
            ax.set_zlim( zlim )

        data = self.absorbance.data
        print( 'data.shape=', data.shape )

        size = data.shape[1]

        peak_j = None
        tail_i = self.absorbance.tail_index
        absorbance_baseline_type = get_setting( 'absorbance_baseline_type' )
        for i, wv in enumerate( self.absorbance.wl_vector ):
            if i == tail_i:
                break
            else:
                if wv < self.wvlen_lower or wv > self.wvlen_upper:
                    continue
                if i != self.absorbance.index and i % 2 > 0:
                    continue

            X = np.ones( size ) * wv
            Y = np.arange( size )
            Z = data[i,:]
            if i == self.absorbance.index:
                alpha   = 1
                color   = 'blue'
            else:
                alpha   = 0.2
                color   = '#1f77b4'
            ax.plot( X, Y, Z, color=color, alpha=alpha )

            if absorbance_baseline_type == 0 or self.absorbance.base_curve is None:
                continue

            if i  == tail_i:
                slice_ = slice( self.absorbance.rail_points[0], self.absorbance.rail_points[1]+1 )
                Y_ = Y[slice_]
                curve_size = len(self.absorbance.tail_curve)
                for ii in range( i-5, i+6, 2 ):
                    X_ = np.ones( curve_size ) * self.absorbance.wl_vector[ii]
                    ax.plot( X_, Y_, self.absorbance.tail_curve, color='red', alpha=0.5 )

                X_ = np.ones( curve_size ) * self.absorbance.std_wvlen
                ax.plot( X_, Y_, self.absorbance.base_curve, color='red', alpha=0.5 )

    def draw_3d_detail( self, ax ):
        wvlen   = self.absorbance.wl_vector
        lower   = self.wvlen_lower
        upper   = self.wvlen_upper
        absorbance  = self.absorbance

        wvlen_ = wvlen[self.i_slice]

        if False:
            lower_j  = absorbance.a_curve.lower_abs
            middle_j = absorbance.a_curve.middle_abs
            upper_j  = absorbance.a_curve.upper_abs
        else:
            n       = absorbance.a_curve.primary_peak_no
            info    = absorbance.a_curve.peak_info[n]
            lower_j  = info[0]
            middle_j = int(info[1] + 0.5)
            upper_j  = info[2]

        section_array = []
        colors = [ 'green', 'orange', 'green', 'purple', 'purple' ]
        for i, j in enumerate( [ lower_j, middle_j, upper_j, absorbance.jump_j, absorbance.right_jump_j ] ):
            print( 'draw_3d_absorbance_detail: (i, j)=', (i, j) )
            if j is None:
                continue

            X = wvlen_
            Y = np.ones( len(wvlen_) ) * j
            Z = self.absorbance.data[self.i_slice,j]
            section_array.append( [ j, Z ] )
            color = colors[i]
            ax.plot( X, Y, Z, ':', color=color )

        # self.draw_wave_length_side_2d( ax_2d,  wvlen_, section_array )

    def draw_annotion( self, ax ):
        wvlen   = self.absorbance.wl_vector
        absorbance  = self.absorbance

        bline = absorbance.get_bottomline_matrix()

        xmin, xmax = ax.get_xlim()
        zmin, zmax = ax.get_zlim()
        xoffset = ( xmax - xmin ) * 0.01
        zoffset = ( zmax - zmin ) * 0.005

        Y = np.arange( absorbance.data.shape[1] )
        for i in [0, 3]:
            wvlen = absorbance.reg_wvlens[i]
            X = np.ones( absorbance.data.shape[1] ) * wvlen
            Z = bline[:, i]
            if i == 3:
                color   = 'red'     # 'cyan'
                alpha   = 1
            else:
                color   = 'red'
                alpha   = 1
            ax.plot( X, Y, Z, ':', color=color, alpha=alpha )
            if i in [0,3]:
                ax.text( wvlen - xoffset, Y[-1], Z[-1] + zoffset, 'λ=%g' % wvlen )

        A, B, C = absorbance.baseplane_params

        xmin, xmax = absorbance.reg_wvlens[[0, 3]]
        ymin, ymax = 0, absorbance.data.shape[1]-1

        """
        xx, yy = np.meshgrid( np.linspace(xmin, xmax, 10), np.linspace(ymin, ymax, 10) )
        zz = xx * A + yy * B + C
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='red' )
        """
        yy = np.array( [ymin, ymax] )

        for i, wv in enumerate( self.absorbance.wl_vector ):
            if wv < xmin:
                continue
            if wv > xmax:
                break

            if i % 2 > 0:
                continue

            xx = np.array( [ wv, wv ] )
            zz = xx * A + yy * B + C

            ax.plot( xx, yy, zz, color='red', alpha=0.2 )
