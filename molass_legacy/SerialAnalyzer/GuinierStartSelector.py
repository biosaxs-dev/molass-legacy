# coding: utf-8
"""

    ファイル名：   GuinierStartSelector.py

    処理内容：

        Guinier区間開始点の選択画面

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF

"""

import numpy                as np
from scipy                  import stats
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
from matplotlib.patches     import Polygon
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
from molass_legacy.KekLib.BasicUtils             import get_caller_module
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib          import NavigationToolbar

GUNIER_PLOT_SIZE    = 100
GUNIER_START_MAX    = 60
GUNIER_SIZE_EXCEPT  = 20

class GuinierStartSelector( Dialog ):
    def __init__( self, parent, serial_data, init_ranges ):

        self.created_parent = False

        self.grab = 'local'     # used in grab_set
        self.parent     = parent
        self.title_     = 'Guinier interval start point selection'
        self.serial_data    = serial_data
        self.init_ranges    = init_ranges
        self.num_peaks      = len(init_ranges)
        self.gs         = None
        self.applied    = None
        self.caller_module = get_caller_module( level=2 )

        self.peak_no    = self.parent.guinier_fig_peak_no.get()
        self.set_plot_data()

    def set_plot_data( self ):

        self.one_peak = self.init_ranges[self.peak_no][1][1]
        # print( self.init_ranges, one_peak )
        one_peak_data = self.serial_data.intensity_array[self.one_peak, :, : ]
        x_  = one_peak_data[0:GUNIER_PLOT_SIZE, 0]
        y_  = one_peak_data[0:GUNIER_PLOT_SIZE, 1]
        e_  = one_peak_data[0:GUNIER_PLOT_SIZE, 2]
        positive = np.logical_and( y_ > 0, e_ > 0 )

        self.x      = x_[positive]
        self.y      = y_[positive]
        self.x2     = self.x**2
        self.log_y  = np.log( y_[positive] )

    def show( self ):
        self.figsize    = ( 16, 8 )
        self.message    = None
        self.button_labels  = [ "OK", "Cancel" ]
        self.toolbar    = True

        Dialog.__init__( self, self.parent, self.title_ )
        # TODO: self.resizable(width=False, height=False)

    def destroy_parent( self ):
        if self.created_parent:
            self.parent.destroy()

    def body( self, body_frame ):   # overrides parent class method
        tk_set_icon_portable( self, module=self.caller_module )

        if self.message is not None:
            self.msg = Tk.Label( body_frame, text=self.message, bg='white' )
            self.msg.pack( fill=Tk.BOTH, expand=1, pady=20 )
            # msg.insert( Tk.INSERT, self.message )
            # msg.config( state=Tk.DISABLED )

        cframe = Tk.Frame( body_frame )
        cframe.pack()

        frame_ = Tk.Frame( body_frame )
        frame_.pack( fill=Tk.X, padx=200 )

        frame_left = Tk.Frame( frame_ )
        frame_left.pack( side=Tk.LEFT )

        label_ = Tk.Label( frame_left, text="Peak selection for Guinier plot: " )
        label_.pack( side=Tk.LEFT  )

        self.guinier_fig_peak_no = Tk.IntVar()
        self.guinier_fig_peak_no.set( self.parent.guinier_fig_peak_no.get() )
        entry_ = Tk.Spinbox( frame_left, textvariable=self.guinier_fig_peak_no,
                                            from_=0, to=self.num_peaks-1, increment=1, 
                                            justify=Tk.CENTER, width=6, state=Tk.NORMAL )
        entry_.pack( side=Tk.LEFT  )
        self.guinier_fig_peak_no.trace( "w", self.guinier_fig_peak_no_tracer )

        frame_right = Tk.Frame( frame_ )
        frame_right.pack( side=Tk.RIGHT )

        label_ = Tk.Label( frame_right, text="Guinier interval start point: " )
        label_.pack( side=Tk.LEFT  )

        self.guinier_start_point = Tk.IntVar()
        self.guinier_start_point.set( self.parent.guinier_start_point.get() )
        entry_ = Tk.Spinbox( frame_right, textvariable=self.guinier_start_point,
                                            from_=0, to=50, increment=1, 
                                            justify=Tk.CENTER, width=6, state=Tk.NORMAL )
        entry_.pack( side=Tk.LEFT  )
        self.guinier_start_point.trace( "w", self.guinier_start_point_tracer )

        figsize_ = ( 18, 9 ) if self.figsize is None else self.figsize

        fig = plt.figure( figsize=figsize_ )

        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas.draw()
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

        # it seems that draw_func should be called after the creation of mpl_canvas
        # in order to enable 3d-rotation of the figure
        self.draw_func( fig )
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

    def ok( self ):
        self.destroy_parent()
        self.applied  = True
        self.parent.guinier_start_point.set( self.guinier_start_point.get() )
        self.parent.guinier_fig_peak_no.set( self.guinier_fig_peak_no.get() )
        self.destroy()

    def cancel( self ):
        self.destroy_parent()
        self.applied  = False
        self.destroy()

    def draw_func( self, fig, clear=False, start=None ):
        if start is None:
            start   = self.guinier_start_point.get()

        if self.gs is None:
            self.fig = fig
            self.gs  = gridspec.GridSpec( 4, 2 )
            self.ax1 = fig.add_subplot( self.gs[0:3, 0] )
            self.ax2 = fig.add_subplot( self.gs[0:3, 1] )
            self.ax3 = fig.add_subplot( self.gs[3, 1] )
            self.ax4 = fig.add_subplot( self.gs[3, 0] )
            self.axes = [ self.ax1, self.ax2, self.ax3, self.ax4 ]

        ax1 = self.ax1
        ax2 = self.ax2
        ax3 = self.ax3
        ax4 = self.ax4

        if clear:
            for ax in self.axes:
                ax.cla()

        self.ax1.set_title( 'Guinier plot at elution No. %d' % self.one_peak )
        self.ax2.set_title( 'Zoomed plot of the left fig.' )
        self.ax3.set_title( 'Residuals in the Guinier interval' )
        self.ax4.set_title( 'Xray elution curve' )

        ax1.plot( self.x2, self.log_y, 'o', markersize=3 )
        ax2.plot( self.x2[0:GUNIER_START_MAX], self.log_y[0:GUNIER_START_MAX], 'o',  markersize=3 )

        for ax in self.axes[0:3]:
            ax.set_xlabel( 'Q²' )
            ax.set_ylabel( 'ln(I)' )

        ax4.set_xlabel( 'elution №' )
        ax4.set_ylabel( 'Intensity' )
        ivector = self.serial_data.ivector
        ax4.plot( ivector, color='orange' )
        ax4.plot( [ self.one_peak, self.one_peak ], [ 0, ivector[self.one_peak] ] )
        py = self.y[self.guinier_start_point.get()]
        ax4.plot( self.one_peak, py, 'o', color='red', markersize=5 )

        ax1.set_xlim( ax1.get_xlim() )
        ax1.set_ylim( ax1.get_ylim()  )

        xmin, xmax = ax2.get_xlim()
        ymin, ymax = ax2.get_ylim()

        ax2.set_xlim( xmin, xmax )
        ax2.set_ylim( ymin, ymax )
        xoffset = ( xmax + xmin ) * 0.05
        yoffset = ( ymax - ymin ) * 0.05
        for i in [ 10, 20, 30 ]:
            x = self.x2[i]
            ax2.plot( [x, x], [ymin, ymax], '-', color='gray', alpha=0.5 )
            ax2.annotate( '%d-th Q' % i, xy=(x, ymin), alpha=0.5,
                            xytext=( x + xoffset, ymin + yoffset ),
                            arrowprops=dict( headwidth=3, width=0.5, color='black', alpha=0.5 ),
                            )

        ax1.plot( [ xmin, xmax, xmax, xmin, xmin ], [ ymin, ymin, ymax, ymax, ymin ], ':', color='gray', alpha=0.5 )

        self.draw_elution_curve_area()
        self.draw_guinier_interval( start )

        fig.tight_layout()

    def guinier_start_point_tracer( self, *args ):
        try:
            start   = self.guinier_start_point.get()
        except:
            return

        self.draw_func( self.fig, clear=True, start=start )
        self.mpl_canvas.draw()

    def draw_elution_curve_area( self ):
        xmin, xmax = self.ax1.get_xlim()
        ymin, ymax = self.ax1.get_ylim()

        start   = self.serial_data.xray_slice.start
        stop    = self.serial_data.xray_slice.stop + 1  # include stop
        slice_  = slice( start, stop )
        x2_     = self.x2[slice_]
        log_y_  = self.log_y[slice_]
        verts = [ (x2_[0], ymin) ] + list(zip(x2_, log_y_)) + [(x2_[-1], ymin)]

        for ax in self.axes[0:2]:
            # you must give a Polygon separately for each axis
            poly = Polygon(verts, facecolor='orange', alpha=0.2 )
            ax.add_patch(poly)

        xoffset = ( xmax + xmin ) * 0.1

        tx  = np.average( x2_ )
        ty  = ymin*0.9 + ymax*0.1
        self.ax1.annotate( 'Range used for elution curve averaging',
                        xy=(tx, ty), alpha=0.5, va='center',
                        xytext=( tx + xoffset, ty ), 
                        arrowprops=dict( headwidth=3, width=0.5, color='black', alpha=0.5 ),
                        )

    def draw_guinier_interval( self, start ):
        stop = self.compute_guinier_stop( start )
        # print( 'qrg_limit=', qrg_limit )
        xr  = self.x2[start:stop]
        yr  = self.log_y[start:stop]
        slope, intercept, r_value, p_value, std_err = stats.linregress( xr, yr )

        x   = self.x2[ [start, stop] ]
        y   = slope*x + intercept
        for ax in [self.ax1, self.ax2]:
            ax.plot( x, y, color='red' )
            ax.plot( xr[0], yr[0], 'o', color='red', markersize=5 )

        self.ax3.set_xlim( self.ax2.get_xlim() )

        self.ax3.plot( x, [ 0, 0 ], color='red' )
        yr_ = yr - ( slope*xr + intercept )
        self.ax3.plot( xr, yr_ )
        self.ax3.plot( xr[0], yr_[0], 'o', color='red', markersize=5 )

    def compute_guinier_stop( self, start ):
        stop_range = slice( start+1, None )
        x2_ = self.x2[stop_range]
        log_y_ = self.log_y[stop_range]
        slope   =  (log_y_ - self.log_y[start]) /( x2_ - self.x2[start] )
        rg = np.sqrt( -3*slope )
        qrg = self.x[stop_range] * rg
        try:
            qrg_sizet = np.where( qrg < 1.3 )[0][-1]
        except:
            qrg_sizet = GUNIER_SIZE_EXCEPT
        return start + qrg_sizet + 1

    def guinier_fig_peak_no_tracer( self, *args ):
        try:
            self.peak_no    = self.guinier_fig_peak_no.get()
        except:
            return
        self.set_plot_data()
        self.draw_func( self.fig, clear=True )
        self.mpl_canvas.draw()
