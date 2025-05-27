# coding: utf-8
"""
    ManualDecomposer.py.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
import copy
import numpy                as np
import logging
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import matplotlib.patches   as mpl_patches      # 'as patches' does not work properly
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
from molass_legacy.KekLib.TkUtils                import is_low_resolution
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib          import NavigationToolbar
from molass_legacy.Models.ElutionCurveModels     import EMG

param_names = [ 'h', 'mu', 'sigma', 'tau' ]


class ParamControlPanel:
    def __init__( self, parent, order, pframe, name, init_params ):
        self.parent = parent
        self.order  = order

        frame = Tk.Frame( pframe )
        frame.pack()

        self.init_params = init_params
        self.params = copy.deepcopy( init_params )

        self.scale_list = []

        for k, name in enumerate(param_names):
            init_value = init_params[ name ].value
            from_v  = max( 0.01, init_value * 0.1 )
            to_v    = init_value * 2.0
            if k > 0:
                to_v    = max( 10, to_v )
            resolution = ( to_v - from_v ) / 100

            label = Tk.Label( frame, text=name )
            label.grid( row=k, column=0 )
            var = Tk.DoubleVar()
            scale = Tk.Scale( frame, from_=from_v, to=to_v, resolution=resolution, sliderlength=10, length=200, orient=Tk.HORIZONTAL,
                    command=lambda v, name_=name : self.scale_tracer( v, name_ ) )
            self.scale_list.append( scale )
            scale.grid( row=k, column=1 )
            scale.set( init_value )
            # var.trace( 'w', lambda *args, k_=k, scale_=scale : self.scale_tracer( k_, scale_ ) )

    def scale_tracer( self, v, name ):
        print( 'v=', v, 'name=', name )
        self.params[name].value = float( v )
        self.parent.draw()

    def reset( self ):
        self.params = copy.deepcopy( self.init_params )

class ManualDecomposer( Dialog ):
    def __init__( self, parent, title, y, peak_info, model ):
        self.logger  = logging.getLogger( __name__ )
        self.parent = parent
        self.title_ = title
        self.y  = y
        self.x  = np.arange( len(y) )
        self.peak_info = peak_info
        self.model  = model

    def show( self ):
        Dialog.__init__( self, self.parent, self.title_ )

    def body( self, body_frame ):   # overrides parent class method
        tk_set_icon_portable( self )

        cframe = Tk.Frame( body_frame )
        cframe.pack( side=Tk.LEFT )
        ctlframe = Tk.Frame( body_frame )
        ctlframe.pack( side=Tk.LEFT  )

        emg_model = EMG()

        gs = gridspec.GridSpec( 4, 1 )
        figsize = ( 10, 8 ) if is_low_resolution() else ( 12, 10 )
        fig = plt.figure( figsize=figsize )
        self.ax1 = fig.add_subplot( gs[0:3, 0] )
        self.ax2 = fig.add_subplot( gs[3, 0] )

        self.fig = fig
        self.mpl_canvas = FigureCanvasTkAgg( self.fig, cframe )
        self.mpl_canvas.draw()
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

        num_panels = 2

        fit_params = self.get_params()
        h   = fit_params['h']
        fit_params['h'].value = h/num_panels
        self.panels = []
        for k in range(num_panels):
            fit_params_ = copy.deepcopy( fit_params )
            panel = ParamControlPanel( self, k, ctlframe, 'Peak%d' % k, fit_params_ )
            self.panels.append( panel )

        self.draw()

        self.fig.tight_layout()
        self.mpl_canvas.draw()

        self.toolbar = NavigationToolbar( self.mpl_canvas, cframe )
        self.toolbar.update()
        # self.mpl_canvas.mpl_connect( 'button_press_event', self.manual_decomposer )

        self.reset_btn  = Tk.Button( ctlframe, text="Reset", command=self.reset )
        self.reset_btn.pack()

    def buttonbox(self):
        '''add standard button box.

        override if you do not want the standard buttons
        '''

        box = Tk.Frame(self)

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.ok_button = w
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.cancel_button = w

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def get_params( self ):
        info = self.peak_info[0]
        start   = info[0]
        head    = int(info[1]+0.5)
        stop    = info[2] + 1
        slice_  = slice( start, stop )
        y_  = self.y[slice_]
        x_  = self.x[slice_]
        params  = self.model.guess(y_, x=x_)
        try:
            out = self.model.fit(y_, params, x=x_)
        except:
            # EMG model simetimes raises ValueError("The input contains nan values")
            self.logger.warning( "%s.fit default method failed. resorting to 'least_squares'." % ( model.name ) )
            out = model.fit(y_, params, x=x_, method='least_squares')

        print( out.params )
        return out.params

    def draw( self ):

        self.ax1.cla()
        self.ax2.cla()

        self.ax1.plot( self.y, color='orange' )

        y_ = copy.deepcopy( self.y )

        my = np.zeros( len(y_) )

        for panel in self.panels:

            ey = self.model.eval( panel.params, x=self.x )
            self.ax1.plot( ey )
            y_ -= ey
            my += ey

        self.ax1.plot( my, ':', color='red' )

        self.ax2.plot( y_, color='gray' )

        ymin, ymax = self.ax1.get_ylim()
        for info in self.peak_info:
            f = info[0]
            t = info[2]
            p = mpl_patches.Rectangle(
                    (f, ymin),      # (x,y)
                    t - f,          # width
                    ymax - ymin,    # height
                    facecolor   = 'cyan',
                    alpha       = 0.2,
                )
            self.ax1.add_patch( p )

        self.mpl_canvas.draw()

    def reset( self ):
        for panel in self.panels:
            panel.reset()
        self.draw()
