# coding: utf-8
"""
    DecompDemo.py.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
import copy
import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import matplotlib.patches   as mpl_patches      # 'as patches' does not work properly
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
from molass_legacy.KekLib.TkUtils                import is_low_resolution
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib          import NavigationToolbar, get_default_colors
from molass_legacy.Models.ElutionCurveModels     import EMG
from XrayDecomposer         import XrayDecomposer

class DecompDemo( Dialog ):
    def __init__( self, parent, title, sd, mapper, peaks=None ):
        self.parent = parent
        self.title_ = title
        self.sd     = sd
        self.mapper = mapper
        self.peaks  = peaks

    def show( self ):
        Dialog.__init__( self, self.parent, self.title_ )

    def body( self, body_frame ):   # overrides parent class method
        tk_set_icon_portable( self )

        cframe = Tk.Frame( body_frame )
        cframe.pack()

        emg_model = EMG()

        gs = gridspec.GridSpec( 4, 2 )
        figsize = ( 15, 8 ) if is_low_resolution() else ( 18, 10 )
        fig = plt.figure( figsize=figsize )
        ax2_ = fig.add_subplot( gs[0:3,0] )
        ax3_ = fig.add_subplot( gs[0:3,1] )
        ax5_ = fig.add_subplot( gs[3,0] )
        ax6_ = fig.add_subplot( gs[3,1] )
        axes_list   = [ [ax2_, ax3_], [ax5_, ax6_] ]
        name_list   = [ "EGH" ]
        model_list  = [ None ]

        self.fig = fig
        self.mpl_canvas = FigureCanvasTkAgg( self.fig, cframe )
        self.mpl_canvas.draw()
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

        self.draw( axes_list, name_list, model_list )

        self.fig.tight_layout()
        self.mpl_canvas.draw()

        self.toolbar = NavigationToolbar( self.mpl_canvas, cframe )
        self.toolbar.update()

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

    def draw( self, axes_list, name_list, model_list ):
        mapper  = self.mapper
        sd      = self.sd

        x_curve = mapper.x_curve
        a_baseline = mapper.a_base + mapper.a_base_adjustment
        x_baseline = mapper.x_base + mapper.x_base_adjustment

        x_corrected = mapper.x_vector - x_baseline
        sd.apply_baseline_correction( mapper.get_mapped_info() )
        data    = sd.intensity_array

        y   = x_corrected
        x   = np.arange( len(y) )

        if self.peaks is None:
            self.peaks = peaks = x_curve.get_emg_peaks()
        else:
            peaks   = self.peaks

        print( 'peaks=', peaks )

        self.chisqr_info_list = []

        k = 0
        ax2, ax3 = axes_list[k]
        ax5, ax6 = axes_list[1]
        name = name_list[k]
        fa1  = XrayDecomposer( x_curve, x, y, data, peaks=None, retry_valley=True, model=None, deeply=True )
        fa2  = XrayDecomposer( x_curve, x, y, data, peaks=peaks, retry_valley=True, model=None, deeply=True )

        ax2.set_title( "Decomposition as two peaks", fontsize=16 )
        ax3.set_title( "Decomposition as three peaks", fontsize=16 )

        ax5.set_title( "Residuals", fontsize=16 )
        ax6.set_title( "Residuals", fontsize=16 )

        ax2.plot( x_corrected, color='orange' )
        ax3.plot( x_corrected, color='orange' )

        default_colors = get_default_colors()

        def draw_decomposition( ax, ax_, fit_recs, colors ):

            x_residual = copy.deepcopy( x_corrected )

            for i, rec in enumerate( fit_recs ):
                func    = rec[1]
                ey = func( x )
                ax.plot( ey, linewidth=5.0, color=colors[i] )
                x_residual -= ey

            ax_.plot( x_residual, color='gray' )

        draw_decomposition( ax2, ax5, fa1.init_fit_recs, [ default_colors[0], default_colors[1] ] )
        draw_decomposition( ax3, ax6, fa2.fit_recs, [ default_colors[0], default_colors[2], default_colors[1] ] )

        ax6.set_ylim( ax5.get_ylim() )

        for ax in [ ax2, ax3, ax5, ax6 ]:
            ax.tick_params( labelsize=16 )

        if False:

            for ax in axes_list[k]:
                ymin, ymax = ax.get_ylim()
                for peak in peaks:
                    f = peak.flimL
                    t = peak.flimR
                    p = mpl_patches.Rectangle(
                            (f, ymin),      # (x,y)
                            t - f,          # width
                            ymax - ymin,    # height
                            facecolor   = 'cyan',
                            alpha       = 0.2,
                        )
                    ax.add_patch( p )
