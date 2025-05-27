# coding: utf-8
"""
    DecompViewer.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
import os
import copy
import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.patches   as mpl_patches      # 'as patches' does not work properly
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
from molass_legacy.KekLib.TkUtils                import is_low_resolution
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib          import NavigationToolbar
from molass_legacy.Models.ElutionCurveModels     import EGH, EGHA, EMG, EMGA
from molass_legacy.ElutionDecomposer         import ElutionDecomposer
from molass_legacy.KekLib.ExceptionTracebacker   import ExceptionTracebacker
from DualOptimzer import get_dual_optimized_info, EmgaOptimzer, EghaOptimzer
import matplotlib
from molass_legacy._MOLASS.SerialSettings         import get_setting

RETRY_FIT_ERROR_LIMIT   = 0.0001
ALT_ADOPT_RATIO         = 0.5
USE_DUAL_OPTIMIZER      = True

class DecompViewer( Dialog ):
    def __init__( self, parent, title, sd, mapper, peaks=None, flexible=True, manual=False ):
        self.parent = parent
        self.title_ = title
        self.sd     = sd
        self.mapper = mapper
        self.peaks  = peaks
        self.flexible   = flexible
        self.manual     = manual
        self.mplt_ge_2_2 = matplotlib.__version__ >= '2.2'

    def show( self ):
        Dialog.__init__( self, self.parent, self.title_ )

    def body( self, body_frame ):   # overrides parent class method
        tk_set_icon_portable( self )

        cframe = Tk.Frame( body_frame )
        cframe.pack()

        if self.flexible:
            egh_model   = EGHA()
            emg_model   = EMGA()
        else:
            egh_model   = EGH()
            emg_model   = EMG()

        if True:
            figsize = ( 15, 8 ) if is_low_resolution() else ( 18, 10 )
            fig = plt.figure( figsize=figsize )
            ax1_ = fig.add_subplot( 231 )
            ax2_ = fig.add_subplot( 232 )
            ax3_ = fig.add_subplot( 233 )
            ax4_ = fig.add_subplot( 234 )
            ax5_ = fig.add_subplot( 235 )
            ax6_ = fig.add_subplot( 236 )
            axes_list   = [ [ax1_, ax2_, ax3_], [ax4_, ax5_, ax6_] ]
            model_list  = [ egh_model, emg_model ]
            self.event_axes = [ ax2_, ax5_ ]
        else:
            figsize = ( 15, 5 ) if is_low_resolution() else ( 18, 5 )
            fig = plt.figure( figsize=figsize )
            ax1_ = fig.add_subplot( 131 )
            ax2_ = fig.add_subplot( 132 )
            ax3_ = fig.add_subplot( 133 )
            axes_list   = [ [ax1_, ax2_, ax3_] ]
            model_list  = [ None ]

        self.fig = fig
        self.mpl_canvas = FigureCanvasTkAgg( self.fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

        in_folder = get_setting( 'in_folder' )
        title = "Decomposition of Xray elution from " + in_folder
        fig.suptitle( title, fontsize=16 )

        self.draw( axes_list, model_list )
        # print( 'self.fa_list[1].init_fit_recs[0]=', self.fa_list[1].init_fit_recs[0] )

        fig.tight_layout()
        fig.subplots_adjust( top=0.92 )

        if self.mplt_ge_2_2:
            self.mpl_canvas.draw()
        else:
            self.mpl_canvas.show()
        self.toolbar = NavigationToolbar( self.mpl_canvas, cframe )
        self.toolbar.update()
        if self.manual:
            self.mpl_canvas.mpl_connect( 'button_press_event', self.manual_decomposer )

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

    def draw( self, axes_list, model_list ):
        mapper  = self.mapper
        sd      = self.sd

        x_curve = mapper.x_curve
        a_baseline = mapper.a_base + mapper.a_base_adjustment
        x_baseline = mapper.x_base + mapper.x_base_adjustment

        x_corrected = mapper.x_vector - x_baseline
        sd.apply_baseline_correction( mapper.get_mapped_info() )
        data    = sd.intensity_array

        self.y = y = x_corrected

        x   = np.arange( len(y) )

        if self.peaks is None:
            self.peaks = peaks = x_curve.get_emg_peaks()
        else:
            peaks   = self.peaks

        print( 'peaks=', peaks )

        self.chisqr_info_list = []
        self.params_info_list = []
        self.fa_list = []
        last_fa = None
        min_error = None

        for k, model in enumerate(model_list):

            ax1, ax2, ax3 = axes_list[k]
            try:
                fa  = ElutionDecomposer( x_curve, x, y, data, peaks=peaks, retry_valley=True, model=model, deeply=True )
            except:
                etb = ExceptionTracebacker()
                print( etb )
                fa  = last_fa

            fit_error = fa.compute_residual_error()
            print( [k], '------------------------ fit_error=', fit_error )
            if fit_error > RETRY_FIT_ERROR_LIMIT:
                # as of 2018-08-10, this is only for SUB_TRN1
                alt_model = EGHA()
                try:
                    fa_alt  = ElutionDecomposer( x_curve, x, y, data, peaks=peaks, retry_valley=True, model=alt_model, deeply=True )
                except:
                    etb = ExceptionTracebacker()
                    print( etb )
                    fa_alt  = last_fa
                fit_error_alt = fa_alt.compute_residual_error()
                print( [k], '------------------------ fit_error_alt=', fit_error_alt )
                if fit_error_alt < fit_error*ALT_ADOPT_RATIO:
                    fa          = fa_alt
                    fit_error   = fit_error_alt

            if min_error is None or fit_error < min_error:
                min_error = fit_error

            self.fa_list.append( fa )
            last_fa = fa
            name = fa.result_model_name

            ax3_title = None
            if USE_DUAL_OPTIMIZER:
                if name == 'EMGA':
                    optimizer_class = EmgaOptimzer
                elif name == 'EGHA':
                    optimizer_class = EghaOptimzer
                else:
                    optimizer_class = None
                if optimizer_class is None:
                    fit_recs_ax3 = fa.fit_recs
                else:
                    ax3_title = "Optimized simultaneously using %s Model" % name
                    opt_recs = get_dual_optimized_info(optimizer_class, x_curve, x, y, fa.fit_recs )
                    fit_recs_ax3 = opt_recs
            else:
                fit_recs_ax3 = fa.fit_recs

            if ax3_title is None:
                ax3_title = "Decomposition of all peaks using %s Model" % name

            ax1.set_title( "Corrected baseline with LPM and Adjustment" )
            ax2.set_title( "Decomposition of major peaks using %s Model" % name )
            ax3.set_title( ax3_title )

            # ax0.plot( mapper.a_vector )
            # ax0.plot( a_baseline, color='red' )
            ax1.plot( mapper.x_vector, color='orange' )
            ax1.plot( x_baseline, color='red' )
            ax2.plot( x_corrected, color='orange' )
            ax3.plot( x_corrected, color='orange' )

            def draw_decomposition( ax, fit_recs, x, y ):

                x_residual = copy.deepcopy( y )

                for rec in fit_recs:
                    func = rec[1]
                    ey = func( x )
                    ax.plot( ey, label='tau = %g' % func.dict_params['tau'] )
                    x_residual -= ey

                ax.plot( x_residual )
                ax.legend()

            draw_decomposition( ax2, fa.init_fit_recs, x, y )
            draw_decomposition( ax3, fit_recs_ax3, x, y )

            init_chisqrs = [ [ rec[0], rec[2] ]  for rec in fa.init_fit_recs ]
            opt_chisqrs  = [ [ rec[0], rec[2] ]  for rec in fa.fit_recs ]
            opt_params   = [ str( rec[1] ) for rec in fa.fit_recs ]
            self.chisqr_info_list.append( [ init_chisqrs, opt_chisqrs ] )
            self.params_info_list.append( opt_params )

            for ax in axes_list[k]:
                ymin, ymax = ax.get_ylim()
                for peak in peaks:
                    f, t = peak.get_fit_limits()
                    p = mpl_patches.Rectangle(
                            (f, ymin),      # (x,y)
                            t - f,          # width
                            ymax - ymin,    # height
                            facecolor   = 'cyan',
                            alpha       = 0.2,
                        )
                    ax.add_patch( p )

        self.min_error = min_error

    def get_min_error( self ):
        return self.min_error

    def get_chisqr_info_list( self ):
        return self.chisqr_info_list

    def get_params_info_list( self ):
        return self.params_info_list

    def manual_decomposer( self, event ):
        if not event.dblclick:
            return

        if event.inaxes not in self.event_axes or event.xdata is None or event.ydata is None:
            return

        if event.inaxes == self.event_axes[0]:
            i = 0
            model = EGH()
        else:
            i = 1
            model = EMG()

        print( 'manual_decomposer', i )

        from ManualDecomposer import ManualDecomposer

        title_parts = self.title_.split( '-' )

        mdecomp = ManualDecomposer( self, "Manual Decomposer -" + title_parts[-1],  self.y, self.fa_list[i].peak_info, model )
        mdecomp.show()

    def save_the_figure( self, folder, analysis_name ):
        # print( 'save_the_figure: ', folder, analysis_name )
        filename = analysis_name.replace( 'analysis', 'figure' )
        path = os.path.join( folder, filename )
        self.fig.savefig( path )
