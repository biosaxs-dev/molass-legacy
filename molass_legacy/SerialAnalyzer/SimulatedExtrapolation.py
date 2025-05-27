# coding: utf-8
"""

    ファイル名：   SimulatedExtrapolation.py

    処理内容：

        ゼロ濃度外挿シミュレーションのプロット

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF

"""
import sys
import os
import numpy    as np
import matplotlib.pyplot        as plt
from mpl_toolkits.mplot3d       import Axes3D
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter                 import Tk, Dialog
from molass_legacy.KekLib.BasicUtils                 import clear_dirs_with_retry, get_caller_module
from molass_legacy.KekLib.TkSupplements              import tk_set_icon_portable
from molass_legacy.KekLib.NumpyUtils                 import np_savetxt
from SphericalModel             import XrayScattering
import OurStatsModels           as sm
from SerialAtsasTools           import AlmergeExecutor

this_dir = os.path.dirname( os.path.abspath( __file__ ) )

SLICE_NUM_POINTS = 400

LOG_PLOT = False
LOG_PLOT_A = True

class SimulatedExtrapolation:
    def __init__( self ):
        self.work_dir = os.path.abspath( this_dir + '/../../work' )
        self.data_dir = self.work_dir + '/data'
        clear_dirs_with_retry( [ self.work_dir, self.data_dir ] )

        self.generate_data()

    def __del__( self ):
        clear_dirs_with_retry( [ self.work_dir ] )

    def generate_data( self ):
        self.x = x = np.linspace( 0.005, 0.8, 800 )
        C = 1e-2
        self.cvector_all = 0.1 * np.exp( - C * ( np.arange( 50 ) - 25 )**2  )
        self.indeces = np.arange( 0, 25 )

        intensity_list = []
        d_error_list = []
        filepaths = []
        for i in self.indeces:
            c = self.cvector_all[i]
            intensity = XrayScattering( c, 29 )
            if len(sys.argv) == 1:
                noise_level = 1/( 0.01 + c)
                ii = intensity( x ) * ( 1 + ( noise_level * 1e-4 + ( x - 0.05 ) **2 )* np.random.normal( 0, noise_level, len(x) ) )
                e = np.ones( len(x) ) * 1e-5 * noise_level
            else:
                noise_level = 1/c
                ii = intensity( x ) * ( 1 + ( x**2 )* np.random.normal( 0, noise_level, len(x) ) )
                e = np.ones( len(x) ) * 1e-6 * noise_level
            intensity_list.append( ii )
            d_error_list.append( e )
            filepath = self.data_dir + '/' +'SAMPLE-%04d.dat' % i
            data = np.array( [ x, ii, e ] ).T
            np_savetxt( filepath, data )
            filepaths.append( filepath )

        self.intensity_array = np.array( intensity_list )
        self.d_error_array = np.array( d_error_list )
        self.filepaths = filepaths
        # print( 'intensity_array.shape=', self.intensity_array.shape )
        # print( 'd_error_array.shape=', self.d_error_array.shape )

        self.x_slice = slice( 0, SLICE_NUM_POINTS )
        self.x_ = self.x[self.x_slice]

        self.x_i = 50
        self.x_f = self.x_[self.x_i]
        self.cvector_ = self.cvector_all[self.indeces]

        self.x1_ = self.cvector_
        self.x2_ = self.cvector_**2
        self.X = np.array( [ self.x1_, self.x2_] ).T

        param_list = []
        error_list = []
        for i in range(0,SLICE_NUM_POINTS):
            z_f = self.intensity_array[self.indeces,i]
            e2  = self.d_error_array[self.indeces,i]**2
            w   = 1/e2
            model   = sm.WLS(z_f, self.X, weights=w)
            result  = model.fit()
            param_list.append( result.params )
            # error_list.append( np.diag( result.cov_params() ) )
            error = np.sqrt( np.dot( model.XtWX_inv_XtW**2, e2 ) )
            error_list.append( error )

        self.params_array = np.array( param_list )
        self.error_array = np.array( error_list )

    def plot_concentration_elution( self, ax ):
        ax.set_title( 'Concentration Curve' )
        ax.set_xlabel( 'Elution No.' )
        ax.set_ylabel( 'Concentration' )
        ax.plot( self.indeces, self.cvector_ )

    def plot_xray_scattering( self, ax ):
        ax.set_title( 'Scattering Curves on Q[0:200] and the plane at Q[%d]=%.3g' % (self.x_i, self.x_f) )
        ax.set_xlabel( 'Q' )
        ax.set_ylabel( 'Elution No.' )
        zlabel_ = 'ln(Intensity)' if LOG_PLOT else 'Intensity'
        ax.set_zlabel( zlabel_ )

        x_ = self.x_

        for i in self.indeces:
            c = self.cvector_all[i]
            # y_ = np.ones( len(x_) ) * c
            y_ = np.ones( len(x_) ) * i
            if LOG_PLOT:
                z_ = np.log( self.intensity_array[i,self.x_slice] )
            else:
                z_ = self.intensity_array[i,self.x_slice]
            ax.plot( x_, y_, z_ )

        ymin, ymax = ax.get_ylim()
        ax.set_ylim( ymin, ymax )
        zmin, zmax = ax.get_zlim()
        ax.set_zlim( zmin, zmax )
        ax.plot( np.ones(5)*self.x_f,
                    [ymin, ymax, ymax, ymin, ymin ],
                    [zmin, zmin, zmax, zmax, zmin ],
                    ':', color='black' )
        if LOG_PLOT:
            z_i = np.log( self.intensity_array[self.indeces,self.x_i] )
        else:
            z_i = self.intensity_array[self.indeces,self.x_i]
        ax.plot( np.ones(len(self.cvector_))*self.x_f, self.indeces, z_i, ':', color='red' )

    def plot_xray_scattering_elution( self, ax ):
        ax.set_title( 'Scattering Variation with errorbar at Q[%d]=%.3g' % (self.x_i, self.x_f) )
        ax.set_xlabel( 'Seq No.' )
        ax.set_ylabel( 'Concentration' )
        y_f = self.intensity_array[self.indeces,self.x_i]
        y_e = self.d_error_array[self.indeces,self.x_i]
        ax.plot( self.indeces, y_f, ':', color='red' )
        ax.errorbar( self.indeces, y_f, yerr=y_e, color='blue', fmt='none' )

    def plot_regression( self, ax ):
        ax.set_title( 'Multivariate Regression at Q[%d]=%.3g' % (self.x_i, self.x_f) )
        ax.set_xlabel( 'C' )
        ax.set_ylabel( 'C²' )
        ax.set_zlabel( 'Scattering' )

        params = self.params_array[self.x_i,:]
        z_f = self.intensity_array[self.indeces,self.x_i]
        z_r = np.dot( self.X, params )

        ax.scatter( self.x1_, self.x2_, z_f )
        ax.scatter( self.x1_, self.x2_, z_r, color='orange' )

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim( xmin, xmax )
        ax.set_ylim( ymin, ymax )

        xx, yy = np.meshgrid( np.linspace(xmin, xmax, 10), np.linspace(ymin, ymax, 10) )
        zz = xx * params[0] + yy * params[1]
        ax.plot_surface(xx, yy, zz, alpha=0.1 )

    def plot_regression_params( self, ax_a, ax_b ):
        ax_a.set_title( 'Extrapolated A(q)' )
        ax_a.set_xlabel( 'Q' )
        ylabel_ = 'ln( Intensity / C )' if LOG_PLOT_A else 'Intensity / C'
        ax_a.set_ylabel( ylabel_ )

        ax_b.set_title( 'Extrapolated B(q)' )
        ax_b.set_xlabel( 'Q' )
        ax_b.set_ylabel( 'Intensity / C²' )

        A = self.params_array[:,0]
        B = self.params_array[:,1]
        A_ = np.log(A) if LOG_PLOT_A else A
        x_ = self.x_
        x_f = self.x_f
        x_i = self.x_i

        ax_a.plot( x_, A_, color='red', label='SA A(q)' )
        ax_b.plot( x_, B, color='red', label='SA B(q)' )
        ax_a.errorbar( x_f, A[x_i], yerr=self.error_array[x_i,0], color='blue', fmt='none' )
        ax_b.errorbar( x_f, B[x_i], yerr=self.error_array[x_i,1], color='blue', fmt='none' )

        c = 1
        intensity = XrayScattering( c, 29 )
        At = intensity.term1( x_ )
        Bt = -At*intensity.term2( x_ )

        ax_a.plot( x_, np.log(At), color='cyan', label='True A(q)' )
        ax_b.plot( x_, Bt, color='cyan', label='True B(q)' )

        almerge = AlmergeExecutor()
        exz_file = self.work_dir + '/exz_data.dat'
        almerge_result = almerge.execute( self.cvector_, self.filepaths, self.indeces, exz_file )
        max_c  = np.max( self.cvector_ )
        Aa = almerge_result.exz_array[self.x_slice,1] / max_c

        ax_a.plot( x_, np.log(Aa), color='gray', label='Almerge A(q)' )

        # np_savetxt( 'A.csv', A )
        # np_savetxt( 'B.csv', B )

        for ax in [ax_a, ax_b]:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim( ymin, ymax )
            ax.plot( [ x_f, x_f ], [ymin, ymax], ':', color='black' )

        for ax in [ax_a, ax_b]:
            ax.legend()

class SimulatedExtrapolationDialog( Dialog ):
    def __init__( self, title, parent=None ):
        self.grab = 'local'     # used in grab_set
        self.parent     = parent
        self.title_     = title
        self.applied    = None
        self.caller_module = get_caller_module( level=2 )

    def show( self, figsize=None, message=None ):
        self.figsize    = figsize
        self.message    = message
        self.parent.config( cursor='wait' )
        self.parent.update()

        Dialog.__init__( self, self.parent, self.title_ )
        # TODO: self.resizable(width=False, height=False)

    def body( self, body_frame ):   # overrides parent class method
        tk_set_icon_portable( self, module=self.caller_module )

        if self.message is not None:
            self.msg = Tk.Label( body_frame, text=self.message, bg='white' )
            self.msg.pack( fill=Tk.BOTH, expand=1, pady=20 )
            # msg.insert( Tk.INSERT, self.message )
            # msg.config( state=Tk.DISABLED )

        cframe = Tk.Frame( body_frame )
        cframe.pack()

        figsize_ = ( 20, 11 ) if self.figsize is None else self.figsize

        fig = plt.figure( figsize=figsize_ )

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
        se = SimulatedExtrapolation()

        ax1 = fig.add_subplot( 231 )
        ax2 = fig.add_subplot( 232, projection='3d' )
        ax3 = fig.add_subplot( 233 )
        ax4 = fig.add_subplot( 234, projection='3d' )
        ax5 = fig.add_subplot( 235 )
        ax6 = fig.add_subplot( 236 )

        se.plot_concentration_elution( ax1 )
        se.plot_xray_scattering( ax2 )
        se.plot_xray_scattering_elution( ax3 )
        se.plot_regression( ax4 )
        se.plot_regression_params( ax5, ax6 )

        fig.tight_layout()
