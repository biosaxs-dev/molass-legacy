# coding: utf-8
"""
    DriftLinearModel..py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
import copy
import numpy as np
from mpl_toolkits.mplot3d   import Axes3D
from matplotlib.collections import PolyCollection
from LmfitThreadSafe        import minimize, Parameters
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
from CanvasFrame            import CanvasFrame
from molass_legacy.KekLib.TkUtils                import is_low_resolution
from DataModels             import _GuinierPorod

USE_GUNIER_POROD_MODEL  = True
if USE_GUNIER_POROD_MODEL:
    from DataModels         import _GuinierPorod

TO_USE_QVECTOR_RATIO    = 0.3

LPM_BASE_SHIFT  = 0

class DriftLinearModel:
    def __init__( self, qvector, num_elutions ):
        qstop   = max( 250, int( len(qvector)*TO_USE_QVECTOR_RATIO ) )
        self.qvec   = qvector[0:qstop]
        self.num_elutions = num_elutions
        self.xx     = self.qvec**2
        self.jvec   = np.arange( num_elutions )
        self.fitted_indeces     = None
        self.fitted_baselines   = None

    def simulate( self, params ):
        self.params = params
        Rg      = params[ 'Rg' ]
        Gstart  = params[ 'Gstart' ]
        Gstop   = params[ 'Gstop' ]
        base_pos    = params[ 'base_pos' ]

        A   = Gstart
        B   = ( Gstop - Gstart ) / self.jvec[-1]
        Gb  = A + B*base_pos

        if USE_GUNIER_POROD_MODEL:
            d   = params[ 'd' ]
            Zb  = _GuinierPorod( Gb, Rg, d, self.qvec )
        else:
            z_  = np.exp( -Rg**2 * self.xx/3 )
            Zb  = Gb * z_

        z_list = []
        for j in self.jvec:
            Gj  = A + B*j
            if USE_GUNIER_POROD_MODEL:
                z   = _GuinierPorod( Gj, Rg, d, self.qvec )
            else:
                z   = Gj * z_
            z_list.append( z)

        return Zb, np.array( z_list )

    def guinier_porod_fit( self, q_, y_, sign, particle=None ):

        params = Parameters()
        if particle is None:
            G_init = 1.0
            Gb_init = G_init * ( 10 + sign*2)/10
            params.add( 'Gb',   value=G_init,   min=1e-6,   max=10 )
            params.add( 'G',    value=Gb_init,  min=1e-6,   max=10 )
            params.add( 'Rg',   value=100,  min=30,     max=1000 )
            params.add( 'd',    value=2.5,  min=1,      max=4 )
        else:
            Gb_init  = particle['Gb']
            params.add( 'Gb',   value=Gb_init, vary=False )
            # params.add( 'Gb',   value=Gb_init, min=Gb_init*0.1, max=Gb_init*10)
            G_init  = particle['G']
            params.add( 'G',    value=G_init, min=G_init*0.1, max=G_init*10 )
            params.add( 'Rg',   value=particle['Rg'], vary=False )
            params.add( 'd',    value=particle['d'], vary=False )

        def objective_func( params ):
            Gb  = params['Gb']
            G   = params['G']
            Rg  = params['Rg']
            d   = params['d']
            yb  = _GuinierPorod( Gb, Rg, d, q_ )
            y   = _GuinierPorod( G - LPM_BASE_SHIFT, Rg, d, q_ )
            return ( y - yb ) - y_

        result  = minimize( objective_func, params, args=() )
        ret_params = {}
        for key in [ 'Gb', 'G', 'Rg', 'd']:
            ret_params[key] = result.params[key].value

        if False:
            from CanvasDialog   import CanvasDialog
            Gb  = ret_params['Gb']
            G   = ret_params['G']
            Rg  = ret_params['Rg']
            d   = ret_params['d']
            q1  = 1/Rg * np.sqrt( 3*d/2 )
            # print( 'sign=', sign, 'q1=', q1 )

            def plot_func( fig ):
                ax1 = fig.add_subplot( 121 )
                ax2 = fig.add_subplot( 122 )
                yb  = _GuinierPorod( Gb, Rg, d, q_ )
                ax1.plot( q_, yb, label='yb' )
                y   = _GuinierPorod( G, Rg, d, q_ )
                ax1.plot( q_, y, label='y' )
                ax2.plot( q_, y_, label='y_' )
                ax2.plot( q_, yb - y, label='yb - y' )
                for ax in [ax1, ax2]:
                    ymin, ymax = ax.get_ylim()
                    ax.set_ylim( ymin, ymax )
                    ax.plot( [q1, q1], [ymin, ymax], ':', color='black' )
                    ax.legend()
                fig.tight_layout()

            dialog = CanvasDialog( "Debug: q_, y_", adjust_geometry=True )
            dialog.show( plot_func, figsize=(12, 6), toolbar=True )

        return ret_params

    def compute_initial_values( self, q_, bl_array ):
        curve1  = bl_array[:,0]
        curve2  = bl_array[:,-1]
        min0    = min( curve1[0], curve2[0] )
        max0    = max( curve1[0], curve2[0] )
        sign    = 0
        if min0 > 0:
            sign    = -1
        if max0 < 0:
            sign    = +1
        params1 = self.guinier_porod_fit( q_, curve1, sign )
        params2 = self.guinier_porod_fit( q_, curve2, sign )

        print( 'params1=', params1 )
        print( 'params2=', params2 )
        Gb  = np.sqrt( params1['Gb'] * params2['Gb'] )
        Rg  = np.sqrt( params1['Rg'] * params2['Rg'] )
        d   = np.sqrt( params1['d'] * params2['d'] )

        particle = {}
        particle['Gb'] = Gb
        particle['G'] = params1['G']
        particle['Rg'] = Rg
        particle['d'] = d

        params1 = self.guinier_porod_fit( q_, curve1, sign, particle=particle )

        particle['G'] = params2['G']
        params2 = self.guinier_porod_fit( q_, curve2, sign, particle=particle )
        print( 'params1=', params1 )
        print( 'params2=', params2 )
        G1  = params1['G']
        G2  = params2['G']

        return Gb, G1, G2, Rg, d

    def fit( self, indeces, baselines ):
        self.fitted_indeces     = indeces
        self.fitted_baselines   = baselines

        q_  = self.qvec[indeces]

        Gb_init, G1_init, G2_init, Rg_init, d_init = self.compute_initial_values( q_, np.array( baselines ) )

        qq_ = q_**2

        params = Parameters()

        if True:
            Rg_init     = Rg_init
            Gstart_init = G1_init
            Gstop_init  = G2_init
            slope = (G2_init - G1_init)/(self.num_elutions - 1)
            base_pos_init   = ( Gb_init - G1_init ) / slope
            # params.add( 'Rg',   value=Rg_init, vary=False )
            params.add( 'Rg',       value=Rg_init,      min=Rg_init*0.5,    max=Rg_init*5 )
        else:
            Rg_init     = 250
            Gstart_init = 1.0
            Gstop_init  = 0.5
            base_pos_init   = -50
            params.add( 'Rg',       value=Rg_init,      min=Rg_init*0.5,    max=Rg_init*5 )
        params.add( 'Gstart',   value=Gstart_init,  min=0.2,              max=Gstart_init*2 )
        params.add( 'Gstop',    value=Gstop_init,   min=0.2,              max=Gstop_init*2 )
        params.add( 'base_pos',     value=base_pos_init,    min=-len(self.jvec)*0.5,    max=len(self.jvec)*1.5 )

        if USE_GUNIER_POROD_MODEL:

            params.add( 'd',    value=d_init,      min=1.0,    max=4.0 )
            w_   = self.jvec / self.jvec[-1]

            def objective_func( params ):
                Rg      = params['Rg']
                Gstart  = params['Gstart']
                Gstop   = params['Gstop']
                base_pos    = params['base_pos']
                d       = params['d']

                Zs  = _GuinierPorod( Gstart - LPM_BASE_SHIFT, Rg, d, q_ )
                Zp  = _GuinierPorod( Gstop - LPM_BASE_SHIFT,  Rg, d, q_ )

                A   = Gstart
                B   = ( Gstop - Gstart ) / self.jvec[-1]
                Gb  = A + B*base_pos
                Zb  = _GuinierPorod( Gb, Rg, d, q_ )

                diff    = np.zeros( len(self.jvec) )
                for i, baseline in enumerate( baselines ):
                    diff +=  np.abs( ( Zs[i]*(1 - w_) + Zp[i]*w_ - Zb[i] ) - baseline )

                return diff

        else:
            def objective_func( params ):
                Rg      = params['Rg']
                Gstart  = params['Gstart']
                Gstop   = params['Gstop']
                base_pos    = params['base_pos']

                z_  = np.exp( -Rg**2 * qq_/3 )

                A   = Gstart
                B   = ( Gstop - Gstart ) / self.jvec[-1]
                Gb  = A + B*base_pos
                Zb  = Gb * z_

                diff    = np.zeros( len(self.jvec) )
                for i, baseline in enumerate( baselines ):
                    diff +=  np.abs( ( z_[i] * (B * self.jvec + A) - Zb[i] ) - baseline )

                return diff

        result  = minimize( objective_func, params, args=() )

        self.params = {}
        for key in [ 'Rg', 'Gstart', 'Gstop', 'base_pos' ]:
            self.params[key] = result.params[key].value

        self.params['slope'] = ( self.params['Gstop'] - self.params['Gstart'] ) / self.jvec[-1]
        self.params['intercept'] = self.params['Gstart']
        self.params['Gbg'] = self.params['intercept'] + self.params['slope'] * self.params['base_pos' ]

        if USE_GUNIER_POROD_MODEL:
            self.params['d'] = result.params['d'].value

    def draw( self, axes, Zb, data ):
        ax1, ax2 = axes

        ax1.set_title( "Before Background Subtraction" )
        ax2.set_title( "After Background Subtraction" )
        for ax in axes:
            ax.set_xlabel( 'Q' )
            ax.set_ylabel( 'Elution' )
            ax.set_zlabel( 'Intensity' )

        sub_list = []   
        for curve in data:
            sub_list.append( curve - Zb )

        base_pos = self.params[ 'base_pos' ]
        y   = np.ones( len(self.qvec) ) * base_pos

        ax1.plot( self.qvec, y, Zb, color='green', label='Background' )
        ax2.plot( self.qvec, y, np.zeros( len(self.qvec) ), color='green', label='Background' )

        color   = 'red'
        sub_data    = np.array( sub_list )

        def is_to_make_polygon( k ):
            return k % 10 == 0

        poly_alpha = 0.3

        verts1 = []
        verts2 = []
        zs_list = []

        for i, q in enumerate( self.qvec ):
            x   = np.ones(len(self.jvec)) * q
            z1  = data[:,i]
            z2  = sub_data[:,i]

            if is_to_make_polygon( i ):
                zs_list.append( q )
                zb_val  = Zb[i]

                if base_pos < 0 or base_pos >= len(self.jvec):
                    ax1.plot( [q, q], [base_pos, 0], [zb_val, zb_val], ':', color='green', alpha=poly_alpha )
                    ax2.plot( [q, q], [base_pos, 0], [0, 0], ':', color='green', alpha=poly_alpha )

                je  = self.jvec[-1]
                zb  = np.ones( len(self.jvec) ) * zb_val
                ax1.plot( x, self.jvec, zb, color='green', alpha=poly_alpha )
                yp  = [ 0, 0, je, je ]
                zp  = [ zb_val, z1[0], z1[-1], zb_val ]
                verts1.append( list( zip( yp, zp ) ) )

                zb = np.zeros( len(self.jvec) )
                ax2.plot( x, self.jvec, zb, color='green', alpha=poly_alpha )
                zp  = [ 0, z2[0], z2[-1], 0 ]
                verts2.append( list( zip( yp, zp ) ) )

            if i == 0:
                label = 'Stain Intensity'
            else:
                label = None

            ax1.plot( x, self.jvec, z1, color=color, alpha=0.2, label=label )
            ax2.plot( x, self.jvec, z2, color=color, alpha=0.2, label=label )

        if self.fitted_indeces is not None:
            for k, i in enumerate(self.fitted_indeces):
                q   = self.qvec[i]
                x   = np.ones(len(self.jvec)) * q
                z   = self.fitted_baselines[k]
                ax2.plot( x, self.jvec, z, color='yellow', alpha=0.2 )

        poly1 = PolyCollection(verts1)
        poly1.set_alpha(0.2)
        ax1.add_collection3d(poly1, zs=zs_list, zdir='x')

        poly2 = PolyCollection(verts2)
        poly2.set_alpha(0.2)
        ax2.add_collection3d(poly2, zs=zs_list, zdir='x')

        if USE_GUNIER_POROD_MODEL:
            # TODO: plot q1-plane
            pass

        if False:
            zmin1, zmax1 = ax1.get_zlim()
            zmin2, zmax2 = ax2.get_zlim()

            zmin_ = min( zmin1, zmin2 )
            zmax_ = max( zmax1, zmax2 )

            ax1.set_zlim( zmin_, zmax_ )
            ax2.set_zlim( zmin_, zmax_ )

        ax1.legend()
        ax2.legend()

class DriftLinearModelDialog( Dialog ):
    def __init__( self, parent, model, init_params ):
        self.parent = parent
        self.model  = model
        self.init_params = init_params

    def show( self ):
        title = "Linear Drift Simulator"
        Dialog.__init__(self, self.parent, title )

    def body( self, body_frame ):
        # tk_set_icon_portable( self )

        figsize = ( 12, 10 ) if is_low_resolution() else ( 14, 12 )
        self.canvas_frame = canvas_frame = CanvasFrame( body_frame, figsize=figsize )
        canvas_frame.pack()
        fig = self.canvas_frame.fig
        ax1 = fig.add_subplot( 221, projection='3d' )
        ax2 = fig.add_subplot( 222, projection='3d' )
        ax3 = fig.add_subplot( 223, projection='3d' )
        ax4 = fig.add_subplot( 224, projection='3d' )

        fig.tight_layout()

        Zb1, data1  = self.model.simulate( self.init_params )
        self.model.draw( [ ax1, ax2 ], Zb1, data1 )

        copy_params = copy.deepcopy( self.init_params )
        copy_params['Gstart'] += 2.0
        copy_params['Gstop'] += 2.0

        Zb2, data2  = self.model.simulate( copy_params )
        self.model.draw( [ ax3, ax4 ], Zb2, data2 )

        ax1.set_zlim( ax3.get_zlim() )

        self.canvas_frame.show()
