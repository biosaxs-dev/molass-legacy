# coding: utf-8
"""
    SvdTutorial.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
import numpy                as np
import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d   import Axes3D
from matplotlib.patches     import Circle, Ellipse
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
from molass_legacy.KekLib.TkUtils                import is_low_resolution
from CanvasFrame            import CanvasFrame
from OurMatplotlib3D        import pathpatch_2d_to_3d, pathpatch_translate

DEBUG   = False

class SvdTutorial( Dialog ):
    def __init__( self, parent, array, indeces ):
        self.parent = parent
        self.array  = array
        print( self.array.shape )
        self.Qindex = indeces[0]
        self.Eindex = indeces[1]
        self.M  = array[self.Eindex,:,1][:,self.Qindex].T
        print( self.M.shape )
        U, s, VT = np.linalg.svd( self.M )
        print( U )
        print( s )
        print( VT )
        self.U  = U
        self.s  = s
        self.VT = VT

    def show( self ):
        Dialog.__init__( self, self.parent, "SVD Tutorial Demo" )

    def body( self, body_frame ):   # overrides parent class method

        base_frame = Tk.Frame( body_frame );
        base_frame.pack( expand=1, fill=Tk.BOTH )

        figsize = ( 12, 11 ) if is_low_resolution() else ( 12, 11 )
        self.canvas_frame = canvas_frame = CanvasFrame( base_frame, figsize=figsize )
        canvas_frame.pack( side=Tk.LEFT )

        panel_frame = Tk.Frame( base_frame )
        panel_frame.pack( side=Tk.LEFT, anchor=Tk.N )

        fig = canvas_frame.fig

        self.axes = []
        for i in range(2):
            axes_row = []
            for j in range(2):
                projection = '3d' if j==1 else None
                ax = fig.add_subplot( 220 + i*2 + (j+1), projection=projection )
                axes_row.append( ax )
            self.axes.append( axes_row )

        fig.tight_layout()

        self.elution_canvas = CanvasFrame( panel_frame, figsize=( 6, 8.25 ) )
        self.elution_canvas.pack()

        fig = self.elution_canvas.fig

        self.p_ax0 = fig.add_subplot( 311 )
        self.p_ax1 = fig.add_subplot( 312 )
        self.p_ax2 = fig.add_subplot( 313 )

        fig.tight_layout()

        self.draw_panel()
        self.draw_svd()

    def draw_panel( self ):
        self.p_ax0.set_title( "210-d Elution Vectors" )
        self.p_ax1.set_title( "210-d Elution Vector at Q[%d]" % self.Qindex[1] )
        self.p_ax2.set_title( "1176-d Scattering Vectors" )

        e_size = self.array.shape[0]

        y66 = np.zeros( e_size )
        y66[66] = 1

        y141 = np.zeros( e_size )
        y141[141] = 1

        ym = np.zeros( e_size )
        ym[66] = 0.5
        ym[141] = 0.5

        for y in [ y66, y141, ym ]:
            self.p_ax0.plot( y )

        for ax in [ self.p_ax0, self.p_ax1 ]:
            ax.plot( self.array[ :, self.Qindex[1], 1 ], color='orange' )

        ax = self.p_ax1
        ymin, ymax = ax.get_ylim()
        ax.set_ylim( ymin, ymax )
        for j in self.Eindex:
            ax.plot( [ j, j ], [ ymin, ymax ], ':', color='red' )

        quarter = self.array.shape[1]//4
        my66    = self.array[ self.Eindex[0], 0:quarter, 1 ]
        my141   = self.array[ self.Eindex[1], 0:quarter, 1 ]
        self.p_ax2.plot( my66 )
        self.p_ax2.plot( my141 )
        self.p_ax2.plot( (my66 + my141)/2 )

        ymin, ymax = self.p_ax2.get_ylim()
        self.p_ax2.set_ylim( ymin, ymax )
        for k in self.Qindex:
            self.p_ax2.plot( [ k, k ], [ ymin, ymax ], ':', color='red' )

    def draw_svd( self ):
        axE = self.axes[0][0]
        axS = self.axes[0][1]
        axF = self.axes[1][0]
        axG = self.axes[1][1]

        # Elution Curve Space
        axE.set_title( "Elution Curve Space", y=0.95 )
        axE.set_xlabel( "E%d" % self.Eindex[0])
        axE.set_ylabel( "E%d" % self.Eindex[1] )

        Epoints = []
        labels = []

        label = "E%d" % self.Eindex[0]
        axE.plot( 1, 0, 'o', label=label )
        Epoints.append( [1,0] )
        labels.append( label )

        label = "E%d" % self.Eindex[1]
        axE.plot( 0, 1, 'o', label=label )
        Epoints.append( [0,1] )
        labels.append( label )

        label = "Emiddle"
        axE.plot( 0.5, 0.5, 'o', label=label )
        Epoints.append( [0.5,0.5] )
        labels.append( label )

        scale = 1

        for i, q in enumerate(self.Qindex):
            x   = self.M[i,0]*scale
            y   = self.M[i,1]*scale
            label = "Q%d" % q 
            axE.plot( x, y, 'o', label=label)
            Epoints.append( [x,y] )
            labels.append( label )

        circle = Circle((0, 0), 1.0, alpha=0.1)
        axE.add_patch( circle )

        axE.plot( [1,0], [0,1], ':', color='black' )
        axE.plot( 0, 0, 'o', color='red', label='zero' )

        axE.legend()

        # Scattering Curve Space
        axS.set_title( "Scattering Curve Space", y=1.03 )
        axS.set_xlabel( "Q%d" % self.Qindex[0])
        axS.set_ylabel( "Q%d" % self.Qindex[1] )
        axS.set_zlabel( "Q%d" % self.Qindex[2] )

        Spoints = []
        for k, p in enumerate(Epoints):
            S   = np.dot( self.M, np.array(p).reshape( (2,1) )*scale ).reshape( (3,) )
            axS.plot( [ S[0] ], [ S[1] ], [ S[2] ], 'o', label=labels[k] )
            Spoints.append( [ S[0], S[1], S[2] ] )

        Spoints_ = np.array( Spoints )
        axS.plot( Spoints_[0:2,0], Spoints_[0:2,1], Spoints_[0:2,2], ':', color='black' )
        axS.plot( [0], [0], [0], 'o', color='red', label='zero' )

        p = Ellipse((0,0), self.s[0]*2, self.s[1]*2, alpha=0.1 )    # Add an ellipse in the xy plane
        axS.add_patch(p)

        normal = np.dot( self.U, [0, 0, -11] ).reshape( (3,) )
        print( 'normal=', normal )
        pathpatch_2d_to_3d(p, z = 0, normal=normal)

        axS.legend()

        # Roteted Elution Curve Space
        axF.set_title( "Roteted Elution Curve Space", y=0.95 )
        Fpoints = []
        for k, p in enumerate(Epoints):
            F   = np.dot( self.VT, np.array(p).reshape( (2,1) )*scale )
            axF.plot( F[0,0], F[1,0], 'o', label=labels[k] )
            Fpoints.append( [ F[0,0], F[1,0] ] )

        Fpoints_ = np.array( Fpoints )

        circle = Circle((0, 0), 1.0, alpha=0.1)
        axF.add_patch( circle )
        axF.plot( Fpoints_[0:2,0], Fpoints_[0:2,1], ':', color='black' )
        axF.plot( 0, 0, 'o', color='red', label='zero' )

        axF.legend()

        # Scaled Elution Curve Space embedded into Scattering Curve Space
        axG.set_title( "Scaled R-EC Space embedded into SC Space", y=1.03 )

        p = Ellipse((0,0), self.s[0]*2, self.s[1]*2, alpha=0.1 )    # Add an ellipse in the xy plane
        axG.add_patch(p)
        pathpatch_2d_to_3d(p, z = 0, normal = 'z')

        Gpoints = []
        for k, p in enumerate(Fpoints):
            p_  = self.s * p
            axG.plot( [ p_[0] ], [ p_[1] ], [0], 'o', label=labels[k] )
            Gpoints.append( p_ )

        Gpoints_ = np.array( Gpoints )

        axG.plot( Gpoints_[0:2,0], Gpoints_[0:2,1], [0], ':', color='black' )
        axG.plot( [0], [0], [0], 'o', color='red', label='zero' )

        axG.legend()

        if False:
            # check whether sel.U transforms Gpoints[1] to the right position
            p   = Gpoints[1]
            Sd  = np.dot( self.U, [p[0], p[1], 0] ).reshape( (3,) )
            axS.plot( [Sd[0]], [Sd[1]], [Sd[2]], 'o', color='yellow', label='debug' )

        xyz_lim = np.max( self.s )

        for ax in [ axS, axG ]:
            ax.set_xlim( -xyz_lim, xyz_lim )
            ax.set_ylim( -xyz_lim, xyz_lim )
            ax.set_zlim( -xyz_lim, xyz_lim )

        self.canvas_frame.show()
        