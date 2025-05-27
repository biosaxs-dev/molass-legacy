# coding: utf-8
"""
    AbsorbancePlot.py

    Copyright (c) 2017-2018, Masatsuyo Takahashi, KEK-PF
"""
import numpy                as np
from bisect                 import bisect_right
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
from mpl_toolkits.mplot3d   import Axes3D
from molass_legacy.UV.Absorbance             import Absorbance
from molass_legacy.SerialAnalyzer.AnalyzerUtil           import compute_conc_factor_util
from molass_legacy._MOLASS.SerialSettings         import get_setting

DEBUG = False

def get_temp_absorbance( dialog, serial_data ):
    serial_data.wait_until_ready()

    abs_ = serial_data.absorbance
    std_wvlen = dialog.absorbance_picking.get()
    try:
        end_wvlen = dialog.zero_absorbance.get()
    except:
        end_wvlen = None

    temp_absorbance = Absorbance( abs_.wl_vector, abs_.data, abs_.i_curve, std_wvlen=std_wvlen, end_wvlen=end_wvlen )
    return temp_absorbance

def show_absorbance_figure_util( dialog, make_temp=False ):
    print( 'show_absorbance_figure_util' )
    dialog.config( cursor='wait' )
    dialog.update()

    if make_temp:
        dialog.temp_absorbance = absorbance = get_temp_absorbance( dialog, dialog.parent.serial_data )
    else:
        absorbance = dialog.parent.serial_data.absorbance

    conc_factor = compute_conc_factor_util()
    # absorbance.apply_mapping( conc_factor, dialog.parent.serial_data.xray_curve )
    # why this call ?

    plot = AbsorbancePlot( absorbance )
    plot.draw_3d_for_setting_gui( dialog, 'Absorbance from ' + dialog.parent.uv_folder.get() )
    dialog.config( cursor='' )

class AbsorbancePlot:
    def __init__( self, absorbance ):
        self.absorbance     = absorbance
        self.wl_vector      = absorbance.wl_vector
        self.wvlen_lower    = 245
        self.wvlen_upper    = 450
        f = bisect_right( absorbance.wl_vector, self.wvlen_lower )
        t = bisect_right( absorbance.wl_vector, self.wvlen_upper )
        self.i_slice        = slice( f, t )

    def draw_3d_absorbance( self, ax, data, title, absorbance_baseline_type=None, zlim=None ):

        lower   = self.wvlen_lower
        upper   = self.wvlen_upper

        ax.set_title( title )
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

        print( 'data.shape=', data.shape )

        size = data.shape[1]

        peak_j = None
        tail_i = self.absorbance.tail_index
        if absorbance_baseline_type is None:
            absorbance_baseline_type = get_setting( 'absorbance_baseline_type' )
        for i, wv in enumerate( self.absorbance.wl_vector ):
            if i == tail_i:
                pass
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

    def draw_3d_absorbance_detail( self, ax, ax_2d ):
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

        bline = absorbance.get_bottomline_matrix()

        xmin, xmax = ax.get_xlim()
        zmin, zmax = ax.get_zlim()
        xoffset = ( xmax - xmin ) * 0.01
        zoffset = ( zmax - zmin ) * 0.005

        Y = np.arange( absorbance.data.shape[1] )
        for i in range(4):
            wvlen = absorbance.reg_wvlens[i]
            X = np.ones( absorbance.data.shape[1] ) * wvlen
            Z = bline[:, i]
            if i == 3:
                color   = 'cyan'
                alpha   = 1
            else:
                color   = 'red'
                alpha   = 0.5
            ax.plot( X, Y, Z, ':', color=color, alpha=alpha )
            if i in [0,3]:
                ax.text( wvlen - xoffset, Y[-1], Z[-1] + zoffset, 'λ=%g' % wvlen )

    def get_component( self, U, s, V, i ):
        # Be aware that we are slicing on the column of V, because V is (Vtt=V), not Vt.
        return s[ i ] * np.dot( np.transpose( [ U[ :, i ] ] ), np.array( [ V[ :, i ] ] ) )

    def draw_wave_length_side_2d( self, ax, wvlen, section_array ):
        ax.set_title( 'Absorbance at sequential sections' )
        ax.set_xlabel( 'Wave Length' )
        ax.set_ylabel( 'Absorbance' )
        ax.set_xlim( self.wvlen_lower - 10 , self.wvlen_upper + 10 )

        colors = [ 'green', 'orange', 'green', 'purple', 'purple' ]
        max_z = []
        labels = [ 'left', 'peak', 'right', 'jump point', 'jump point' ]

        m = np.argmax( section_array[1][1] )

        for i, rec in enumerate( section_array ):
            j, Z = rec
            line, = ax.plot( wvlen, Z, ':', color=colors[i],  label='seqno %d ( %s )' % ( j, labels[i] ) )
            max_z.append( Z[m] )

        A, B, C, S  = self.absorbance.get_params()
        b_points    = self.absorbance.get_bottomline_vector_near_peak()
        print( 'b_points.shape=', b_points.shape )
        j_peak, Z_peak = section_array[1]
        i0 = self.absorbance.index_vector[0] - self.i_slice.start
        peak_conc   = Z_peak[i0] - b_points[1][0]

        print( 'A, B, C=', A.value, B.value, C.value )
        print( 'b_points[1][0]=', b_points[1][0] )

        section_bottom = A * wvlen + B * j_peak + C
        ax.plot( wvlen, section_bottom, ':', color='pink' )

        for i in [ 0, 2 ]:
            if i >= len(max_z):
                continue

            j, Z = section_array[i]
            print( i, 'j=', j )
            section_bottom = A * wvlen + B * j + C
            Z_ = Z - section_bottom
            scale = peak_conc / Z_[i0]
            z = Z_ * scale + section_bottom
            ax.plot( wvlen, z, ':', color='yellow', label='seqno %d scaled' % j )
            ax.plot( wvlen, section_bottom, ':', color='pink' )

        x0  = self.absorbance.reg_wvlens[0]
        z0  = self.absorbance.get_standard_vector()
        y0_min = np.min( z0 )
        y0_max = np.max( z0 )
        ax.plot( [ x0, x0 ], [ y0_min, y0_max ], color='blue', alpha=0.5 )

        x = self.absorbance.reg_wvlens
        for i, bv in enumerate( b_points ):
            print( i, 'bv=', bv )
            ax.scatter( x[:-1], bv[:-1], color='red', alpha=0.5, s=20 )
            ax.scatter( x[-1], bv[-1], color='cyan', alpha=1, s=20 )

        ymin, ymax = ax.get_ylim()
        yoffset = ( ymax - ymin )*0.05
        y_ = np.min( b_points ) - yoffset*0.3

        for i in [0,3]:
            x_ = x[i]
            ax.annotate( 'λ=%g' % (x_), xy=(x_, y_),
                            xytext=( x_, y_ - yoffset ),
                            ha='center', va='center',
                            arrowprops=dict( headwidth=5, width=0.5, color='black', shrink=0.05),
                            )
        ax.legend()

    def draw_3d_with_svd( self, title ):

        U, s, Vt = np.linalg.svd( self.absorbance.data )
        V = np.transpose( Vt )

        components = []
        for i in range(3):
            components.append( self.get_component( U, s, V, i ) )

        fig = plt.figure( figsize=( 15, 9 ) )
        gs = gridspec.GridSpec( 3, 3 )
        ax1 = fig.add_subplot( gs[:, 0:2], projection='3d' )
        ax2 = fig.add_subplot( gs[0, 2], projection='3d' )
        ax3 = fig.add_subplot( gs[1, 2], projection='3d' )
        ax4 = fig.add_subplot( gs[2, 2], projection='3d' )

        self.draw_3d_absorbance( ax1, self.absorbance.data, title )

        fig_2d = plt.figure( figsize=( 8, 6 ) )
        ax_2d  = fig_2d.add_subplot( 111 )
        self.draw_3d_absorbance_detail( ax1, ax_2d )
        fig_2d.tight_layout()

        zlim = ax1.get_zlim()
        self.draw_3d_absorbance( ax2, components[0], "Component 0", zlim=zlim )
        self.draw_3d_absorbance( ax3, components[1], "Component 1", zlim=zlim )
        self.draw_3d_absorbance( ax4, components[2], "Component 2", zlim=zlim )

        fig.tight_layout()
        plt.show()

    def draw_3d_for_setting_gui( self, parent, title, absorbance_baseline_type=None ):
        from DebugCanvas            import DebugCanvas
        from molass_legacy.KekLib.TkUtils                import is_low_resolution

        def draw_func( fig ):
            # fig.set_size_inches( 20, 10 )
            # gs  = gridspec.GridSpec( 1, 2 )
            # ax1 = fig.add_subplot( gs[0,0], projection='3d' )
            # ax2 = fig.add_subplot( gs[0,1] )
            ax1 = fig.add_subplot( 111, projection='3d' )
            ax2 = None
            self.draw_3d_absorbance( ax1, self.absorbance.data, title, absorbance_baseline_type=absorbance_baseline_type )
            self.draw_3d_absorbance_detail( ax1, ax2 )
            fig.tight_layout()

        figsize = ( 9, 8 ) if is_low_resolution() else ( 12, 11 )
        self.canvas = DebugCanvas( "Absorbance Data 3D Plot", draw_func, parent=parent, figsize=figsize )
        self.canvas.show( cursor_update=False )
