# coding: utf-8
"""
    AbsorbanceViewUtils.py

    Copyright (c) 2018-2022, SAXS Team, KEK-PF
"""
import numpy                as np
from molass_legacy._MOLASS.SerialSettings         import get_setting
from .Absorbance import LOW_PERCENTILE_BASE_1ST

def draw_3d( ax, absorbance, wvlen_lower, wvlen_upper, i_slice, zlim=None,
                title="", title_fontsize=None, low_percentile=True ):
    ax.set_title( title, fontsize=title_fontsize )
    ax.set_xlabel( '\nwave length (nm)' )
    ax.set_ylabel( '\nsequential №' )
    ax.set_zlabel( '\nabsorbance' )

    if zlim is None:
        # zlim = ( -0.05, 0.2 )
        # ax.set_zlim( zlim )
        pass
    else:
        ax.set_zlim( zlim )

    data = absorbance.data
    print( 'data.shape=', data.shape )

    size = data.shape[1]

    peak_j = None
    tail_i = absorbance.tail_index
    absorbance_baseline_type = get_setting( 'absorbance_baseline_type' )

    if low_percentile:
        zpp = np.percentile( data, LOW_PERCENTILE_BASE_1ST )
        wpp = np.where( data < zpp )
        npp = data.shape[0]*data.shape[1]
        if False:
            print( 'data.shape=', data.shape, npp)
            print('wpp[0].shape=', wpp[0].shape, wpp[0].shape[0]/npp, 'wpp[1].shape=', wpp[1].shape[0], wpp[1].shape[0]/npp)
            print( 'wpp=', wpp )
        x_ = absorbance.wl_vector[wpp[0]]
        y_ = wpp[1]
        z_ = data[wpp[0], wpp[1]]
        ax.plot(x_, y_, z_, 'o', color='yellow')

    for i, wv in enumerate( absorbance.wl_vector ):
        if i == tail_i:
            pass
        else:
            if wv < wvlen_lower or wv > wvlen_upper:
                pass
            if i != absorbance.index and i % 2 > 0:
                continue

        X = np.ones( size ) * wv
        Y = np.arange( size )
        Z = data[i,:]
        if i == absorbance.index:
            alpha   = 1
            color   = 'blue'
        else:
            alpha   = 0.2
            color   = '#1f77b4'
        ax.plot( X, Y, Z, color=color, alpha=alpha )

        if absorbance_baseline_type == 0 or absorbance.base_curve is None:
            continue

        if i  == tail_i:
            slice_ = slice( absorbance.rail_points[0], absorbance.rail_points[1]+1 )
            Y_ = Y[slice_]
            curve_size = len(absorbance.tail_curve)
            for ii in range( i-5, i+6, 2 ):
                X_ = np.ones( curve_size ) * absorbance.wl_vector[ii]
                ax.plot( X_, Y_, absorbance.tail_curve, color='red', alpha=0.5 )

            X_ = np.ones( curve_size ) * absorbance.std_wvlen
            ax.plot( X_, Y_, absorbance.base_curve, color='red', alpha=0.5 )

    draw_3d_detail( ax, absorbance, i_slice )

def draw_3d_detail( ax, absorbance, i_slice ):
    wvlen   = absorbance.wl_vector
    absorbance  = absorbance

    wvlen_ = wvlen[i_slice]

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
    # no flow_changes plot here
    colors = [ 'green', 'orange', 'green' ]
    for i, j in enumerate( [ lower_j, middle_j, upper_j ]):
        print( 'draw_3d_absorbance_detail: (i, j)=', (i, j) )
        if j is None:
            continue

        X = wvlen_
        Y = np.ones( len(wvlen_) ) * j
        Z = absorbance.data[i_slice,j]
        section_array.append( [ j, Z ] )
        color = colors[i]
        ax.plot( X, Y, Z, ':', color=color )

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

    if False:
        ipp = absorbance.istd + absorbance.wpp[0]
        xpp = absorbance.wl_vector[ ipp ]
        ypp = absorbance.jump_j_safe + absorbance.wpp[1]
        zpp = absorbance.data[ ipp, ypp ]
        ax.plot( xpp, ypp, zpp, 'o', color='yellow', markersize=3, alpha=0.1 )

        xff, yff, zff = absorbance.final_base_point
        print( 'xff, yff, zff=', xff, yff, zff )
        ax.plot( [xff], [yff], [zff], 'o', color='red', markersize=5 )
