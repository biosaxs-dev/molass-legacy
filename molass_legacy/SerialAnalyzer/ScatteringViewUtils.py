"""
    ScatteringViewUtils.py

    Copyright (c) 2018-2023, SAXS Team, KEK-PF
"""
import numpy                as np

VERY_SMALL_ANGLE_LIMIT  = 0.03
NUM_SAMPLE_CURVES       = 20
ZLIM_EXPAND             = 3.0

def is_to_plot( index, i, q, very_small_angle_only=False ):
    # return q >= 0.01 and ( i == index or i % 8 == 0 )
    if very_small_angle_only:
        return q < VERY_SMALL_ANGLE_LIMIT or i == index
    else:
        return i == index or i % 8 == 0

def compute_baselines( qvector, index, corrector ):

    vsa_base_list = []
    all_base_list = []
    for i, q  in enumerate( qvector ):
        i_select = ( i == index or i % 8 == 0 )
        if q < VERY_SMALL_ANGLE_LIMIT or i_select:
            baseline = corrector.correct_a_single_q_plane( i, return_baseline=True )
            if q < VERY_SMALL_ANGLE_LIMIT or i == index:
                vsa_base_list.append( baseline )
            if i_select:
                all_base_list.append( baseline )

    return vsa_base_list, all_base_list

def draw_3d_scattering( ax, data, qvector, index, xray_curve_y,
                        title,
                        vsa_base_list, all_base_list,
                        zlim=None, zlim_expand=None, experimental=False,
                        very_small_angle_only=False,
                        sim_base=None ):

    ax.set_title( title, y=1.1 )
    # ax.set_xlabel( '\nQ(Å⁻¹)' )
    ax.set_xlabel( '\nQ($Å^{-1}$)' )
    ax.set_ylabel( '\nElution №' )
    ax.set_zlabel( '\nIntensity' )
    # ax.zaxis._set_scale('log')

    if zlim is None:
        if zlim_expand is None:
            zlim_expand = ZLIM_EXPAND
        zmin    = np.min( xray_curve_y )
        zmax    = np.max( xray_curve_y )
        # zmin, zmax = ax.get_zlim()
        zmin_   = zlim_expand * zmin + ( 1-zlim_expand ) * zmax
        zmax_   = ( 1-zlim_expand ) * zmin + zlim_expand * zmax
        zlim    = ( zmin_, zmax_ )

    ax.set_zlim( zlim )

    print( 'data.shape=', data.shape )

    size = data.shape[0]
    Y = np.arange( size )

    peak_j = None
    i_ = -1
    i_list = []
    for i, q in enumerate( qvector ):
        if not is_to_plot( index, i, q, very_small_angle_only=very_small_angle_only ):
            continue

        i_list.append( i )

        X = np.ones( size ) * q

        if i == index:
            alpha   = 1
            color   = 'orange'
            Z = xray_curve_y
        else:
            alpha   = 0.2
            color   = '#1f77b4'
            if len(data.shape) == 3:
                Z = data[:,i,1]
            else:
                Z = data[:,i]

        ax.plot( X, Y, Z, color=color, alpha=alpha )

        i_ += 1
        if very_small_angle_only:
            baseline = vsa_base_list[i_]
        else:
            baseline = all_base_list[i_]
        ax.plot( X, Y, baseline, color='red', alpha=0.2 )

    iv = np.array(i_list)

    if very_small_angle_only:
        line_array = np.array( vsa_base_list )
    else:
        line_array = np.array( all_base_list )

    Y_ = np.linspace( 0, size-1, NUM_SAMPLE_CURVES, dtype=int )

    for k, j in enumerate( Y_ ):
        x = qvector[iv]
        y = np.ones( len(iv) ) * j
        z = line_array[:,j]
        ax.plot( x, y, z, color='yellow', alpha=0.5 )
        if sim_base is not None:
            z = sim_base[j,iv]
            ax.plot( x, y, z, ':', color='cyan', alpha=0.5 )

    if experimental:
        ax.text2D( 0.15, 0.7, "Experimental Feature", transform=ax.transAxes, alpha=0.2, fontsize=50 )
