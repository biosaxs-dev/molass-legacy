# coding: utf-8
"""
    EvolvingFactorAnalysis.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
import copy
import numpy as np
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
from mpl_toolkits.mplot3d   import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib             import colors as mcolors
from molass_legacy.Models.ElutionCurveModels     import EMGA, EGHA
from XrayDecomposer         import XrayDecomposer, proof_plot
import molass_legacy.KekLib.DebugPlot as dplt
from molass_legacy.KekLib.BasicUtils import clear_dirs_with_retry
from molass_legacy.KekLib.NumpyUtils import np_savetxt

CONCENTRATION_NOMALIZE  = True
NUM_SVS_TO_SHOW         = 5
DRAW_IN_THE_ORDER_OF_SV = True

def rectify( n, pair ):
    ret = []
    for rec in pair:
        r = rec[0:n]
        ret.append( list(r) + [0]*(n-(len(r))) )
    return ret

class EvolvingFactorAnalysis:
    def __init__( self, data, ecurve, eslice, y, s_lim=None ):

        self.data   = data
        self.ecurve = ecurve
        self.eslice = eslice
        self.y      = y
        self.y_     = y_ = y if eslice is None else y[eslice]

        if False:
            dplt.plot( y )
            dplt.plot( y_ )
            dplt.show()

        if CONCENTRATION_NOMALIZE:
            ratio = 1/np.max( [ y_, np.ones( len(y_) )* 0.001 ], axis=0 )
            normalized_data = np.prod( [ data, ratio ], axis=0 )
        else:
            normalized_data = data

        # dslice = slice(2,6)
        # dslice = slice(0,2)
        dslice = slice(0,5)
        # dslice = slice(1,2)
        U, s, VT = np.linalg.svd( normalized_data )
        print( 's=', s[0:10] )
        denoised_data =np.dot(  np.dot( U[:,dslice], np.diag( s[dslice] ) ), VT[dslice, :] )
        U, s, VT = np.linalg.svd( denoised_data )
        print( 'denoised s=', s[0:10] )

        s_max = s[0]
        if s_lim is None:
            s_lim = s_max*0.2

        if not CONCENTRATION_NOMALIZE:
            s_lim_ratio = s_lim / s_max

        rank_list = []
        sv_list = []
        sv_list2 = []
        for t in range( 1, denoised_data.shape[1] ):
            M   = denoised_data[:,0:t]
            U, s, V = np.linalg.svd( M )

            s_ = s/s[0]

            sv_list.append( [ s, s_ ] )
            sv_list2.append( rectify(NUM_SVS_TO_SHOW, [ s, s_ ]) )
            if t < 20:
                print( [t], 's=', s[0:10] )

            if CONCENTRATION_NOMALIZE:
                s_lim_  = s_lim
            else:
                s_lim_ = s[0] * s_lim_ratio

            rank_list.append( len( np.where( s > s_lim_ )[0] ) )

        self.rank_list = rank_list
        self.sv_list = sv_list
        self.sv_array = np.array(sv_list2)
        print( 'self.sv_array.shape=', self.sv_array.shape )

    def show_plot( self, sd, in_folder, rank_boundaries=None ):

        model = EGHA()
        x   = self.ecurve.x
        y   = self.y
        fa  = XrayDecomposer( self.ecurve, x, y, sd.intensity_array, retry_valley=True, model=model, deeply=True )

        x_  = np.arange(self.eslice.start, self.eslice.stop)

        fig = plt.figure( figsize=( 24, 8 ) )
        gs = gridspec.GridSpec( 1, 3 )
        ax1 = fig.add_subplot( gs[0, 0] )
        ax2 = ax1.twinx()
        ax3 = fig.add_subplot( gs[0, 1], projection='3d' )
        ax4 = fig.add_subplot( gs[0, 2], projection='3d' )

        fontsize = 14

        plt.yticks(fontsize=fontsize)

        ax1.set_title( 'Evolving Factor Analysis for ' + in_folder, fontsize=fontsize )
        ax1.set_ylabel( 'Intensity', fontsize=fontsize )
        ax1.set_xlabel( 'Elution No.', fontsize=fontsize )
        ax2.set_ylabel( 'Matrix Rank', fontsize=fontsize )
        ax1.plot( x_, self.y_, color='orange', linewidth=5 )

        def draw_decomposition( ax, fit_recs ):

            x_residual = copy.deepcopy( self.y_ )

            for _, func, _ in fit_recs:
                ey = func( x_ )
                ax.plot( x_, ey )
                x_residual -= ey

            ax.plot( x_, x_residual )

        draw_decomposition( ax1, fa.fit_recs )

        ax2.plot( x_[1:], self.rank_list, linewidth=5, color='gray' )
        ax2.set_yticks([0, 1, 2])

        ax3.set_title( "Concentration-normalized Singular Values", fontsize=fontsize )
        ax3.set_xlabel( '\nElution No.', fontsize=fontsize )
        ax3.set_ylabel( '\nRank', fontsize=fontsize )
        ax3.set_zlabel( '\nScale', fontsize=fontsize )
        ax4.set_title( "Maxvalue-normalized Singular Values", fontsize=fontsize )
        ax4.set_xlabel( '\nElution No.', fontsize=fontsize )
        ax4.set_ylabel( '\nRank', fontsize=fontsize )
        ax4.set_zlabel( '\nScale Ratio', fontsize=fontsize )

        for ax in [ ax1, ax2, ax3, ax4 ]:
            ax.tick_params( labelsize=fontsize )

        start = self.eslice.start

        if DRAW_IN_THE_ORDER_OF_SV:

            limit = min( self.sv_array.shape[2], NUM_SVS_TO_SHOW )
            for i in range(limit):
                x   = np.arange( start+i, self.eslice.stop-1 )
                y   = np.ones(len(x)) * (i+1)
                x_  = x - start
                z   = self.sv_array[x_,0,i]
                ax3.plot( x, y, z, 'o', markersize=3, label='s%d' % i  )
                z_  = self.sv_array[x_,1,i]
                ax4.plot( x, y, z_, 'o', markersize=3, label='s%d' % i  )

            ax3.legend(fontsize=fontsize, loc='upper left')
            ax4.legend(fontsize=fontsize, loc='upper left')

        else:

            for k, sv in enumerate( self.sv_list ):
                limit = min( len(sv[0]), NUM_SVS_TO_SHOW ) 
                x   = np.ones(limit) * (start + k)
                y   = np.arange( limit ) + 1
                z   = sv[0][0:limit]
                ax3.plot( x, y, z, 'o', markersize=3 )
                z_  = sv[1][0:limit]
                ax4.plot( x, y, z_, 'o', markersize=3 )

        if rank_boundaries is not None:
            ymin, ymax = ax1.get_ylim()
            ax1.set_ylim( ymin, ymax )
            for b in rank_boundaries:
                ax1.plot( [b, b], [ymin, ymax ], color='red', linewidth=5 )

            for k, ax in enumerate([ ax3, ax4 ]):
                ymin, ymax = ax.get_ylim()
                zmin, zmax = ax.set_zlim()
                zs_list = rank_boundaries
                verts1  = []
                for b in rank_boundaries:
                    yp  = [ ymin, ymin, ymax, ymax ]
                    zp  = [ zmax, zmin, zmin, zmax ]
                    verts1.append( list( zip( yp, zp ) ) )
                    if DRAW_IN_THE_ORDER_OF_SV:
                        x   = np.ones(NUM_SVS_TO_SHOW)*b
                        y   = np.arange( NUM_SVS_TO_SHOW ) + 1
                        z   = self.sv_array[b-start,k,0:NUM_SVS_TO_SHOW]
                        ax.plot( x, y, z, ':', color='gray' )

                r = mcolors.to_rgba('red', alpha=0.2)
                poly1 = PolyCollection(verts1, facecolors=[r]*len(verts1) )
                # poly1.set_alpha(0.2)
                ax.add_collection3d(poly1, zs=zs_list, zdir='x')

        fig.tight_layout()
        plt.show()

class ProofData:
    def __init__( self, sd, mapper ):
        print( mapper.x_curve.peak_info )
        x_curve = mapper.x_curve
        a_baseline = mapper.a_base + mapper.a_base_adjustment
        x_baseline = mapper.x_base + mapper.x_base_adjustment

        x_corrected = mapper.x_vector - x_baseline
        sd.apply_baseline_correction( mapper.get_mapped_info() )
        data    = sd.intensity_array

        print( 'sd.xray_slice=', sd.xray_slice )
        self.xray_slice = sd.xray_slice

        y   = x_corrected
        x   = np.arange( len(y) )
        emg_model = EMGA()
        self.fa = XrayDecomposer( x_curve, x, y, data, retry_valley=True, model=emg_model, deeply=True )
        self.ridges = [ data[int(info[1]+0.5),:,1] for info in x_curve.peak_info ]
        assert len( self.ridges ) == len( self.fa.fit_recs )

        self.qvector = sd.qvector
        self.evector = np.zeros( len(self.qvector) )
        self.C  = np.array( [ rec[1](x) for rec in self.fa.fit_recs ] )

        if False:
            for ridge in self.ridges:
                dplt.plot( ridge )
            dplt.show()

    def show( self, parent ):
        proof_plot( self.fa, parent )

    def save_weak( self, folder ):
        clear_dirs_with_retry( [folder] )

        A   = np.array( self.ridges ).T
        M   = np.dot( A, self.C )
        for j in range( M.shape[1] ):
            file = folder + '/Weak_%05d.dat' % j
            data = np.array( [ self.qvector, M[:,j], self.evector ] ).T
            np_savetxt( file, data )

    def save_strong( self, folder ):
        clear_dirs_with_retry( [folder] )

        shift_amount = 20
        zero_slice_ = slice( self.xray_slice.start + shift_amount, self.xray_slice.stop + shift_amount )

        strong_ridges = []
        for k, ridge in enumerate(self.ridges):
            if k % 2 == 0:
                ridge_ = ridge
            else:
                ridge_ = ridge
                ridge_[zero_slice_] = 0
            strong_ridges.append( ridge_ )

        A   = np.array( strong_ridges ).T
        M   = np.dot( A, self.C )
        for j in range( M.shape[1] ):
            file = folder + '/Strong_%05d.dat' % j
            data = np.array( [ self.qvector, M[:,j], self.evector ] ).T
            np_savetxt( file, data )
