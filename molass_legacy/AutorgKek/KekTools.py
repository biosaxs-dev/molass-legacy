# coding: utf-8
"""
    KekTools.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF
"""

import sys
import traceback
import time
import numpy                as np
import scipy.stats          as stats
import matplotlib.pyplot    as plt
from bisect                 import bisect_right
from IntensityData          import IntensityData, ACCEPTABLE_BASIC_QUALITY
from ErrorResult            import ErrorResult

DEBUG = False
MINIMUM_INTERVAL_SIZE   = 5
RVALUE_ALLOWANCE        = 0.01

class DummyFit:
    def __init__( self, **entries ): 
        self.__dict__.update(entries)

class AutorgKek:
    def __init__( self, file=None ):

        if file is None: return

        if type( file ) == str or type( file ) == np.ndarray:
            try:
                intensity = IntensityData( file, add_smoother=False )
            except ( ValueError, IndexError ):
                # these errors are known for very low quality data
                # which include many negative values.
                intensity = None
        elif type( file ) == IntensityData:
            intensity = file
        else:
            assert( False )

        self.intensity  = intensity
        self.interval   = None
        self.guinier    = None
        self.optimal_interval_q = None

    def run( self, robust=False, optimize=True,
                qrg_limits=[ 0.0, 1.3 ],
                retry_mode=False
           ):

        self.fit = DummyFit()

        self.X  = X = self.intensity.X
        self.Y  = Y = self.intensity.Y
        self.smoother   = self.intensity.smoother

        self.m   = m = np.argmax( Y )
        print( 'm=', m )

        self.small_boundary = small_boundary = int( len( Y ) ) - 1
        stop_j  = self.seartch_for_stop( m, self.small_boundary )
        start_i = self.seartch_for_start( m, stop_j )
        stop_j  = self.seartch_for_stop( start_i, stop_j )

        slope, intercept, r_value, p_value, std_err = stats.linregress( X[start_i:stop_j], Y[start_i:stop_j] )
        Q   = np.sqrt( self.X[stop_j-1] )
        Rg  = np.sqrt( 3*(-slope) ) if slope < 0 else None
        print( (start_i, stop_j), 'Rg=', Rg, 'Q*Rg=', Q*Rg )

        # self.solve_by_annealing()

        self.result = ErrorResult()
        return self.result

    def seartch_for_stop( self, start, stop ):
        r_list = []
        s_list = []
        for j in range( start + MINIMUM_INTERVAL_SIZE, stop ):
            slope, intercept, r_value, p_value, std_err = stats.linregress( self.X[start:j], self.Y[start:j] )
            r_list.append( r_value )
            s_list.append( std_err )
            Q   = np.sqrt( self.X[j] )
            if slope < 0:
                Rg  = np.sqrt( 3*(-slope) )
                if Q*Rg > 1.3:
                    break
            else:
                continue

        r_array = np.array( r_list )
        s_array = np.array( s_list )
        rm = np.argmin( r_array )
        print( 'rm=', rm, self.X[rm], 'Rg=', Rg )
        stop_j = min( j, rm + 1)

        # self.debug_plot( r_array, s_array, start )    # broken

        return stop_j

    def seartch_for_start( self, start, stop ):
        r_list = []
        s_list = []
        # qrg_list = []
        Q   = np.sqrt( self.X[stop-1] )

        for i in range( start, stop - MINIMUM_INTERVAL_SIZE ):
            slope, intercept, r_value, p_value, std_err = stats.linregress( self.X[i:stop], self.Y[i:stop] )
            r_list.append( r_value )
            s_list.append( std_err )
            Rg  = np.sqrt( 3*(-slope) ) if slope < 0 else None
            # qrg_list.append( Q*Rg )

        r_array = np.array( r_list )
        s_array = np.array( s_list )
        rm = np.argmin( r_array )
        print( 'rm=', rm, self.X[rm], 'Rg=', Rg )
        return rm

    def solve_by_annealing( self ):
        from SimulatedAnnealing     import SimulatedAnnealing

        X   = self.X
        Y   = self.Y
        m   = self.m
        small_boundary  = self.small_boundary

        def xconstaints( x ):
            return x[0] + 5 < x[1]

        def f( x ):
            i = int( x[0] )
            j = int( x[1] )
            slope, intercept, r_value, p_value, std_err = stats.linregress( X[i:j], Y[i:j] )
            return r_value + std_err

        anneal = SimulatedAnnealing()
        anneal.minimize( f, xrange=[ [m, small_boundary], [m, small_boundary] ], start=[m, small_boundary], seed=1234,
                        xconstaints=xconstaints )

        print( 'anneal.start=', anneal.start )
        print('Best solution: ' + str(anneal.xc))
        print('Best objective: ' + str(anneal.fc))

    def debug_plot( self, r_array, s_array, start_i, stop_j ):
        fig = plt.figure( figsize=( 18, 12) )
        ax1  = fig.add_subplot( 331 )
        ax2  = fig.add_subplot( 332 )
        ax3  = fig.add_subplot( 333 )
        ax5  = fig.add_subplot( 335 )
        ax8  = fig.add_subplot( 338 )
        ax1.plot( self.intensity.X, self.intensity.Y, 'o', markersize=3 )
        xmin1, xmax1 = ax1.get_xlim()
        ymin1, ymax1 = ax1.get_ylim()

        xmin2   = xmin1 * ( 257/256 ) + xmax1 * ( -1/256 )
        xmax2   = xmin1 * 61/64 + xmax1 * 3/64
        ymin2   = ymin1 * 3/8 + ymax1 * 5/8
        ymax2   = ymax1
        ax2.set_xlim( xmin2, xmax2 )
        ax2.set_ylim( ymin2, ymax2 )
        ax2.plot( self.intensity.X, self.intensity.Y, 'o', markersize=3 )

        xmin3   = xmin1 * ( 257/256 ) + xmax1 * ( -1/256 )
        xmax3   = xmin1 * 125/128 + xmax1 * 3/128
        ymin3   = ymin1 * 2/16 + ymax1 * 14/16
        ymax3   = ymax1
        ax3.set_xlim( xmin3, xmax3 )
        ax3.set_ylim( ymin3, ymax3 )
        ax3.plot( self.intensity.X, self.intensity.Y, 'o', markersize=3 )

        ax5.set_xlim( xmin2, xmax2 )
        ax5.set_ylim( -1, -0.95 )
        ax5.plot( self.X[start_i:stop_j], r_array )
        ax8.set_xlim( xmin2, xmax2 )
        ax8.set_ylim( 0, 15 )
        ax8.plot( self.X[start_i:stop_j], s_array )

        for ax in [ ax1, ax2, ax3 ]:
            ax.plot( self.intensity.X[self.m], self.intensity.Y[self.m], 'o', color='red' )

        fig.tight_layout()
        plt.show()

def autorg( file, robust=False, optimize=True,
            qrg_limits=[ 0, 1.3 ], qrg_limits_apply=True ):
    autorg_ = AutorgKek( file  )
    autorg_.run( robust=robust, optimize=optimize,
                    qrg_limits=qrg_limits )
    return autorg_.result, autorg_.intensity
