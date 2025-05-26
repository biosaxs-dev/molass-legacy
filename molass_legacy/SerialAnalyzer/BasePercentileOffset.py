# coding: utf-8
"""

    BasePercentileOffset.py

    scattering baseline solver assuming Gaussian Curves

    Copyright (c) 2017-2020, Masatsuyo Takahashi, KEK-PF

"""
import os
import numpy            as np
from scipy.stats        import norm
from bisect             import bisect_right

TABLE_SIZE          = 100
N_SCALE             = 0.01
MAX_NOISINESS       = 1.0
START_SIZE_SIGMA    = 3
END_SIZE_SIGMA      = 13
NUM_SIZE_SIGMAS     = 21
DEFAULT_OFFSET      = 10

def compute_noise_and_size_sigma_dependency( num_iteretions=1000 ):
    N = 300
    np.random.seed(1234)

    ym  = norm.pdf( 0, 0, 1 )
    scale = 1/ym

    dat_list = []
    std_list = []

    for n in range(TABLE_SIZE):
        nn = N_SCALE * (n+1)

        p_list = [ nn ]
        e_list = [ nn ]

        for size_sigma in np.linspace( START_SIZE_SIGMA, END_SIZE_SIGMA, NUM_SIZE_SIGMAS ):
            x   = np.linspace( -size_sigma, size_sigma, N )
            y0  = norm.pdf( x, 0, 1 )
            y_  = y0 * scale

            ys_array = []
            zi_array = []

            for i in range(num_iteretions):
                noise = np.random.normal(0, nn, N)
                y   = y_ + noise
                ys  = sorted( y )
                zi  = bisect_right( ys, 0 )
                if False:
                    import molass_legacy.KekLib.DebugPlot as plt
                    if i==0 and n%10==0 and abs(size_sigma - 9) < 0.1:
                        plt.push()
                        fig, ax = plt.subplots()
                        axt = ax.twinx()
                        ax.set_title("nn=%.3g, %.3g sigma width" % (nn, size_sigma))
                        ax.plot(y_)
                        ax.plot(y)
                        axt.plot(ys, ':')
                        axt.plot(zi, ys[zi], 'o', color='red')
                        fig.tight_layout()
                        plt.show()
                        plt.pop()
                zi_array.append( zi )
                ys_array.append( ys[zi] )

            p = np.average( zi_array ) * 100 / N
            e = np.std( zi_array )
            p_list.append( p )
            e_list.append( e )

        dat_list.append( p_list )
        std_list.append( e_list )

    return np.array(dat_list), np.array(std_list)

bpo_table = None
std_table = None

def load_bpo_table():
    global bpo_table
    if bpo_table is not None: return

    folder = os.path.dirname( __file__ )
    bpo_dat_path = folder + '/base_percentile_offset.dat'
    # bpo_std_path = folder + '/base_percentile_offset_std.dat'
    if not os.path.exists( bpo_dat_path ):
        dat, std = compute_noise_and_size_sigma_dependency( num_iteretions=1000 )
        np.savetxt( bpo_dat_path, dat )
        # np.savetxt( bpo_std_path, std )
        bpo_table = dat
        # std_table = std
    else:
        # import time
        # start_ = time.time()
        bpo_table = np.loadtxt( bpo_dat_path )
        # end_ = time.time()
        # std_table = np.loadtxt( bpo_std_path )
        # print( 'took', end_ - start_, 'to load bpo_table.' )

    # print('bpo_table.shape=', bpo_table.shape)

load_bpo_table()

def base_percentile_offset( noisiness, size_sigma=5 ):
    # See Also ElutionBaseCurve.py
    # return DEFAULT_OFFSET

    if not np.isfinite( noisiness ):
        noisiness   = MAX_NOISINESS

    nn = max( 0, min( TABLE_SIZE - 1, int( round( noisiness / N_SCALE ) ) ) )
    j = max( 0, min( NUM_SIZE_SIGMAS - 1, int( round( ( size_sigma - START_SIZE_SIGMA )*2 ) ) ) )
    # print('nn, j=', (nn, j))
    bpo =  bpo_table[ nn, j+1 ]
    # print( 'base_percentile_offset: noisiness=', noisiness, 'size_sigma=', size_sigma, 'bpo=', bpo  )
    return bpo
