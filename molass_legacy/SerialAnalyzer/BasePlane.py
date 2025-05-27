# coding: utf-8
"""
    BasePlane.py

    Copyright (c) 2019-2020, SAXS Team, KEK-PF
"""
import numpy as np
import statsmodels.api as sm
from LmfitThreadSafe import minimize, Parameters
from ThreeDimUtils import compute_plane
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
import molass_legacy.KekLib.DebugPlot as plt
# import matplotlib.pyplot as plt

LOW_PERCENTILE_BASE_1ST = 30
LOW_PERCENTILE_BASE_FIN = 15
def debug_plot(title, data, x, ix, iy, e_curve, smp_i, A, B, C):
    from mpl_toolkits.mplot3d import Axes3D
    xs = x[ix]
    ys = np.arange( iy.start, iy.stop )
    smp_x = x[smp_i]

    fig = plt.figure( figsize=( 10, 10 ) )
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    # ax.set_xlim(x[0], x[-1])

    xx, yy = np.meshgrid( np.linspace(xs[0], xs[-1], 10), np.linspace(ys[0], ys[-1], 10) )
    zz = xx * A + yy * B + C
    ax.plot_surface(xx, yy, zz, color='red', alpha=0.1 )

    y_ = ys
    x_ = np.ones(len(y_)) * smp_x
    z_ = e_curve.y
    ax.plot(x_, y_, z_, color='blue')
    for b in e_curve.boundaries:
        xp = smp_x
        zp = data[smp_i,b]
        # ax.plot([xp], [b], [zp], marker='o', color='yellow')
        # workaround for matplotlib 3 bug?
        ax.plot([xp, xp], [b, b], [zp, zp], marker='o', color='yellow')

    x_ = xs

    for k, info in enumerate( e_curve.peak_info ):
        peak_i  = int( info[1]+0.5 )
        y_ = np.ones(len(x_)) * peak_i
        z_ = data[ix,peak_i]
        ax.plot(x_, y_, z_, ':', color='green')
        xp = smp_x
        yp = peak_i
        zp = data[smp_i,peak_i]
        # ax.plot([xp], [yp], [zp], marker='o', color='red')
        # workaround for matplotlib 3 bug?
        ax.plot([xp, xp], [yp, yp], [zp, zp], marker='o', color='red')

    fig.tight_layout()
    plt.show()

class LowPercentilePlane:
    def __init__(self, data, x, ecurve_y, ix, iy, smp_i, debug=False):
        """
            z = A * x + B * y + C
            
        """
        usable_data_orig = np.array( data[ ix, iy ] )
        usable_data = np.array( usable_data_orig )

        for i in range(3):
            zpp = np.percentile( usable_data, LOW_PERCENTILE_BASE_1ST )
            # print( 'usable_data < zpp=', usable_data < zpp )
            wpp = np.where( usable_data < zpp )
            # print( 'wpp=', wpp )

            x_  = x[ix.start + wpp[0]]
            y_  = iy.start + wpp[1]
            z_  = usable_data_orig[ wpp[0], wpp[1] ]

            X   = np.array( [ x_, y_, np.ones( len(x_) ) ] ).T
            mod = sm.OLS( z_, X )
            res = mod.fit()
            A, B, C = res.params
            ux  = x[ ix ]
            uy  = np.arange( iy.start, iy.stop)
            u_base  = compute_plane( A, B, C, ux, uy )
            if i == 0:
                usable_data -= u_base

        # TODO: replace LOW_PERCENTILE_BASE_FIN with a dynamically determined percentile
        kth = int( usable_data.size * LOW_PERCENTILE_BASE_FIN / 100 )
        zii     = np.argpartition( usable_data, kth, axis=None )
        zii_2d  = np.unravel_index( zii[0], usable_data.shape )
        kth_index = [ ix.start + zii_2d[0], iy.start + zii_2d[1] ]
        iff = kth_index[0]
        xff = x[iff]
        yff = kth_index[1]
        zff = data[ iff, yff ]
        # self.final_base_point = ( xff, yff, zff )
        C_  = zff - ( A * xff + B * yff )

        if debug:
            uv_elution  = ecurve_y[iy]
            e_curve     = ElutionCurve( uv_elution )
            debug_plot("Absorbance.solve_bottomplane_LPM debug", data, x, ix, iy, e_curve, smp_i, A, B, C_)

        self.params = ( A, B, C_ )

    def get_params(self):
        return self.params

class LambertBeerPlane:
    def __init__(self, data, x, ix, iy, ecurve_y, smp_i, istd_, debug=False):
        print('ix=', ix)
        print('iy=', iy)
        print('data.shape=', data.shape)
        print('data[ix, iy].shape=', data[ix,iy].shape)
        y   = np.arange( iy.start, iy.stop )
        uv_elution  = ecurve_y[iy]
        e_curve     = ElutionCurve( uv_elution )

        if False:
            from molass_legacy.Elution.CurveUtils import simple_plot
            print('iy.start, iy.stop=', iy.start, iy.stop)
            print('e_curve.peak_info=', e_curve.peak_info)
            print('e_curve.boundaries=', e_curve.boundaries)
            fig = plt.figure()
            ax = fig.gca()
            simple_plot(ax, e_curve, "LambertBeerPlane.__init__")
            plt.show()

        x_  = x[ix]
        M   = data[ix,iy]

        def obj_func( params ):
            A   = params['A']
            B   = params['B']
            C   = params['C']
            BP  = compute_plane( A, B, C, x_, y )
            M_  = M - BP

            P_list = []
            C_list = []
            start = 0
            for k, info in enumerate( e_curve.peak_info ):
                if k < len(e_curve.boundaries):
                    stop = e_curve.boundaries[k] - iy.start
                else:
                    stop = len(uv_elution) - iy.start

                # print([k], start, stop)
                if start >= stop:
                    # occured in 20170307
                    # is this ok?
                    continue

                peak_j  = int( info[1]+0.5 ) - iy.start
                P_list.append( M_[:,peak_j] )
                C_ = np.zeros(len(uv_elution))
                V_ = M_[istd_,start:stop]       # extract the corrected concentration
                C_[start:stop] = V_/np.max(V_)  # normalize to the max
                C_list.append(C_)

                start = stop

            P_ = np.array(P_list).T
            C_ = np.array(C_list)

            if False:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure( figsize=( 10, 10 ) )
                ax = fig.add_subplot(111, projection='3d')
                ax.set_title("obj_func2 debug")
                for k, info in enumerate( e_curve.peak_info ):
                    peak_j  = int( info[1]+0.5 )
                    y_ = np.ones(len(x_))*peak_j
                    z_ = P_[:,k]
                    ax.plot(x_, y_, z_, color='green')
                    xs = np.ones(len(y))*x          # bug x
                    z_ = C_[k,:]*P_[istd_,k]
                    ax.plot(xs, y, z_)
                fig.tight_layout()
                plt.show()

            return ( np.dot(P_, C_) - M_ ).flatten()

        params = Parameters()
        params.add('A', value=0,   min=-1e-2,  max=+1e-2 )
        params.add('B', value=0,   min=-1e-2,  max=+1e-2 )
        params.add('C', value=0,   min=-1.0,   max=+1.0 )

        result  = minimize( obj_func, params, args=() )
        A   = result.params['A'].value
        B   = result.params['B'].value
        C   = result.params['C'].value

        if debug:
            debug_plot("Absorbance.solve_bottomplane_LB debug", data, x, ix, iy, e_curve, smp_i, A, B, C)

        self.params = (A, B, C)

    def get_params(self):
        return self.params
