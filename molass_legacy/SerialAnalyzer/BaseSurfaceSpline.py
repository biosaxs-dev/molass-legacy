"""
    BaseSurfaceSpline.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from molass_legacy.KekLib.ThreeDimUtils import compute_plane
from .AveragingDiscussion import extrapolate
import molass_legacy.KekLib.DebugPlot as plt

def compute_concentration(a, b, c, sign=1):
    return ( -b + sign*np.sqrt(b**2 - 4*a*c) ) / (2*a)

class BaseSurfaceSpline:
    def __init__(self, sd):
        self.q = sd.intensity_array[0,:,0]
        self.data = sd.intensity_array[:,:,1].T
        self.e_curve = sd.xray_curve

        """
        argmin(a,b,c,d,e,f,g) ||AB・W - M_||
        such that

        AB = [d*x+e, f*x+g].T

        z = a*x + b*y + c
        BP = compute_plane(a, b, c, x, y)

        M_ = M - BP
        M_(x,j) = (d*x+e)*u[j] + (f*x+g)*u[j]**2
        u = compute_concentration(f*x+g, d*x+e, -M_(x,:))

        U = [u, u**2]

        """
        peak_i = self.e_curve.primary_peak_i
        xslice = slice(0,100)
        yslice = slice(peak_i-50, peak_i+50)
        y = np.arange(self.data.shape[1])[yslice]

        u   = self.e_curve.y[yslice]
        u_  = u/np.max(u)

        wsize = 10
        w_list = []
        abc_list = []
        for k in range(0, 100, wsize):
            wslice = slice(k,k+wsize)
            M   = self.data[wslice,yslice]
            P   = extrapolate(M, u_)
            Aq  = P[:,0]
            Bq  = P[:,1]
            d_init = np.average(Aq)
            e_init = np.average(Bq)
            init_params = np.array([0, 0, 0, d_init, e_init])

            w   = self.q[wslice]
            i   = wsize//2
            ones = np.ones(M.shape[0])
            def obj_func(p):
                BP = compute_plane(*p[0:3], w, y)
                M_ = M - BP
                r = compute_concentration(p[4], p[3], -M_[i,:])
                # r_ = r/np.max(r)
                r_ = r
                if False:
                    fig = plt.figure()
                    ax = fig.gca()
                    ax.plot(M_[i,:])
                    ax.plot(r_*np.max(M_[i,:]))
                    plt.show()
                R = np.array([r_, r_**2])
                AB = np.array([ones*p[3], ones*p[4]]).T
                ret = np.sum((AB @ R - M_).flatten()**2)
                # print('ret=', ret)
                return ret

            res = minimize(obj_func, init_params)
            a,b,c = res.x[0:3]
            w_list.append(w)
            abc_list.append([a,b,c])

        if True:
            x = self.q[xslice]
            xx, yy = np.meshgrid( x, y )
            i = np.arange(xslice.start, xslice.stop)
            j = np.arange(yslice.start, yslice.stop)
            ii, jj = np.meshgrid( i, j )

            zz = self.data[ii, jj]

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(xx, yy, zz, color='green', alpha=0.3 )

            if False:
                x_ = self.q[wslice]
                y_ = np.ones(len(x_))*peak_i
                ax.plot(x_, y_, Aq, color='red')
                ax.plot(x_, y_, Bq, color='blue')

            for k, w in enumerate(w_list):
                a,b,c = abc_list[k]
                print([k], a, b, c)
                ww, yy = np.meshgrid( w[[0,-1]], y[[0,-1]] )
                zz = a*ww + b*yy + c
                ax.plot_surface(ww, yy, zz, color='red', alpha=0.3 )

            fig.tight_layout()
            plt.show()
