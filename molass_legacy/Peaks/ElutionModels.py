"""

    Peaks.ElutionModels.py

    Copyright (c) 2017-2023, SAXS Team, KEK-PF

"""
import numpy as np
from scipy.special import erfc, erfcx
from scipy.optimize import fsolve

def gaussian(x, h, m, s):
    return h * np.exp(-((x-m)/s)**2)

def egh_impl(x, H, tR, sigma, tau):
    x_  = x - tR
    s2  = 2 * sigma**2
    z   = s2 + tau*x_
    z_neg   = z <= 0
    z_pos   = z > 0

    zero_part = np.zeros( len(x) )[z_neg]
    posi_part = H * np.exp( -x_[z_pos]**2/( s2 + tau*x_[z_pos] ) )

    if tau > 0:
        parts = [ zero_part, posi_part ]
    else:
        parts = [ posi_part, zero_part ]

    return np.hstack( parts )

def egh(x, H, tR, sigma, tau):
    if np.isscalar(x):
        x = np.array([x])
        return egh_impl(x, H, tR, sigma, tau)[0]
    else:
        return egh_impl(x, H, tR, sigma, tau)

"""
Kevin Lan, James W. Jorgenson, Journal of Chromatography A, 915 (2001) 1?13
"""
A0i = np.array([4, -6.293724, 9.232834, -11.342910, 9.123978, -4.173753, 0.827797])
e0 = np.poly1d(A0i[::-1])
"""
e0(th) = a0 + a1*th + a2*th**2 + ... + am*th**m
"""
A1i = np.array([0.75, 0.033807, -0.301080, 1.200371, -1.813317, 1.279318, -0.326582])
e1 = np.poly1d(A1i[::-1])

A2i = np.array([1, -0.982254, 0.568593, 0.512587, -1.184361, 0.939222, -0.240814])
e2 = np.poly1d(A2i[::-1])

# A3i = np.array([0.5, -0.664611, 0.706192, -0.293500, -0.083980, 0.200306, -0.064264])
A3i = np.array([0.49106035, -0.45648717, -0.93606276, 5.25394258, -9.16354757, 7.36039624, -2.25720537])
e3 = np.poly1d(A3i[::-1])

SQRT_PI_8 = np.sqrt(np.pi/8)
def egh_pdf(x, tR=0, sigma=1.0, tau=0.0, scale=1):
    tau_ = abs(tau)
    th = np.arctan2(tau_, sigma)
    H_ = (sigma*SQRT_PI_8 + tau_)*e0(th)
    return egh(x, scale, tR, sigma, tau)/H_

def compute_moments(x, y):
    W = np.sum(y)
    M1 = np.sum(x*y)/W              # raw moment
    M2 = np.sum(y*(x-M1)**2)/W      # central moment
    M3 = np.sum(y*(x-M1)**3)/W      # central moment
    return M1, M2, M3

def compute_moments_from_egh_params(tR, sigma, tau):
    tau_ = abs(tau)
    theta = np.arctan2(tau_, sigma)
    M1 = tR + tau*e1(theta)
    M2 = (sigma**2 + sigma*tau_ + tau**2)*e2(theta)
    M3 = tau*(3*sigma**2 + 4*sigma*tau_ + 4*tau**2)*e3(theta)
    return M1, M2, M3

def compute_egh_params(init_params, moments):
    M1, M2, M3 = moments

    def equations(p):
        tR, sigma, tau = p
        tau_ = abs(tau)         # better for 2019118_3
        # tau_ = tau            # seems better for some cases such as 20181203, although abs(tau) is mentioned in the paper.
        th = np.arctan2(tau_, sigma)
        return (
                    tR + tau*e1(th) - M1,
                    (sigma**2 + sigma*tau_ + tau**2)*e2(th) - M2,
                    tau*(3*sigma**2 + 4*sigma*tau_ + 4*tau**2)*e3(th) - M3,
               )

    x, infodict, ier, mesg = fsolve(equations, init_params, full_output=True)
    if ier in [1, 4, 5]:
        """
        5 : The iteration is not making good progress, as measured by the
            improvement from the last ten iterations.
        4 : The iteration is not making good progress, as measured by the
            improvement from the last five Jacobian evaluations.
        """
        tR, sigma, tau = x
    else:
        assert False
    # print('init_params=', init_params)
    # print('new_params=', (h, tR, sigma, tau))
    return tR, sigma, tau

"""
    This implemetation is based on the following wikipedia page
        https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
"""
VERY_SMALL_VALUE    = 1e-8
SQRT2       = 1/np.sqrt(2)
SQRTPI2     = np.sqrt(np.pi/2)
Z_LIMIT     = 6.71e7
LARGE_VALUE = 1e10

def emg_orig(x, h=1, mu=0, sigma=1.0, tau=0):

    if abs(tau) > VERY_SMALL_VALUE:
        z   = SQRT2 * ( sigma/tau - (x - mu)/sigma )
        z_neg   = z < 0
        z_mid   = np.logical_and( z >= 0, z <= Z_LIMIT )
        z_lrg   = z > Z_LIMIT

        ret_neg = h * sigma / tau * SQRTPI2 * np.exp( 0.5*(sigma/tau)**2 - (x[z_neg] - mu)/tau ) * erfc( z[z_neg] )
        ret_mid = h * np.exp( -0.5 * ( (x[z_mid] - mu)/sigma )**2 ) * sigma/tau * SQRTPI2 * erfcx( z[z_mid] )
        x_mu    = x[z_lrg] - mu
        ret_lrg = h * np.exp( -0.5 * ( x_mu/sigma )**2 ) / ( 1 + x_mu*tau/(sigma**2) )
        ret     = np.hstack( [ ret_lrg, ret_mid, ret_neg ] )
    else:
        x_mu    = x - mu
        ret     = h * np.exp( -0.5 * ( x_mu/sigma )**2 ) / ( 1 + x_mu*tau/(sigma**2) )

    return ret

def emg(x, h, mu, sigma, tau):
    # task: not sure if this works for a sclar x
    if tau > 0:
        ret = emg_orig( x, h, mu, sigma, tau )
    else:
        """
            reverse transformation is required
            because emg_orig assumes the ascending order of x
        """
        ret = emg_orig( (mu - x[::-1]), h, 0, sigma, -tau )[::-1]

    if False:
        print( 'tau=', tau )
        fig = dplt.figure()
        ax  = fig.add_subplot(111)
        ax.set_title( "emg" )
        ax.plot( x, ret )
        dplt.show()

    return ret
