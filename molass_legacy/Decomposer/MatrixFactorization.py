# coding: utf-8
"""

    MatrixFactorization.py

    Copyright (c) 2019, SAXS Team, KEK-PF

"""
import copy
import numpy as np
from scipy import optimize
from scipy.interpolate import UnivariateSpline
from mpl_toolkits.mplot3d import Axes3D
from molass_legacy.KekLib.SciPyCookbook import smooth
from ModeledData import ModeledData, simple_plot_3d
import molass_legacy.KekLib.DebugPlot as plt
from SvdDenoise import get_denoised_data

MAX_ITERATION_FMIN_CG   = 10000

def C_smooth_penalty_impl(C):
    penalty = 0
    for i in range(C.shape[0]):
        y = C[i,:]
        j = np.arange(C.shape[1]-1)
        penalty += np.sum((y[j] - y[j+1])**2)
    return penalty

def deriv_C_smooth_penalty_impl(C):
    penalty_list = []
    for i in range(C.shape[0]):
        y = C[i,:]
        j = np.arange(C.shape[1]-1)
        penalty_list.append(np.hstack([2*(y[j]-y[j+1]), [0]]))
    return np.hstack(penalty_list)

class MatrixFactorization:
    def __init__(self, M, index, ecurve):
        self.M = M
        self.index = index
        self.ecurve = ecurve

        n_angs = M.shape[0]
        n_elns = M.shape[1]

        P_init = self.make_P_init()
        C_init = self.make_C_init()

        if True:
            self.debug_plot(P_init, C_init, M, "Initial State")

        rank = len(self.ecurve.peak_info) + 1

        self.M_ = get_denoised_data(M, rank=rank)
        self.P_init = P = P_init
        self.C_init = C = C_init

        for k in range(3):
            P_ = self.solve_P(P, C)
            C_ = self.solve_C(P_, C)
            P = P_
            C = C_

            if True:
                self.debug_plot(P, C, self.M_, "After Iteration %d" % k)

    def make_P_init(self):
        P_list = []
        for rec in self.ecurve.peak_info:
            k = rec[1]
            P_list.append(smooth(self.M[:,k]))
        return np.array(P_list).T

    def make_C_init(self):
        C_list = []
        start = 0
        ey = self.ecurve.sy
        for p, rec in enumerate(self.ecurve.peak_info):
            k = rec[1]
            if p < len(self.ecurve.boundaries):
                stop = self.ecurve.boundaries[p]
            else:
                stop = len(ey)
            y = ey[start:stop]
            y_ = np.hstack([ np.zeros(start), y/np.max(y), np.zeros(len(ey)-stop) ])
            C_list.append(y_)
            start = stop
        return np.array(C_list)

    def debug_plot(self, P, C, M, title):
        fig = plt.figure(figsize=(14,7))
        fig.suptitle(title)

        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        simple_plot_3d(ax1, M, alpha=0.2)

        x = np.arange(M.shape[0])
        y = np.arange(M.shape[1])
        for p, rec in enumerate(self.ecurve.peak_info):
            k = rec[1]
            y_ = np.ones(len(x))*k
            z_ = P[:,p]
            ax1.plot(x, y_, z_, color='yellow')
            scale = np.max(z_)
            x_ = np.ones(len(y))*self.index
            z_ = C[p,:]*scale
            ax1.plot(x_, y, z_)
            ax2.plot(y, z_)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.show()

    def solve_P(self, P, C):

        self.C = C
        P_ = self.M_ @ np.linalg.pinv(C)

        self.num_iterations = 0
        def num_iterations_count(x):
            self.num_iterations += 1

        ret = optimize.fmin_cg( self.compute_P_error, P_, fprime=self.compute_deriv_P_error,
                maxiter=MAX_ITERATION_FMIN_CG,
                full_output=True,
                callback=num_iterations_count )

        P_flat = ret[0]
        return P_flat.reshape( self.P_init.shape )

    def compute_P_error(self, P_flat):
        P = P_flat.reshape( self.P_init.shape )
        reconstructed = np.dot( P, self.C )

        # compute a squared Frobenius norms
        rec_error = np.linalg.norm( reconstructed - self.M_ )**2
        return rec_error

    def compute_deriv_P_error(self, P_flat):
        P = P_flat.reshape( self.P_init.shape )
        reconstructed = np.dot( P, self.C )

        # compute the derivative of the squared Frobenius norm
        # || P@C - M ||**2 ==> 2*( P@C - M )@C.T
        d_rec = 2 * np.dot( reconstructed - self.M_, self.C.T ).flatten()
        return d_rec

    def solve_C(self, P, C):

        self.P = P

        self.num_iterations = 0
        def num_iterations_count(x):
            self.num_iterations += 1

        ret = optimize.fmin_cg( self.compute_C_error, C, fprime=self.compute_deriv_C_error,
                maxiter=MAX_ITERATION_FMIN_CG,
                full_output=True,
                callback=num_iterations_count )

        C_flat = ret[0]
        return C_flat.reshape( self.C_init.shape )

    def compute_C_error(self, C_flat):
        C = C_flat.reshape( self.C_init.shape )
        reconstructed = np.dot( self.P, C )

        # compute a squared Frobenius norms
        recon_error = np.linalg.norm( reconstructed - self.M_ )**2
        smooth_penalty = C_smooth_penalty_impl(C)
        monot_penalty = self.compute_C_monot_penalty(C)

        return recon_error + smooth_penalty + monot_penalty

    def compute_deriv_C_error(self, C_flat):
        C = C_flat.reshape( self.C_init.shape )
        reconstructed = np.dot( self.P, C )

        # compute the derivative of the squared Frobenius norm
        # || P@C - M ||**2 ==> 2*( P@C - M )@C.T
        d_recon_error = 2 * np.dot( self.P.T, reconstructed - self.M_ ).flatten()
        d_smooth_penalty = deriv_C_smooth_penalty_impl(C)
        d_monot_penalty = self.compute_deriv_C_monot_penalty(C)

        return d_recon_error + d_monot_penalty

    def compute_C_monot_penalty(self, C):
        x = self.ecurve.x
        penalty = 0
        for i, rec in enumerate(self.ecurve.peak_info):
            top_j = rec[1]

            y = C[i,:]
            spline = UnivariateSpline(x, y, s=0, ext=3)
            dspline = spline.derivative(1)
            dy = dspline(x)
            dy[np.logical_and(dy > 0, x < top_j)] = 0
            dy[np.logical_and(dy < 0, x > top_j)] = 0
            penalty += np.sum(dy**2)
        return penalty

    def compute_deriv_C_monot_penalty(self, C):
        x = self.ecurve.x
        penalty = 0
        deriv_list =[]
        for i, rec in enumerate(self.ecurve.peak_info):
            top_j = rec[1]

            y = C[i,:]
            spline = UnivariateSpline(x, y, s=0, ext=3)
            dspline = spline.derivative(1)
            dy = dspline(x)
            dy[np.logical_and(dy > 0, x < top_j)] = 0
            dy[np.logical_and(dy < 0, x > top_j)] = 0
            deriv_list.append(2*dy)

        return np.hstack(deriv_list)
