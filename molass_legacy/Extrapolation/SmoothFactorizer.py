# coding: utf-8
"""
    SmoothFactorizer.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
from scipy import optimize
from SmoothnessPenalty import SmoothnessPenalty
MAX_ITERATION_FMIN_CG   = 10000

class SmoothFactorizer:
    def __init__(self, M, E, c):
        self.M  = M
        self.E  = E
        self.c  = c

    def solve(self, with_const=False, apositive=False, smoothed=False):
        self.apositive = apositive
        self.smoothed = smoothed
        self.ignore_bq =False
        self.penalty_weights = [ 0.5, 0.01, 1.0, 0.1 ]
        c_ = self.c
        c_list = [c_, c_**2]
        if with_const:
            c_list.append( np.ones(len(c_)) )

        self.C = np.array( c_list )
        self.Cpinv = np.linalg.pinv(self.C)
        self.P_init = np.dot(self.M, self.Cpinv)
        # self.e = np.sqrt( np.dot(self.E**2, self.Cpinv**2) )
        # print( self.P.shape, self.e.shape )

        self.sp = SmoothnessPenalty(self.P_init.shape[0])

        self.Zero = np.zeros( self.P_init.shape )
        n = self.P_init.shape[1]
        self.P2A = np.zeros( (n, n) )
        for i in range(0, n, 2):
            self.P2A[i,i] = 1

        self.P = self.mimimize()

    def compute_error( self, P_flat ):
        P = P_flat.reshape( self.P_init.shape )
        reconstructed = np.dot( P, self.C )

        # compute a squared Frobenius norms
        rec_error = np.linalg.norm( reconstructed - self.M )**2

        if self.apositive:
            A   = np.dot( P, self.P2A )
            negA = np.min( [ self.Zero, A ], axis=0 )
            neg_error = np.linalg.norm( negA )**2
        else:
            neg_error = 0
        error   = rec_error + neg_error*self.penalty_weights[0]

        if self.smoothed:
            if self.ignore_bq:
                penalties = self.sp.get_penalties_bq_ignore(P_flat, self.penalty_weights[2])
                error += np.sum(penalties)
            else:
                penalties = self.sp.get_penalties(P_flat, self.penalty_weights[2:])
                error += np.sum(penalties)

        return error

    def compute_error_grad( self, P_flat ):
        P = P_flat.reshape( self.P_init.shape )
        reconstructed = np.dot( P, self.C )

        # compute the derivative of the squared Frobenius norm
        # || P@C - D ||**2 ==> 2*( P@C - D )@C.T
        rec_grad = 2 * np.dot( reconstructed - self.M, self.C.T )

        if self.apositive:
            A   = np.dot( P, self.P2A )
            negA = np.min( [ self.Zero, A ], axis=0 )
            neg_grad = 2 * negA
            neg_grad_ = neg_grad.flatten()*self.penalty_weights[0]
        else:
            neg_grad_ = 0

        error_grad  = rec_grad.flatten() + neg_grad_

        if self.smoothed:
            if self.ignore_bq:
                error_grad += self.sp.get_penalty_diff_bq_ignore(P_flat, self.penalty_weights[2])
            else:
                error_grad += self.sp.get_penalty_diff(P_flat, self.penalty_weights[2:])

        return error_grad

    def mimimize( self ):
        ret = optimize.fmin_cg( self.compute_error, self.P_init.flatten(), fprime=self.compute_error_grad,
                maxiter=MAX_ITERATION_FMIN_CG,
                full_output=True )
        P_flat = ret[0]
        return P_flat.reshape(self.P_init.shape)
