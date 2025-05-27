"""
    Optimizer.OptimalEgh.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
from time import time
import numpy as np
from scipy import optimize
from matplotlib.gridspec import GridSpec
import molass_legacy.KekLib.DebugPlot as plt
from SvdDenoise import get_denoised_data
from molass_legacy.Models.ElutionCurveModels import (EGHA, EGH, egh, egh_x_from_height_ratio,
                                EMGA, EMG, emg, emg_x_from_height_ratio)
from MatrixData import simple_plot_3d

# PARAMS = ['h', 'mu', 'sigma', 'tau']    # 'a' not used
PARAMS = np.arange(4)
NUM_PARAMS = len(PARAMS)
MAX_ITERATION_FMIN_CG = 100
MAX_ITERATION_FMIN = 1000
VERY_SMALL_VALUE = 1e-6
USE_FACTOED_WIEGHTS = True
USE_DOWNHILL_SIMPLEX = True

class OptimalElution:
    def __init__(self, M, ecurve, fit_recs, rank, error=None, debug=False):
        x = ecurve.x

        pv_init = self.make_param_vector(fit_recs)
        self.fit_recs = fit_recs

        self.M = M
        E = error

        C = self.make_C_matrix(rank, pv_init, x)
        self.C_init = C
        self.P = M @ np.linalg.pinv(C)

        W_ = 1/E.copy()
        W_[np.logical_not(np.isfinite(W_))] = 0
        W_[W_<0] = 0

        if USE_FACTOED_WIEGHTS:
            U, s, VT = np.linalg.svd(W_)
            # self.scale = np.sqrt(np.linalg.norm(M))
            # self.scale = np.sqrt(s[0])
            nM = np.linalg.norm(M)
            scale = 1/np.sqrt(nM)
            print("scale=", scale)
            pw = U[:,0:1]*scale
            cw = VT[0:1,:]*scale
            W = pw @ cw
            print("norm(M)=", nM, "norm(W)=", np.linalg.norm(W))
        else:
            pw = np.sum(W_, axis=1)[:,np.newaxis]
            cw = np.sum(W_, axis=0)[np.newaxis,:]
            W = pw @ cw

        if False:
            from MatrixData import simple_plot_3d
            print('pw.shape=', pw.shape, 'ce.shape=', cw.shape, 'W.shape=', W.shape)
            plt.push()
            fig = plt.figure(figsize=(14, 6))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
            ax1.set_title(r"Error Weights: $ \frac{1}{M_{error}} $", fontsize=32)
            ax2.set_title(r"Factored Weights: $ w_a @ w_e $", fontsize=32)
            simple_plot_3d(ax1, W_)
            simple_plot_3d(ax2, W)
            fig.tight_layout()
            plt.show()
            plt.pop()

        Mw = W*M
        h_indeces = np.arange(rank)*len(PARAMS)

        n = 0
        def objective_function(pv):
            nonlocal n
            # print('objective_function')
            C = self.make_C_matrix(rank, pv, x)
            Cw = C*cw
            Pw = Mw @ np.linalg.pinv(Cw)
            P = Pw/pw
            PC = P @ C
            D_ = W*(PC - M)
            v = np.linalg.norm(D_)
            if debug and n % 1000 == 0:
                print([n], 'v=', v)
                plt.push()
                gs = GridSpec(2,3)
                fig = plt.figure(figsize=(21,11))
                ax00 = fig.add_subplot(gs[0,0], projection='3d')
                ax01 = fig.add_subplot(gs[0,1], projection='3d')
                ax02 = fig.add_subplot(gs[0,2], projection='3d')
                ax10 = fig.add_subplot(gs[1,0], projection='3d')
                ax11 = fig.add_subplot(gs[1,1])
                ax12 = fig.add_subplot(gs[1,2])
                simple_plot_3d(ax00, M)
                simple_plot_3d(ax01, PC)
                simple_plot_3d(ax02, D_)
                simple_plot_3d(ax10, W)
                for k, p in enumerate(P.T):
                    ax11.plot(p)
                for k, c in enumerate(C):
                    ax12.plot(x, c)
                fig.tight_layout()
                plt.show()
                plt.pop()
            n += 1
            return v

        t0 = time()
        if USE_DOWNHILL_SIMPLEX:
            self.result = optimize.fmin(objective_function, pv_init,
                    maxiter=MAX_ITERATION_FMIN,
                    full_output=True)
        else:
            self.result = optimize.fmin_cg(objective_function, pv_init,
                    gtol=1e-7,
                    fprime=None,
                    maxiter=MAX_ITERATION_FMIN_CG,
                    full_output=True)

        print('I took ', time()-t0)

        self.x = x
        self.rank = rank

    def make_param_vector(self, fit_recs):
        param_list = []
        for rec in fit_recs:
            evaluator = rec.evaluator
            param_list.append([evaluator.get_param_value(p) for p in PARAMS])
        return np.array(param_list).flatten()

    def make_C_matrix(self, rank, pv, x):
        C_list = []
        pv_ = pv.reshape((rank, NUM_PARAMS))
        for h, mu, sigma, tau in pv_:
            C_list.append(self.model_func(x, h, mu, sigma, tau))
        C = np.array(C_list)
        return C

    def get_C_matrix(self):
        pv = self.result[0]
        return self.make_C_matrix(self.rank, pv, self.x)

class OptimalEgh(OptimalElution):
    def __init__(self, *args, **kwargs):
        self.model = EGH()
        self.model_func = egh
        self.model_foot = egh_x_from_height_ratio
        OptimalElution.__init__(self, *args, **kwargs)

class OptimalEmg(OptimalElution):
    def __init__(self, *args, **kwargs):
        self.model = EMG()
        self.model_func = emg
        self.model_foot = emg_x_from_height_ratio
        OptimalElution.__init__(self, *args, **kwargs)

def get_h_vector(fit_recs):
    h_vector = np.zeros(len(fit_recs))
    for k, rec in enumerate(fit_recs):
        evaluator = rec.evaluator
        h = evaluator.get_param_value(0)
        h_vector[k] = h

    max_h = np.max(h_vector)
    rank_guess = len(np.where(h_vector > max_h*0.05 )[0] )
    return h_vector, rank_guess

def select_resc(h_vector, rank, fit_recs):
    n = len(fit_recs) - rank
    pp = np.argpartition(h_vector, n)
    return [fit_recs[i] for i in sorted(pp[n:])]

def compute_optimal_elution(D, E, ecurve, model_type, legacy_info=None, rank=None, logger=None, debug=False):
    model = EGHA() if model_type else EMGA()

    if legacy_info is None:
        from CurveDecomposer import decompose
        fit_recs = decompose(ecurve, model=model)
        h_vector, rank_guess = get_h_vector(fit_recs)
    else:
        from DecompUtils import decompose_elution_better
        sd, mapper, corbase_info = legacy_info
        ret = decompose_elution_better(corbase_info, mapper, model,
                                    logger=logger)
        fit_recs = ret.opt_recs 
        h_vector, rank_guess = get_h_vector(fit_recs)

    if rank is None:
        rank = rank_guess

    fit_recs = select_resc(h_vector, rank, fit_recs)

    M = get_denoised_data(D, rank=rank)      # M

    if model_type:
        opimizer = OptimalEgh(M, ecurve, fit_recs, rank, error=E, debug=debug)
    else:
        opimizer = OptimalEmg(M, ecurve, fit_recs, rank, error=E, debug=debug)
    optC = opimizer.get_C_matrix()
    recs = opimizer.fit_recs

    return M, rank, opimizer.C_init, optC, recs
