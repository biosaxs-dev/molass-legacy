"""
    LrfSolver.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from scipy.optimize import fmin_cg, minimize
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting
from .LrfInfoProxy import LrfInfoProxy

class LrfSolver:
    def __init__(self, pdata, popts, debug=False):
        if debug:
            from importlib import reload
            import molass_legacy.Extrapolation.ExtrapolationSolver
            reload(molass_legacy.Extrapolation.ExtrapolationSolver)   
        from molass_legacy.Extrapolation.ExtrapolationSolver import ExtrapolationSolver

        ExtrapolationSolver.__init__(self, pdata, popts)    # required in LrfResultPool
                                                            # for pno, paired_range in enumerate(solver.cnv_ranges)

        from importlib import reload
        import molass_legacy.LRF.PnoScdMap
        reload(molass_legacy.LRF.PnoScdMap)
        from .PnoScdMap import PnoScdMap

        self.xray_index = self.sd.xray_index
        self.logger = logging.getLogger(__name__)
        self.lrf_bound_correction = get_setting("lrf_bound_correction")
        self.logger.info("lrf_bound_correction=%d", self.lrf_bound_correction)
        self.impl = ExtrapolationSolver(pdata, popts)
        self.mc_vector = self.impl.mc_vector                # already scaled with conc_factor
        self.pno_map = PnoScdMap(pdata.sd, self.impl.cnv_ranges)

    def get_pno_map(self):
        return self.pno_map

    def solve_range(self, start, stop, peakset_info,
                    lrf_rank=1,
                    stop_event=None,
                    paired_range=None,
                    debug=False):

        self.logger.info("solve_range called with lrf_rank=%s", str(lrf_rank))

        done = False
        tried_bounded_lrf = False
        if self.lrf_bound_correction and lrf_rank >= 2:
            if debug:
                from importlib import reload
                import molass_legacy.BoundedLRF.BoundedLrfSolver
                reload(molass_legacy.BoundedLRF.BoundedLrfSolver)
                import molass_legacy.LRF.ElutionMatrix
                reload(molass_legacy.LRF.ElutionMatrix)
            from molass_legacy.BoundedLRF.BoundedLrfSolver import BoundedLrfSolver
            from .ElutionMatrix import ElutionMatrix

            tried_bounded_lrf = True
            try:
                em = ElutionMatrix(self.impl)
                C = em.make_elution_matrix(start, stop, peakset_info, lrf_rank=lrf_rank)
                if False:
                    np.savetxt("C-%d-%d.dat" % (start, stop), C)
                if len(C) >= 2:
                    C_ = C
                else:
                    pno, nth, peakset, known_peak_info = peakset_info
                    C_ = C[nth*2:(nth+1)*2]

                range_ = slice(start,stop)
                qv = self.qv
                D = self.data[:,range_]
                E = self.error[:,range_]

                i = self.xray_index
                j = paired_range.top_x
                Y = self.data[i,j]

                if debug:
                    debug_info = self.sd, range_
                else:
                    debug_info = None
                solver = BoundedLrfSolver(self.qv, D, E, C=C_, i=i, debug_info=debug_info)
                P_, C__, Rg, R_, L_, hK, hL, bq_bounds_, coerced_bq_ = solver.solve()
                aq_, bq_ = P_.T[0:2]

                Pe = solver.get_corrected_error()

                # scale = 1/self.impl.conc_factor
                top_c = self.mc_vector[j]
                Y_ = Y/top_c
                scale = Y_/aq_[i]

                Ae = Pe[:,0]*scale
                Be = Pe[:,1]*scale
                Z = None                # realy used?
                lrfE = [Ae, Be, Z]

                A = aq_*scale
                B = bq_*scale

                A_data = np.array([qv, A, Ae]).T
                B_ratio = np.max(B)/np.max(A)

                lrfinfo = LrfInfoProxy(A_data, need_bq=True, B_ratio=B_ratio, data=D, bq_bounds=(bq_bounds_[0]*scale, bq_bounds_[1]*scale))
                ret_C = C_              # is this backward compatible?

                if debug:
                    pass

                result = A, B, Z, lrfE, lrfinfo, ret_C
                done = True
            except:
                # task: identify this case for tests.
                from molass_legacy.KekLib.ExceptionTracebacker import warnlog_exception      # to avaid producing error messages
                warnlog_exception(self.logger, "BoundedLrfSolver failed: ")
                self.logger.info("resorting to the unbounded solver")

        if not done:
            if tried_bounded_lrf:
                self.logger.warning("bounded LRF failed. resorting to the traditional solver.")
            result =  self.impl.extrapolate_wiser(start, stop, peakset_info,
                                    stop_event=stop_event,
                                    lrf_rank=lrf_rank,
                                    )

        return result

class FrobeniusNorm:
    def __init__(self, M, C):
        self.M = M
        self.C = C
        self.p_shape = (M.shape[0], 2)

    def compute(self, x):
        P = x.reshape(self.p_shape)
        return np.linalg.norm(P @ self.C - self.M)**2

    def compute_gradient(self, x):
        P = x.reshape(self.p_shape)
        return ( 2 * ( (P @ self.C - self.M) @ self.C.T ) ).flatten()

class Regularization:
    def __init__(self, qv, R, L):
        self.qv = qv
        self.R = R
        self.L = L
        self.p_shape = (len(qv), 2)
        self.Kv = L**2/(qv*R)**4

    def compute(self, x):
        P = x.reshape(self.p_shape)
        p0 = P[:,0]
        p1 = P[:,1]
        dev = (p1/p0)**2 - self.Kv
        return np.sum(dev[dev > 0]**2)

    def compute_gradient(self, x):
        pass

def boundef_lrf(qv, M, C, Pinit, R, K=1):
    f = FrobeniusNorm(M, C)
    r = Regularization(qv, R, 1)
    Lam = 1

    def obj_func(x):
        return f.compute(x) + Lam*r.compute(x)

    def gradient_func(x):
        return f.compute_gradient(x) + Lam*r.compute_gradient(x)

    # ret = fmin_cg(obj_func, Pinit.flatten(), fprime=gradient_func)
    ret = minimize(obj_func, Pinit.flatten())
    return ret.x.reshape((len(qv),2))

def boundef_lrf_simple(qv, M, C, Pinit, R, K=1):
    p0 = Pinit[:,0].copy()
    p1 = Pinit[:,1].copy()
    L = 1
    bound = L/(qv*R)**2
    e = np.abs(p1/p0) - bound
    pos_e = e > 0
    d = (np.abs(p1[pos_e]) - bound[pos_e]*np.abs(p0[pos_e]))/(bound[pos_e]+1)
    p0 += d
    pospos_ep1 = np.logical_and(pos_e, p1 > 0)
    posneg_ep1 = np.logical_and(pos_e, p1 < 0)
    p1[pospos_ep1] -= (np.abs(p1[pospos_ep1]) - bound[pospos_ep1]*np.abs(p0[pospos_ep1]))/(bound[pospos_ep1]+1)
    p1[posneg_ep1] += (np.abs(p1[posneg_ep1]) - bound[posneg_ep1]*np.abs(p0[posneg_ep1]))/(bound[posneg_ep1]+1)
    return np.array([p0, p1]).T

def demo(root, sd, pno=0, debug=True):
    from bisect import bisect_right
    from molass_legacy.DataStructure.SvdDenoise import get_denoised_data
    from molass_legacy.Theory.Rg import compute_corrected_Rg
    from molass_legacy.Theory.SolidSphere import phi, get_boundary_params_simple
    from molass_legacy.Theory.SynthesizedLRF import synthesized_lrf_spike
    from molass_legacy.Theory.DjKinning1984 import S0
    from molass_legacy.Theory.SfBound import S0_bound

    M, E, qv, ecurve = sd.get_xr_data_separate_ly()

    range_ = ecurve.get_ranges_by_ratio(0.5)[pno]
    f = range_[0]
    p = range_[1]
    t = range_[2]

    pt = M[:,p]

    x = ecurve.x
    y = ecurve.y

    i = bisect_right(qv, 0.02)

    eslice = slice(f,t)
    M0 = M[:,eslice]
    Rg, gf, gt, _ = compute_corrected_Rg(sd, ecurve, pno, qv, M0, E[:,eslice])
    R = np.sqrt(5/3)*Rg

    norm_list = []

    c_ = y[eslice]

    M2 = get_denoised_data(M0, rank=2)
    C2 = np.array([c_, c_**2])
    P2 = M2 @ np.linalg.pinv(C2)

    norm_list.append(np.linalg.norm(P2@C2 - M0))

    b1, b2, k = get_boundary_params_simple(Rg)

    P_ = synthesized_lrf_spike(qv, M0, c_, M2, P2, boundary=b1, k=k)
    y1 = P_[:,0]
    y1n = y1/y1[0]

    Pb = boundef_lrf(qv, M2, C2, P_, R)

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(22,7))
    fig.suptitle(r"$\frac{L}{(qR)^2}$ Bounded LRF Demo", fontsize=30)
    ax0.set_title("I(q) = P(q)*S(q)", fontsize=16)
    ax1.set_title("P(q): Form Factors", fontsize=16)
    ax2.set_title("S(q): Structure Factors", fontsize=16)
    for ax in [ax0, ax1]:
        ax.set_yscale('log')

    py0 = phi(qv, R)**2
    sy0 = S0(qv, R*5.4/2, K=1)

    b1i = bisect_right(qv, b1)
    iy0 = py0*sy0
    ptyn = pt*iy0[b1i]/pt[b1i]

    ax0.plot(qv, iy0, color='green')
    ax0.plot(qv, ptyn)

    ax1.plot(qv, py0, color='green')
    ax1.plot(qv, y1n)
    ax1.plot(qv, Pb[:,0])

    s1y = 1 + P_[:,1]/y1
    ax2.plot(qv, sy0, color='green')
    ax2.plot(qv, s1y)
    ax2.plot(qv, 1 + Pb[:,1]/Pb[:,0])

    for j, ax in enumerate([ax0, ax1, ax2]):
        ymin, ymax = ax.get_ylim()
        ymax_ = 1.5 if j == 2 else ymax
        ax.set_ylim(ymin, ymax_)
        for b in [b1, b2]:
            ax.plot([b, b], [ymin, ymax_], ':', color='gray')

    R_ = R*5.4
    sbu = S0_bound(qv, R_, K=-2)
    sbl = S0_bound(qv, R_, K=2)
    ax2.plot(qv, sbu, ':', color='red')
    ax2.plot(qv, sbl, ':', color='red')


    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    plt.show()
